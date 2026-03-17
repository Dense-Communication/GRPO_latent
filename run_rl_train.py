"""
RL Training Script for Reading Policy in LatentMAS.

Trains a reading policy network using GRPO (Group Relative Policy Optimization).
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch
from tqdm import tqdm

from data import (
    load_gsm8k,
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa,
    load_winogrande,
)
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy import ReadingPolicyNetwork
from training import GRPOTrainer, RewardCalculator, Transition
from utils import auto_device, set_seed


def evaluate(method: LatentMASMethodRL, dataset: List[Dict], args, verbose: bool = False) -> Dict[str, float]:
    """
    Evaluate the method on a dataset.

    Returns:
        Dict with accuracy and efficiency metrics for comparison experiments.
    """
    method.rl_training = False
    if method.reading_policy is not None:
        method.reading_policy.eval()

    correct = 0
    total = 0

    # Efficiency tracking
    total_blocks_sum = 0
    selected_blocks_sum = 0
    selection_ratios = []

    for item in tqdm(dataset, desc="Evaluating"):
        # Use vLLM path if available, otherwise use standard run_item
        if args.use_vllm:
            result = method.run_batch_vllm([item])[0]
        else:
            result = method.run_item(item)

        if result.get("correct", False):
            correct += 1
        total += 1

        # Collect efficiency stats
        stats = method.get_efficiency_stats()
        total_blocks_sum += stats["total_blocks"]
        selected_blocks_sum += stats["selected_blocks"]
        if stats["total_blocks"] > 0:
            selection_ratios.append(stats["selection_ratio"])

    method.rl_training = True

    accuracy = correct / total if total > 0 else 0.0
    avg_selection_ratio = sum(selection_ratios) / len(selection_ratios) if selection_ratios else 1.0

    eval_results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "total_blocks_sum": total_blocks_sum,
        "selected_blocks_sum": selected_blocks_sum,
        "avg_selection_ratio": avg_selection_ratio,
        "latent_reduction_pct": (1 - avg_selection_ratio) * 100,
        "has_reading_policy": method.reading_policy is not None,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("Evaluation Results (for comparison experiments)")
        print("=" * 50)
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Total blocks processed: {total_blocks_sum}")
        print(f"  Selected blocks used: {selected_blocks_sum}")
        print(f"  Avg selection ratio: {avg_selection_ratio:.4f}")
        print(f"  Latent communication reduction: {eval_results['latent_reduction_pct']:.1f}%")
        print(f"  Reading policy enabled: {eval_results['has_reading_policy']}")
        print("=" * 50 + "\n")

    return eval_results


def evaluate_baseline(method: LatentMASMethodRL, dataset: List[Dict], args, verbose: bool = False) -> Dict[str, float]:
    """
    Evaluate WITHOUT reading policy (baseline - uses all blocks).

    This provides the baseline accuracy for comparison:
    - Baseline: No reading policy, uses 100% of latent blocks
    - With Policy: Reading policy selects top-k blocks

    Returns:
        Dict with accuracy and efficiency metrics.
    """
    # Temporarily disable reading policy
    original_policy = method.reading_policy
    method.reading_policy = None
    method.rl_training = False

    correct = 0
    total = 0
    total_blocks_sum = 0

    for item in tqdm(dataset, desc="Baseline Eval (no policy)"):
        if args.use_vllm:
            result = method.run_batch_vllm([item])[0]
        else:
            result = method.run_item(item)

        if result.get("correct", False):
            correct += 1
        total += 1

        # Track total blocks (all are used in baseline)
        stats = method.get_efficiency_stats()
        total_blocks_sum += stats["total_blocks"]

    # Restore reading policy
    method.reading_policy = original_policy
    method.rl_training = True

    accuracy = correct / total if total > 0 else 0.0

    eval_results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "total_blocks_sum": total_blocks_sum,
        "selected_blocks_sum": total_blocks_sum,  # All blocks used
        "avg_selection_ratio": 1.0,  # 100% of blocks used
        "latent_reduction_pct": 0.0,  # No reduction
        "has_reading_policy": False,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("BASELINE Results (NO reading policy - uses ALL blocks)")
        print("=" * 50)
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Total blocks: {total_blocks_sum}")
        print(f"  Selection ratio: 100% (all blocks used)")
        print(f"  Latent reduction: 0% (baseline)")
        print("=" * 50 + "\n")

    return eval_results


def train_epoch(
    method: LatentMASMethodRL,
    trainer: GRPOTrainer,
    dataset: List[Dict],
    args,
) -> Dict[str, float]:
    """Train for one epoch using GRPO."""
    method.rl_training = True
    method.reading_policy.train()

    epoch_rewards = []
    epoch_task_rewards = []
    epoch_consistency_rewards = []
    epoch_cost_penalties = []

    # Collect trajectories in groups
    current_group = []
    groups_collected = 0

    progress = tqdm(dataset, desc="Training")
    for item in progress:
        # Run the method
        if args.use_vllm:
            result = method.run_batch_vllm([item])[0]
        else:
            result = method.run_item(item)

        # Compute reward
        trajectory_info = method.get_trajectory_info()
        if trajectory_info is None:
            continue

        # Get reward components
        total_reward, components = trainer.reward_calculator.compute_total_reward(
            prediction=result.get("prediction"),
            gold=result.get("gold"),
            task_type=args.task,
            num_selected_blocks=args.top_k_blocks,
            total_blocks=trajectory_info.get("num_blocks", 0),
            correct_override=result.get("correct"),
        )

        # Create transition
        transition = method.create_transition(
            reward=total_reward,
            task_reward=components["task_reward"],
            consistency_reward=components["consistency_reward"],
            cost_penalty=components["cost_penalty"],
        )

        if transition is not None:
            current_group.append(transition)
            epoch_rewards.append(total_reward)
            epoch_task_rewards.append(components["task_reward"])
            epoch_consistency_rewards.append(components["consistency_reward"])
            epoch_cost_penalties.append(components["cost_penalty"])

        # Check if group is complete
        if len(current_group) >= args.grpo_group_size:
            trainer.buffer.groups.append(current_group)
            current_group = []
            groups_collected += 1

            # Update policy periodically
            if groups_collected >= args.update_every_n_groups:
                stats = trainer.update_policy(trainer.buffer.groups)
                trainer.buffer.groups = []
                groups_collected = 0

                # Update reference policy periodically
                if trainer.should_update_reference(args.ref_policy_update_freq):
                    trainer.update_reference_policy()

                progress.set_postfix({
                    "loss": f"{stats['total_loss']:.4f}",
                    "reward": f"{sum(epoch_rewards[-100:])/max(1,len(epoch_rewards[-100:])):.3f}",
                })

    # Handle remaining group
    if current_group:
        trainer.buffer.groups.append(current_group)
        if trainer.buffer.groups:
            trainer.update_policy(trainer.buffer.groups)
            trainer.buffer.groups = []

    # Compute epoch statistics
    stats = {
        "mean_reward": sum(epoch_rewards) / max(1, len(epoch_rewards)),
        "mean_task_reward": sum(epoch_task_rewards) / max(1, len(epoch_task_rewards)),
        "mean_consistency_reward": sum(epoch_consistency_rewards) / max(1, len(epoch_consistency_rewards)),
        "mean_cost_penalty": sum(epoch_cost_penalties) / max(1, len(epoch_cost_penalties)),
        "num_samples": len(epoch_rewards),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="RL Training for Reading Policy")

    # Core arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model to use (e.g., 'Qwen/Qwen3-4B')")
    parser.add_argument("--task", type=str, default="gsm8k",
                        choices=["gsm8k", "aime2024", "aime2025", "gpqa",
                                "arc_easy", "arc_challenge", "mbppplus",
                                "humanevalplus", "medqa", "winogrande"],
                        help="Task to train on")
    parser.add_argument("--prompt", type=str, default="sequential",
                        choices=["sequential", "hierarchical"],
                        help="Multi-agent architecture")
    parser.add_argument("--method", type=str, default="latent_mas",
                        help="Method type (for compatibility with ModelWrapper)")

    # Model arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--use_second_HF_model", action="store_true")

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")

    # Reading Policy arguments
    parser.add_argument("--top_k_blocks", type=int, default=4,
                        help="Number of blocks to select")
    parser.add_argument("--policy_num_heads", type=int, default=8)
    parser.add_argument("--policy_num_layers", type=int, default=2)
    parser.add_argument("--policy_dropout", type=float, default=0.1)
    parser.add_argument("--policy_checkpoint", type=str, default=None,
                        help="Load pretrained policy checkpoint")

    # Semantic segmentation arguments
    parser.add_argument("--similarity_threshold", type=float, default=0.85)
    parser.add_argument("--min_block_size", type=int, default=4)
    parser.add_argument("--max_block_size", type=int, default=64)
    parser.add_argument("--segment_layer_idx", type=int, default=16)

    # GRPO training arguments
    parser.add_argument("--policy_lr", type=float, default=1e-5)
    parser.add_argument("--grpo_group_size", type=int, default=8)
    parser.add_argument("--grpo_clip_epsilon", type=float, default=0.2)
    parser.add_argument("--grpo_kl_coef", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--ref_policy_update_freq", type=int, default=100)
    parser.add_argument("--update_every_n_groups", type=int, default=4)

    # Reward arguments
    parser.add_argument("--reward_alpha", type=float, default=1.0,
                        help="Task correctness weight")
    parser.add_argument("--reward_beta", type=float, default=0.5,
                        help="Evidence consistency weight")
    parser.add_argument("--reward_gamma", type=float, default=0.1,
                        help="Read cost penalty weight")

    # Training control
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/rl_policy")
    parser.add_argument("--log_file", type=str, default=None)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-configure for vLLM
    if args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True

    device = auto_device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model_name}")
    model = ModelWrapper(
        args.model_name,
        device,
        use_vllm=args.use_vllm,
        args=args,
    )

    # Initialize reading policy
    hidden_dim = model.get_hidden_dim()
    print(f"Model hidden dim: {hidden_dim}")

    policy_net = ReadingPolicyNetwork(
        hidden_dim=hidden_dim,
        num_heads=args.policy_num_heads,
        num_layers=args.policy_num_layers,
        dropout=args.policy_dropout,
    ).to(device).to(torch.bfloat16)  # Use bfloat16 to match model output dtype
    print(f"Policy network dtype: {next(policy_net.parameters()).dtype}")

    # Load checkpoint if provided
    if args.policy_checkpoint:
        print(f"Loading policy checkpoint: {args.policy_checkpoint}")
        policy_net.load_state_dict(torch.load(args.policy_checkpoint, map_location=device))

    # Create method
    method = LatentMASMethodRL(
        model,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=1,  # Use batch size 1 for RL training
        args=args,
        reading_policy=policy_net,
        top_k_blocks=args.top_k_blocks,
        rl_training=True,
    )

    # Create optimizer and trainer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.policy_lr)

    reward_calculator = RewardCalculator(
        alpha=args.reward_alpha,
        beta=args.reward_beta,
        gamma=args.reward_gamma,
    )

    trainer = GRPOTrainer(
        policy_net=policy_net,
        optimizer=optimizer,
        reward_calculator=reward_calculator,
        group_size=args.grpo_group_size,
        clip_epsilon=args.grpo_clip_epsilon,
        kl_coef=args.grpo_kl_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=device,
    )

    # Load datasets
    print(f"Loading {args.task} dataset...")
    if args.task == "gsm8k":
        train_data = list(load_gsm8k(split="train"))
    elif args.task == "aime2024":
        train_data = list(load_aime2024(split="train"))
    elif args.task == "aime2025":
        train_data = list(load_aime2025(split="train"))
    elif args.task == "gpqa":
        train_data = list(load_gpqa_diamond(split="test"))
    elif args.task == "arc_easy":
        train_data = list(load_arc_easy(split="train"))  # Use train split (2251 samples)
    elif args.task == "arc_challenge":
        train_data = list(load_arc_challenge(split="train"))  # Use train split (1119 samples)
    elif args.task == "mbppplus":
        train_data = list(load_mbppplus(split="test"))
    elif args.task == "humanevalplus":
        train_data = list(load_humanevalplus(split="test"))
    elif args.task == "medqa":
        train_data = list(load_medqa(split="test"))
    elif args.task == "winogrande":
        train_data = list(load_winogrande(split="train"))  # Use train split (9248 samples)

    # Limit samples if specified
    if args.max_samples > 0:
        train_data = train_data[:args.max_samples]

    # Split into train/eval
    eval_data = train_data[:args.eval_samples]
    train_data = train_data[args.eval_samples:]

    print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

    # ========================================
    # Run BASELINE evaluation first (no reading policy)
    # This gives us the reference accuracy to compare against
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 1: Running BASELINE evaluation (NO reading policy)")
    print("        This uses 100% of latent blocks for comparison")
    print("=" * 60)

    baseline_results = evaluate_baseline(method, eval_data, args, verbose=True)

    print("\n" + "=" * 60)
    print("STEP 2: Starting RL training WITH reading policy")
    print(f"        Target: Match baseline accuracy ({baseline_results['accuracy']:.2%})")
    print(f"                while reducing latent communication")
    print("=" * 60)

    # Training loop
    best_accuracy = 0.0
    training_log = []

    start_time = time.time()

    for epoch in range(args.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*50}")

        # Train
        train_stats = train_epoch(method, trainer, train_data, args)
        print(f"Train stats: {json.dumps(train_stats, indent=2)}")

        # Evaluate
        if eval_data:
            eval_results = evaluate(method, eval_data, args, verbose=True)
            accuracy = eval_results["accuracy"]
        else:
            accuracy = train_stats.get("mean_task_reward", 0.0)
            eval_results = {"accuracy": accuracy}

        # Log
        epoch_log = {
            "epoch": epoch + 1,
            "train_stats": train_stats,
            "eval_accuracy": accuracy,
            "eval_results": eval_results,  # Full efficiency metrics
            "global_step": trainer.global_step,
        }
        training_log.append(epoch_log)

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"policy_epoch{epoch + 1}.pt")
        trainer.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_path = os.path.join(args.output_dir, "policy_best.pt")
            torch.save(policy_net.state_dict(), best_path)
            print(f"New best model saved: {best_path} (accuracy: {best_accuracy:.4f})")

    # Training complete
    total_time = time.time() - start_time

    # Final comparison summary
    final_eval = training_log[-1]["eval_results"] if training_log else {}
    final_accuracy = final_eval.get("accuracy", 0.0)
    final_reduction = final_eval.get("latent_reduction_pct", 0.0)
    baseline_accuracy = baseline_results["accuracy"]

    accuracy_diff = final_accuracy - baseline_accuracy
    accuracy_diff_pct = (accuracy_diff / baseline_accuracy * 100) if baseline_accuracy > 0 else 0

    print("\n" + "=" * 60)
    print("FINAL COMPARISON: Baseline vs With Reading Policy")
    print("=" * 60)
    print(f"  BASELINE (no policy):     {baseline_accuracy:.2%} accuracy, 0% reduction")
    print(f"  WITH READING POLICY:      {final_accuracy:.2%} accuracy, {final_reduction:.1f}% reduction")
    print("-" * 60)
    print(f"  Accuracy change:          {accuracy_diff:+.2%} ({accuracy_diff_pct:+.1f}%)")
    print(f"  Latent communication:     {final_reduction:.1f}% SAVED")
    print("=" * 60)
    print(f"Training complete! Total time: {total_time:.2f}s")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print("=" * 60)

    # Save training log with baseline comparison
    log_path = args.log_file or os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "args": vars(args),
            "baseline_results": baseline_results,  # Baseline for comparison
            "training_log": training_log,
            "best_accuracy": best_accuracy,
            "total_time_sec": total_time,
            "comparison": {
                "baseline_accuracy": baseline_accuracy,
                "final_accuracy": final_accuracy,
                "accuracy_change": accuracy_diff,
                "accuracy_change_pct": accuracy_diff_pct,
                "latent_reduction_pct": final_reduction,
            }
        }, f, indent=2)
    print(f"Training log saved: {log_path}")


if __name__ == "__main__":
    main()
