#!/usr/bin/env python3
"""
Reward Function Ablation Study.

Compare different reward weight combinations:
- R = α*R_task + β*R_consistency - γ*R_cost

Configurations:
1. Task-only: α=1, β=0, γ=0
2. Consistency-only: α=0, β=1, γ=0
3. Balanced (no cost): α=0.5, β=0.5, γ=0 (V4 default)
4. Task-focused: α=0.7, β=0.3, γ=0
5. With cost penalty: α=0.5, β=0.5, γ=0.1
"""

import os
import sys
sys.path.insert(0, '/p/scratch/westai0052/liu52/LatentMAS')

import argparse
import torch
import json
from datetime import datetime

from data import load_gsm8k, load_arc_challenge, load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy.reading_policy import ReadingPolicyNetwork
from utils import set_seed

set_seed(42)


def run_reward_ablation(
    task: str,
    alpha: float,
    beta: float,
    gamma: float,
    config_name: str,
    top_k: int = 40,
    train_samples: int = 500,
    test_samples: int = 500,
    training_steps: int = 300,
):
    """
    Train and evaluate policy with specific reward weights.
    """
    print("=" * 60)
    print(f"REWARD ABLATION: {task} - {config_name}")
    print(f"  α={alpha}, β={beta}, γ={gamma}, top_k={top_k}")
    print("=" * 60)

    # Setup
    MODEL_PATH = "/p/scratch/westai0052/liu52/models/Qwen3-4B"
    SEGMENT_LAYER_IDX = 16
    HIDDEN_DIM = 2560  # Qwen3-4B hidden size

    # Create args object
    class Args:
        pass

    args = Args()
    args.model_name = MODEL_PATH
    args.prompt = 'sequential'
    args.method = 'latent_mas'
    args.device = 'cuda:0'
    args.device2 = 'cuda:1'
    args.think = False
    args.latent_steps = 3
    args.top_k_blocks = top_k
    args.similarity_threshold = 0.85
    args.min_block_size = 4
    args.max_block_size = 64
    args.segment_layer_idx = SEGMENT_LAYER_IDX
    args.tensor_parallel_size = 1
    args.gpu_memory_utilization = 0.85
    args.enable_prefix_caching = True
    args.use_second_HF_model = True
    args.latent_space_realign = False
    args.max_new_tokens = 512
    args.task = task

    # Load model
    print('Loading model...')
    sys.stdout.flush()
    model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)

    # Create policy
    print(f'Creating policy...')
    policy = ReadingPolicyNetwork(
        hidden_dim=HIDDEN_DIM,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        use_layer_norm=True,
    ).to(args.device2)

    # Load data
    print(f'\nLoading data...')
    if task == "gsm8k":
        train_data = list(load_gsm8k(split="train"))[:train_samples]
        test_data = list(load_gsm8k(split="test"))[:test_samples]
    elif task == "arc_challenge":
        train_data = list(load_arc_challenge(split="train"))[:train_samples]
        test_data = list(load_arc_challenge(split="test"))[:test_samples]
    elif task == "winogrande":
        train_data = list(load_winogrande(split="train"))[:train_samples]
        test_data = list(load_winogrande(split="validation"))[:test_samples]
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f'Train: {len(train_data)}, Test: {len(test_data)}')
    sys.stdout.flush()

    # ========== TRAIN ==========
    print('\n' + '=' * 60)
    print(f'TRAINING: {config_name}')
    print('=' * 60)

    train_method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=policy, top_k_blocks=top_k, rl_training=True,
    )
    train_method.task = task

    # Train with specific reward weights
    train_method.train_policy_grpo(
        train_data,
        epochs=1,
        steps_per_epoch=training_steps,
        lr=1e-4,
        group_size=4,
        reward_alpha=alpha,
        reward_beta=beta,
        reward_gamma=gamma,
    )

    # ========== TEST BASELINE ==========
    print('\n' + '=' * 60)
    print('BASELINE (read ALL blocks)')
    print('=' * 60)

    baseline_method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=None, top_k_blocks=top_k, rl_training=False,
    )
    baseline_method.task = task

    baseline_correct = 0
    for i, item in enumerate(test_data):
        result = baseline_method.run_batch_vllm([item])[0]
        if result.get('correct', False):
            baseline_correct += 1
        if (i + 1) % 100 == 0:
            print(f'  Baseline: {i+1}/{test_samples}, acc={baseline_correct}/{i+1}')
            sys.stdout.flush()

    baseline_acc = baseline_correct / test_samples

    # ========== TEST POLICY ==========
    print('\n' + '=' * 60)
    print(f'POLICY: {config_name}')
    print('=' * 60)

    policy.eval()
    test_method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=policy, top_k_blocks=top_k, rl_training=False,
    )
    test_method.task = task

    policy_correct = 0
    policy_total_blocks = 0
    policy_selected_blocks = 0

    for i, item in enumerate(test_data):
        result = test_method.run_batch_vllm([item])[0]
        if result.get('correct', False):
            policy_correct += 1

        stats = test_method.get_efficiency_stats()
        policy_total_blocks += stats['total_blocks']
        policy_selected_blocks += stats['selected_blocks']

        if (i + 1) % 100 == 0:
            print(f'  Policy: {i+1}/{test_samples}, acc={policy_correct}/{i+1}')
            sys.stdout.flush()

    policy_acc = policy_correct / test_samples
    read_ratio = policy_selected_blocks / max(policy_total_blocks, 1)
    latent_reduction = (1 - read_ratio) * 100
    acc_change = (policy_acc - baseline_acc) * 100

    # ========== RESULTS ==========
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'Task: {task}')
    print(f'Config: {config_name}')
    print(f'Reward: α={alpha}, β={beta}, γ={gamma}')
    print(f'top_k: {top_k}')
    print(f'')
    print(f'Baseline:  {baseline_acc*100:.2f}% (read 100%)')
    print(f'Policy:    {policy_acc*100:.2f}% (read {read_ratio*100:.2f}%)')
    print(f'')
    print(f'Accuracy change: {acc_change:+.2f}%')
    print(f'Latent reduction: {latent_reduction:.1f}%')
    print('=' * 60)

    # Save results
    results = {
        'task': task,
        'config_name': config_name,
        'reward_weights': {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
        },
        'top_k': top_k,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'training_steps': training_steps,
        'baseline_acc': baseline_acc,
        'policy_acc': policy_acc,
        'read_ratio': read_ratio,
        'accuracy_change_pct': acc_change,
        'latent_reduction_pct': latent_reduction,
        'timestamp': datetime.now().isoformat(),
    }

    os.makedirs('logs/reward_ablation', exist_ok=True)
    safe_name = config_name.replace(' ', '_').replace(':', '_')
    result_file = f'logs/reward_ablation/{task}_{safe_name}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {result_file}')

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['gsm8k', 'arc_challenge', 'winogrande'])
    parser.add_argument('--config', type=str, required=True,
                        choices=['task_only', 'consistency_only', 'balanced',
                                 'task_focused', 'with_cost'])
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--train_samples', type=int, default=500)
    parser.add_argument('--test_samples', type=int, default=500)
    parser.add_argument('--training_steps', type=int, default=300)

    args = parser.parse_args()

    # Define reward configurations
    CONFIGS = {
        'task_only': (1.0, 0.0, 0.0, 'Task Only (α=1, β=0, γ=0)'),
        'consistency_only': (0.0, 1.0, 0.0, 'Consistency Only (α=0, β=1, γ=0)'),
        'balanced': (0.5, 0.5, 0.0, 'Balanced (α=0.5, β=0.5, γ=0)'),
        'task_focused': (0.7, 0.3, 0.0, 'Task Focused (α=0.7, β=0.3, γ=0)'),
        'with_cost': (0.5, 0.5, 0.1, 'With Cost (α=0.5, β=0.5, γ=0.1)'),
    }

    alpha, beta, gamma, config_name = CONFIGS[args.config]

    run_reward_ablation(
        task=args.task,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        config_name=config_name,
        top_k=args.top_k,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        training_steps=args.training_steps,
    )
