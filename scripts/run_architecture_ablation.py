#!/usr/bin/env python3
"""
Policy Architecture Ablation Study.

Compare different policy network configurations:
- num_layers: [1, 2, 3, 4]
- num_heads: [4, 8]
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


def run_architecture_ablation(
    task: str,
    num_layers: int,
    num_heads: int = 8,
    top_k: int = 40,
    train_samples: int = 500,
    test_samples: int = 500,
    training_steps: int = 300,
):
    """
    Train and evaluate policy with specific architecture.
    """
    print("=" * 60)
    print(f"ARCHITECTURE ABLATION: {task}")
    print(f"  num_layers={num_layers}, num_heads={num_heads}, top_k={top_k}")
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

    # Create policy with specific architecture
    print(f'Creating policy: layers={num_layers}, heads={num_heads}')
    policy = ReadingPolicyNetwork(
        hidden_dim=HIDDEN_DIM,
        num_heads=num_heads,
        num_layers=num_layers,
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
    print('TRAINING')
    print('=' * 60)

    train_method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=policy, top_k_blocks=top_k, rl_training=True,
    )
    train_method.task = task

    # V4 reward config: gamma=0
    train_method.train_policy_grpo(
        train_data,
        epochs=1,
        steps_per_epoch=training_steps,
        lr=1e-4,
        group_size=4,
        reward_alpha=0.5,
        reward_beta=0.5,
        reward_gamma=0.0,  # No cost penalty
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
    print(f'POLICY (layers={num_layers}, heads={num_heads})')
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
    print(f'Architecture: layers={num_layers}, heads={num_heads}')
    print(f'top_k: {top_k}')
    print(f'')
    print(f'Baseline:  {baseline_acc*100:.2f}% (read 100%)')
    print(f'Policy:    {policy_acc*100:.2f}% (read {read_ratio*100:.2f}%)')
    print(f'')
    print(f'Accuracy change: {acc_change:+.2f}%')
    print(f'Latent reduction: {latent_reduction:.1f}%')
    print('=' * 60)

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())

    # Save results
    results = {
        'task': task,
        'architecture': {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'hidden_dim': HIDDEN_DIM,
            'num_params': num_params,
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

    os.makedirs('logs/architecture', exist_ok=True)
    result_file = f'logs/architecture/{task}_layers{num_layers}_heads{num_heads}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {result_file}')

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['gsm8k', 'arc_challenge', 'winogrande'])
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--train_samples', type=int, default=500)
    parser.add_argument('--test_samples', type=int, default=500)
    parser.add_argument('--training_steps', type=int, default=300)

    args = parser.parse_args()

    run_architecture_ablation(
        task=args.task,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        top_k=args.top_k,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        training_steps=args.training_steps,
    )
