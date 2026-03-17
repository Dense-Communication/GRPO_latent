#!/usr/bin/env python3
"""
Evaluate heuristic baseline methods for block selection.

This provides baselines to compare with the learned reading policy.
"""

import os
import sys
sys.path.insert(0, '/p/scratch/westai0052/liu52/LatentMAS')

import torch
import json
from datetime import datetime
from typing import List

from data import load_gsm8k, load_arc_challenge, load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from methods.heuristic_selection import get_heuristic_selector
from utils import set_seed

set_seed(42)


def evaluate_heuristic(
    task: str,
    method: str,
    top_k: int,
    test_samples: int = 500,
):
    """
    Evaluate a heuristic selection method.

    Args:
        task: Dataset name ('gsm8k', 'arc_challenge', 'winogrande')
        method: Heuristic method ('random', 'recency', 'similarity', 'time_weighted')
        top_k: Number of blocks to select
        test_samples: Number of test samples to evaluate
    """
    print("=" * 60)
    print(f"HEURISTIC EVALUATION: {task} - {method} - top_k={top_k}")
    print("=" * 60)

    # Setup
    MODEL_PATH = "/p/scratch/westai0052/liu52/models/Qwen3-4B"
    SEGMENT_LAYER_IDX = 16

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

    # Load test data
    print(f'\nLoading TEST data...')
    if task == "gsm8k":
        test_data = list(load_gsm8k(split="test"))[:test_samples]
    elif task == "arc_challenge":
        test_data = list(load_arc_challenge(split="test"))[:test_samples]
    elif task == "winogrande":
        test_data = list(load_winogrande(split="validation"))[:test_samples]
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f'Loaded {len(test_data)} test samples')
    sys.stdout.flush()

    # ========== BASELINE (read ALL blocks) ==========
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
    print(f'\nBaseline: {baseline_correct}/{test_samples} = {baseline_acc*100:.2f}%')

    # ========== HEURISTIC METHOD ==========
    print('\n' + '=' * 60)
    print(f'HEURISTIC: {method} (top_k={top_k})')
    print('=' * 60)

    # Create heuristic selector
    selector = get_heuristic_selector(method, top_k)

    # Create a custom policy wrapper for the heuristic
    class HeuristicPolicy:
        def __init__(self, selector):
            self.selector = selector
            self.eval()  # Always in eval mode

        def eval(self):
            pass

        def __call__(self, query_embed, block_summaries, **kwargs):
            indices = self.selector.select_blocks(query_embed, block_summaries)
            return indices

    heuristic_policy = HeuristicPolicy(selector)

    heuristic_method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=heuristic_policy, top_k_blocks=top_k, rl_training=False,
    )
    heuristic_method.task = task

    heuristic_correct = 0
    heuristic_selected_blocks = 0
    heuristic_total_blocks = 0

    for i, item in enumerate(test_data):
        result = heuristic_method.run_batch_vllm([item])[0]
        if result.get('correct', False):
            heuristic_correct += 1

        stats = heuristic_method.get_efficiency_stats()
        heuristic_total_blocks += stats['total_blocks']
        heuristic_selected_blocks += stats['selected_blocks']

        if (i + 1) % 100 == 0:
            print(f'  {method}: {i+1}/{test_samples}, acc={heuristic_correct}/{i+1}')
            sys.stdout.flush()

    heuristic_acc = heuristic_correct / test_samples
    read_ratio = heuristic_selected_blocks / max(heuristic_total_blocks, 1)
    latent_reduction = (1 - read_ratio) * 100
    acc_change = (heuristic_acc - baseline_acc) * 100

    # ========== RESULTS ==========
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'Task: {task}')
    print(f'Method: {method}')
    print(f'top_k: {top_k}')
    print(f'')
    print(f'Baseline:   {baseline_acc*100:.2f}% (read 100%)')
    print(f'Heuristic:  {heuristic_acc*100:.2f}% (read {read_ratio*100:.2f}%)')
    print(f'')
    print(f'Accuracy change: {acc_change:+.2f}%')
    print(f'Latent reduction: {latent_reduction:.1f}%')
    print('=' * 60)

    # Save results
    results = {
        'task': task,
        'method': method,
        'top_k': top_k,
        'test_samples': test_samples,
        'baseline_acc': baseline_acc,
        'heuristic_acc': heuristic_acc,
        'read_ratio': read_ratio,
        'accuracy_change': acc_change,
        'latent_reduction_pct': latent_reduction,
        'timestamp': datetime.now().isoformat(),
    }

    os.makedirs('logs/heuristics', exist_ok=True)
    result_file = f'logs/heuristics/{task}_{method}_topk{top_k}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {result_file}')

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['gsm8k', 'arc_challenge', 'winogrande'])
    parser.add_argument('--method', type=str, required=True,
                        choices=['random', 'recency', 'similarity', 'time_weighted'])
    parser.add_argument('--top_k', type=int, required=True)
    parser.add_argument('--test_samples', type=int, default=500)

    args = parser.parse_args()

    evaluate_heuristic(
        task=args.task,
        method=args.method,
        top_k=args.top_k,
        test_samples=args.test_samples,
    )
