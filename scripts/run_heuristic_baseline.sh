#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=heur_base
#SBATCH --output=logs/heuristic_baseline_%j.out
#SBATCH --error=logs/heuristic_baseline_%j.err

# Heuristic Baseline: Compare random, recency, similarity on all 3 datasets

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/heuristics

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 << 'EOF'
import torch
import sys
import json
import os
import random
import numpy as np
from datetime import datetime

TOP_K = 40  # Same as our best V4 config
TEST_SAMPLES = 500

# Heuristic methods
METHODS = ['random', 'recency', 'similarity']
TASKS = ['gsm8k', 'arc_challenge', 'winogrande']

print("="*60)
print("HEURISTIC BASELINE EVALUATION")
print(f"Methods: {METHODS}")
print(f"Tasks: {TASKS}")
print(f"top_k={TOP_K}")
print("="*60)
sys.stdout.flush()

from data import load_gsm8k, load_arc_challenge, load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from utils import set_seed

set_seed(42)

class Args:
    model_name = "/p/scratch/westai0052/liu52/models/Qwen3-4B"
    prompt = 'sequential'
    method = 'latent_mas'
    device = 'cuda:0'
    device2 = 'cuda:1'
    think = False
    latent_steps = 3
    top_k_blocks = TOP_K
    similarity_threshold = 0.85
    min_block_size = 4
    max_block_size = 64
    segment_layer_idx = 16
    tensor_parallel_size = 1
    gpu_memory_utilization = 0.85
    enable_prefix_caching = True
    use_second_HF_model = True
    latent_space_realign = False
    max_new_tokens = 512
    task = 'gsm8k'

args = Args()

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)

os.makedirs('logs/heuristics', exist_ok=True)

# Heuristic selection functions
def select_random(block_summaries, query_embed, k):
    """Random selection"""
    n = block_summaries.shape[0]
    indices = torch.randperm(n)[:k].sort().values
    return indices

def select_recency(block_summaries, query_embed, k):
    """Select most recent (last k) blocks"""
    n = block_summaries.shape[0]
    indices = torch.arange(max(0, n-k), n)
    return indices

def select_similarity(block_summaries, query_embed, k):
    """Select by cosine similarity to query"""
    # Normalize
    block_norm = block_summaries / (block_summaries.norm(dim=-1, keepdim=True) + 1e-8)
    query_norm = query_embed / (query_embed.norm() + 1e-8)

    # Cosine similarity
    similarities = torch.matmul(block_norm, query_norm)

    # Top-k
    _, indices = similarities.topk(min(k, len(similarities)))
    return indices.sort().values

HEURISTIC_FNS = {
    'random': select_random,
    'recency': select_recency,
    'similarity': select_similarity,
}

for task in TASKS:
    print(f'\n{"="*60}')
    print(f'TASK: {task}')
    print('='*60)

    args.task = task

    # Load data
    if task == 'gsm8k':
        test_data = list(load_gsm8k(split='test'))[:TEST_SAMPLES]
    elif task == 'arc_challenge':
        test_data = list(load_arc_challenge(split='test'))[:TEST_SAMPLES]
    else:
        test_data = list(load_winogrande(split='validation'))[:TEST_SAMPLES]

    print(f'Loaded {len(test_data)} test samples')

    # Run baseline first (read ALL)
    print('\nRunning baseline (read ALL)...')
    baseline_method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=None, top_k_blocks=TOP_K, rl_training=False,
    )
    baseline_method.task = task

    baseline_correct = 0
    for i, item in enumerate(test_data):
        result = baseline_method.run_batch_vllm([item])[0]
        if result.get('correct', False):
            baseline_correct += 1
        if (i + 1) % 100 == 0:
            print(f'  Baseline: {i+1}/{TEST_SAMPLES}, acc={baseline_correct}/{i+1}')
            sys.stdout.flush()

    baseline_acc = baseline_correct / TEST_SAMPLES
    print(f'Baseline: {baseline_acc*100:.2f}%')

    # Run each heuristic
    for method_name in METHODS:
        print(f'\nRunning heuristic: {method_name}...')

        heuristic_fn = HEURISTIC_FNS[method_name]

        # Create method with heuristic selection
        heur_method = LatentMASMethodRL(
            model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
            args=args, reading_policy=None, top_k_blocks=TOP_K, rl_training=False,
        )
        heur_method.task = task
        heur_method.heuristic_selector = heuristic_fn
        heur_method.use_heuristic = True

        heur_correct = 0
        total_blocks = 0
        selected_blocks = 0

        for i, item in enumerate(test_data):
            result = heur_method.run_batch_vllm([item])[0]
            if result.get('correct', False):
                heur_correct += 1

            stats = heur_method.get_efficiency_stats()
            total_blocks += stats.get('total_blocks', 0)
            selected_blocks += stats.get('selected_blocks', 0)

            if (i + 1) % 100 == 0:
                print(f'  {method_name}: {i+1}/{TEST_SAMPLES}, acc={heur_correct}/{i+1}')
                sys.stdout.flush()

        heur_acc = heur_correct / TEST_SAMPLES
        read_ratio = selected_blocks / max(total_blocks, 1)
        latent_reduction = (1 - read_ratio) * 100
        acc_change = (heur_acc - baseline_acc) * 100

        print(f'\n{method_name} Results:')
        print(f'  Accuracy: {heur_acc*100:.2f}% (Δ{acc_change:+.2f}%)')
        print(f'  Latent Reduction: {latent_reduction:.1f}%')

        results = {
            'task': task,
            'method': method_name,
            'top_k': TOP_K,
            'test_samples': TEST_SAMPLES,
            'baseline_acc': baseline_acc,
            'heuristic_acc': heur_acc,
            'read_ratio': read_ratio,
            'accuracy_change': acc_change,
            'latent_reduction_pct': latent_reduction,
            'timestamp': datetime.now().isoformat(),
        }

        result_file = f'logs/heuristics/{task}_{method_name}.json'
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Saved: {result_file}')

print('\n' + '='*60)
print('HEURISTIC BASELINE COMPLETE!')
print('='*60)
EOF

echo "Done!"
