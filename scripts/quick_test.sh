#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=quick_test
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err

# Quick test to verify baseline vs policy on multiple datasets

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 << 'EOF'
import torch
import sys
print("="*60)
print("Quick Validation: Baseline vs Policy on Multiple Datasets")
print("="*60)
sys.stdout.flush()

from data import load_gsm8k, load_arc_challenge, load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy import ReadingPolicyNetwork
from utils import set_seed

set_seed(42)

class Args:
    model_name = '/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct'
    prompt = 'sequential'
    method = 'latent_mas'
    device = 'cuda:0'
    device2 = 'cuda:1'
    think = False
    latent_steps = 3
    top_k_blocks = 3
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
    task = 'gsm8k'  # will be updated per dataset

args = Args()

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)

hidden_dim = model.HF_model.config.hidden_size
policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)

# Test datasets
datasets_config = {
    'gsm8k': {'load_fn': load_gsm8k, 'split': 'train', 'n_samples': 5},
    'arc_challenge': {'load_fn': load_arc_challenge, 'split': 'train', 'n_samples': 5},
    'winogrande': {'load_fn': load_winogrande, 'split': 'train', 'n_samples': 5},
}

results = {}

for task_name, config in datasets_config.items():
    print(f'\n{"="*60}')
    print(f'Testing: {task_name}')
    print("="*60)
    sys.stdout.flush()

    data = list(config['load_fn'](split=config['split']))[:config['n_samples']]

    method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=policy, top_k_blocks=args.top_k_blocks, rl_training=False,
    )
    method.task = task_name

    # Test Baseline (no policy)
    print('\n--- Baseline (no policy) ---')
    sys.stdout.flush()
    method.reading_policy = None
    method.rl_training = False
    baseline_correct = 0
    for i, item in enumerate(data):
        result = method.run_batch_vllm([item])[0]
        ok = result.get('correct', False)
        if ok:
            baseline_correct += 1
        print(f"  [{i}] gold={item['gold']}, pred={result.get('prediction')}, correct={ok}")
        sys.stdout.flush()

    # Test with Policy
    print('\n--- With Policy ---')
    sys.stdout.flush()
    method.reading_policy = policy
    method.rl_training = False
    policy_correct = 0
    for i, item in enumerate(data):
        result = method.run_batch_vllm([item])[0]
        ok = result.get('correct', False)
        if ok:
            policy_correct += 1
        print(f"  [{i}] gold={item['gold']}, pred={result.get('prediction')}, correct={ok}")
        sys.stdout.flush()

    results[task_name] = {
        'baseline': baseline_correct,
        'policy': policy_correct,
        'total': config['n_samples']
    }

    print(f'\n>>> {task_name}: Baseline={baseline_correct}/{config["n_samples"]}, Policy={policy_correct}/{config["n_samples"]}')
    sys.stdout.flush()

# Summary
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
for task, r in results.items():
    print(f"  {task:15s}: Baseline={r['baseline']}/{r['total']}, Policy={r['policy']}/{r['total']}")
print('='*60)
print('Done!')
EOF

echo "Quick test completed!"
