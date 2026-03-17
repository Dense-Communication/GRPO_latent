#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=eval_8b
#SBATCH --output=logs/eval_8b_all_%j.out
#SBATCH --error=logs/eval_8b_all_%j.err

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

echo "=========================================="
echo "Qwen3-8B V4 Evaluation - All Configs"
echo "=========================================="

python3 -c "
import sys
sys.path.insert(0, '/p/scratch/westai0052/liu52/LatentMAS')

import torch
import json
import os
from datetime import datetime

from data import load_gsm8k, load_arc_challenge, load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy.reading_policy import ReadingPolicyNetwork
from utils import set_seed

set_seed(42)

MODEL_PATH = '/p/scratch/westai0052/liu52/models/Qwen3-8B'
CHECKPOINT_DIR = 'checkpoints/ablation_v4_8b'
TEST_SAMPLES = 500
HIDDEN_DIM = 4096  # 8B hidden size

class Args:
    pass

args = Args()
args.model_name = MODEL_PATH
args.prompt = 'sequential'
args.method = 'latent_mas'
args.device = 'cuda:0'
args.device2 = 'cuda:2'  # Use GPU 2 for HF model (vLLM uses 0,1)
args.think = False
args.latent_steps = 3
args.similarity_threshold = 0.85
args.min_block_size = 4
args.max_block_size = 64
args.segment_layer_idx = 20  # 8B has more layers
args.tensor_parallel_size = 2
args.gpu_memory_utilization = 0.80
args.enable_prefix_caching = True
args.use_second_HF_model = True
args.latent_space_realign = False
args.max_new_tokens = 512

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)

# Configs to evaluate
configs = [
    ('gsm8k', 10), ('gsm8k', 40), ('gsm8k', 50),
    ('arc_challenge', 10), ('arc_challenge', 40), ('arc_challenge', 50),
    ('winogrande', 10), ('winogrande', 40), ('winogrande', 50),
]

os.makedirs('logs/ablation_v4_8b', exist_ok=True)

for task, top_k in configs:
    checkpoint_path = f'{CHECKPOINT_DIR}/{task}_topk{top_k}_policy.pt'
    if not os.path.exists(checkpoint_path):
        print(f'\\nSkipping {task} k={top_k}: checkpoint not found')
        continue

    print('\\n' + '='*60)
    print(f'EVALUATING: {task} top_k={top_k}')
    print('='*60)

    args.top_k_blocks = top_k
    args.task = task

    # Load policy (must match training config: num_heads=4, num_layers=1)
    # Policy must be on args.device (same as vLLM input), not device2
    policy = ReadingPolicyNetwork(
        hidden_dim=HIDDEN_DIM,
        num_heads=4,
        num_layers=1,
    ).to(args.device).to(torch.bfloat16)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    policy.eval()

    # Load data
    if task == 'gsm8k':
        test_data = list(load_gsm8k(split='test'))[:TEST_SAMPLES]
    elif task == 'arc_challenge':
        test_data = list(load_arc_challenge(split='test'))[:TEST_SAMPLES]
    else:
        test_data = list(load_winogrande(split='validation'))[:TEST_SAMPLES]

    # Baseline
    print('Running baseline...')
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
            print(f'  Baseline: {i+1}/{TEST_SAMPLES}, acc={baseline_correct}/{i+1}')
            sys.stdout.flush()
    baseline_acc = baseline_correct / TEST_SAMPLES

    # Policy
    print('Running policy...')
    test_method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=policy, top_k_blocks=top_k, rl_training=False,
    )
    test_method.task = task

    policy_correct = 0
    total_blocks = 0
    selected_blocks = 0

    for i, item in enumerate(test_data):
        result = test_method.run_batch_vllm([item])[0]
        if result.get('correct', False):
            policy_correct += 1
        stats = test_method.get_efficiency_stats()
        total_blocks += stats['total_blocks']
        selected_blocks += stats['selected_blocks']
        if (i + 1) % 100 == 0:
            print(f'  Policy: {i+1}/{TEST_SAMPLES}, acc={policy_correct}/{i+1}')
            sys.stdout.flush()

    policy_acc = policy_correct / TEST_SAMPLES
    read_ratio = selected_blocks / max(total_blocks, 1)
    latent_reduction = (1 - read_ratio) * 100
    acc_change = (policy_acc - baseline_acc) * 100

    print(f'\\nResults for {task} k={top_k}:')
    print(f'  Baseline: {baseline_acc*100:.2f}%')
    print(f'  Policy:   {policy_acc*100:.2f}% (Δ{acc_change:+.2f}%)')
    print(f'  Latent Reduction: {latent_reduction:.1f}%')

    results = {
        'model': 'Qwen3-8B',
        'task': task,
        'top_k': top_k,
        'test_samples': TEST_SAMPLES,
        'baseline_acc': baseline_acc,
        'policy_acc': policy_acc,
        'policy_read_ratio': read_ratio,
        'accuracy_change': acc_change,
        'latent_reduction_pct': latent_reduction,
        'version': 'v4_8b',
        'timestamp': datetime.now().isoformat(),
    }

    result_file = f'logs/ablation_v4_8b/{task}_topk{top_k}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved: {result_file}')

print('\\n' + '='*60)
print('ALL EVALUATIONS COMPLETE!')
print('='*60)
"

echo "Done!"
