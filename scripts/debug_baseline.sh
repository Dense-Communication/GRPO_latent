#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=debug_baseline
#SBATCH --output=logs/debug_baseline_%j.out
#SBATCH --error=logs/debug_baseline_%j.err

# Debug baseline mode to see what's happening

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
print("Debug: Baseline Mode Output")
print("="*60)
sys.stdout.flush()

from data import load_gsm8k
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
    task = 'gsm8k'

args = Args()

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)

hidden_dim = model.HF_model.config.hidden_size
policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)

# Load a few samples
data = list(load_gsm8k(split='train'))[:3]

# Test BASELINE (no policy)
print('\n' + '='*60)
print('BASELINE (no reading policy)')
print('='*60)
sys.stdout.flush()

method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=None, top_k_blocks=args.top_k_blocks, rl_training=False,
)
method.task = 'gsm8k'

for i, item in enumerate(data):
    print(f'\n--- Sample {i} ---')
    print(f"Question: {item['question'][:200]}...")
    print(f"Gold: {item['gold']}")
    sys.stdout.flush()

    result = method.run_batch_vllm([item])[0]

    # Print the full generated text
    print(f"\nRaw prediction (full text):")
    raw = result.get('raw_prediction', 'N/A')
    print(f"'{raw[:500]}'" if raw else "'EMPTY'")
    print(f"\nExtracted prediction: {result.get('prediction')}")
    print(f"Correct: {result.get('correct')}")
    sys.stdout.flush()

# Test WITH POLICY
print('\n' + '='*60)
print('WITH POLICY')
print('='*60)
sys.stdout.flush()

method2 = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=policy, top_k_blocks=args.top_k_blocks, rl_training=False,
)
method2.task = 'gsm8k'

for i, item in enumerate(data):
    print(f'\n--- Sample {i} ---')
    print(f"Question: {item['question'][:200]}...")
    print(f"Gold: {item['gold']}")
    sys.stdout.flush()

    result = method2.run_batch_vllm([item])[0]

    # Print the full generated text
    print(f"\nRaw prediction (full text):")
    raw = result.get('raw_prediction', 'N/A')
    print(f"'{raw[:500]}'" if raw else "'EMPTY'")
    print(f"\nExtracted prediction: {result.get('prediction')}")
    print(f"Correct: {result.get('correct')}")
    sys.stdout.flush()

print('\n' + '='*60)
print('Debug completed!')
print('='*60)
EOF

echo "Debug script completed!"
