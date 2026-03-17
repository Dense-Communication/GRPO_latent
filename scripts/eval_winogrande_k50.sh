#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_wino50
#SBATCH --output=logs/eval_winogrande_k50_%j.out
#SBATCH --error=logs/eval_winogrande_k50_%j.err

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

echo "=========================================="
echo "Winogrande k=50 Evaluation"
echo "=========================================="

python3 -c "
import sys
sys.path.insert(0, '/p/scratch/westai0052/liu52/LatentMAS')

import torch
import json
from datetime import datetime

from data import load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy.reading_policy import ReadingPolicyNetwork
from utils import set_seed

set_seed(42)

MODEL_PATH = '/p/scratch/westai0052/liu52/models/Qwen3-4B'
CHECKPOINT_PATH = 'checkpoints/ablation_v4/winogrande_topk50_policy.pt'
TOP_K = 50
TEST_SAMPLES = 500

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
args.top_k_blocks = TOP_K
args.similarity_threshold = 0.85
args.min_block_size = 4
args.max_block_size = 64
args.segment_layer_idx = 16
args.tensor_parallel_size = 1
args.gpu_memory_utilization = 0.85
args.enable_prefix_caching = True
args.use_second_HF_model = True
args.latent_space_realign = False
args.max_new_tokens = 512
args.task = 'winogrande'

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)

print('Loading policy...')
# Must match training config: num_heads=4, num_layers=1
# Policy must be on args.device (same as vLLM), not device2
policy = ReadingPolicyNetwork(
    hidden_dim=2560,
    num_heads=4,
    num_layers=1,
).to(args.device).to(torch.bfloat16)
policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=args.device))
policy.eval()

print('Loading test data...')
test_data = list(load_winogrande(split='validation'))[:TEST_SAMPLES]
print(f'Loaded {len(test_data)} samples')

# Baseline
print('\\n' + '='*60)
print('BASELINE (read ALL)')
print('='*60)
baseline_method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=None, top_k_blocks=TOP_K, rl_training=False,
)
baseline_method.task = 'winogrande'

baseline_correct = 0
for i, item in enumerate(test_data):
    result = baseline_method.run_batch_vllm([item])[0]
    if result.get('correct', False):
        baseline_correct += 1
    if (i + 1) % 100 == 0:
        print(f'  Baseline: {i+1}/{TEST_SAMPLES}, acc={baseline_correct}/{i+1}')
        sys.stdout.flush()

baseline_acc = baseline_correct / TEST_SAMPLES
print(f'Baseline: {baseline_correct}/{TEST_SAMPLES} = {baseline_acc*100:.2f}%')

# Policy
print('\\n' + '='*60)
print(f'POLICY (top_k={TOP_K})')
print('='*60)
test_method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=policy, top_k_blocks=TOP_K, rl_training=False,
)
test_method.task = 'winogrande'

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

print('\\n' + '='*60)
print('RESULTS')
print('='*60)
print(f'Baseline: {baseline_acc*100:.2f}%')
print(f'Policy:   {policy_acc*100:.2f}% (read {read_ratio*100:.2f}%)')
print(f'Accuracy change: {acc_change:+.2f}%')
print(f'Latent reduction: {latent_reduction:.1f}%')

results = {
    'task': 'winogrande',
    'top_k': TOP_K,
    'test_split': 'validation',
    'test_samples': TEST_SAMPLES,
    'baseline_acc': baseline_acc,
    'policy_acc': policy_acc,
    'policy_read_ratio': read_ratio,
    'accuracy_change': acc_change,
    'latent_reduction_pct': latent_reduction,
    'version': 'v4',
    'timestamp': datetime.now().isoformat(),
}

with open('logs/ablation_v4_winogrande_topk50.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\\nResults saved!')
"

echo "Done!"
