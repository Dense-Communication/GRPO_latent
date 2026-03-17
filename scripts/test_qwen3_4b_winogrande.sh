#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=test_qwen3_4b_winogrande
#SBATCH --output=logs/test_qwen3_4b_winogrande_%j.out
#SBATCH --error=logs/test_qwen3_4b_winogrande_%j.err

# Evaluate TRAINED policy on TEST split: Qwen3-4B on winogrande
# This verifies that latent reduction is not overfitting to train data

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 << 'EOF'
import torch
import sys
import json
from datetime import datetime

MODEL_NAME = "Qwen3-4B"
MODEL_PATH = "/p/scratch/westai0052/liu52/models/Qwen3-4B"
TASK = "winogrande"
N_SAMPLES = 500
POLICY_CHECKPOINT = "/p/scratch/westai0052/liu52/LatentMAS/checkpoints/rl_policy/qwen3_4b_winogrande_14508791/policy_best.pt"
DATA_SPLIT = "validation"

print("="*60)
print(f"TEST SET Evaluation: {MODEL_NAME} on {TASK}")
print(f"Data split: {DATA_SPLIT} (unseen during training)")
print(f"Policy checkpoint: {POLICY_CHECKPOINT}")
print(f"Samples: {N_SAMPLES}")
print("="*60)
sys.stdout.flush()

from data import load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy import ReadingPolicyNetwork
from utils import set_seed

set_seed(42)

class Args:
    model_name = MODEL_PATH
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
    task = TASK

args = Args()

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)

hidden_dim = model.HF_model.config.hidden_size
policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)

# Load trained policy checkpoint
print(f'Loading trained policy from {POLICY_CHECKPOINT}...')
policy.load_state_dict(torch.load(POLICY_CHECKPOINT, map_location=args.device))
policy.eval()
print('Trained policy loaded!')
sys.stdout.flush()

# Load TEST data (unseen during training)
print(f'\nLoading {DATA_SPLIT} split (unseen during training)...')
data = list(load_winogrande(split=DATA_SPLIT))[:N_SAMPLES]
print(f'Loaded {len(data)} samples from {DATA_SPLIT} split')
sys.stdout.flush()

# Baseline evaluation (random block selection)
print('\n' + '='*60)
print('BASELINE (random block selection)')
print('='*60)
sys.stdout.flush()

method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=None, top_k_blocks=args.top_k_blocks, rl_training=False,
)
method.task = TASK

baseline_correct = 0
baseline_total_blocks = 0
baseline_selected_blocks = 0
for i, item in enumerate(data):
    result = method.run_batch_vllm([item])[0]
    if result.get('correct', False):
        baseline_correct += 1
    stats = method.get_efficiency_stats()
    baseline_total_blocks += stats['total_blocks']
    baseline_selected_blocks += stats['selected_blocks']
    if (i + 1) % 50 == 0:
        print(f'  Baseline: {i+1}/{N_SAMPLES}, acc={baseline_correct}/{i+1}')
        sys.stdout.flush()

baseline_acc = baseline_correct / N_SAMPLES
baseline_read_ratio = baseline_selected_blocks / max(baseline_total_blocks, 1)
print(f'\nBaseline: {baseline_correct}/{N_SAMPLES} = {baseline_acc*100:.2f}%')
print(f'Baseline latent: {baseline_selected_blocks}/{baseline_total_blocks} ({baseline_read_ratio*100:.2f}%)')
sys.stdout.flush()

# Trained Policy evaluation
print('\n' + '='*60)
print('TRAINED POLICY')
print('='*60)
sys.stdout.flush()

method2 = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=policy, top_k_blocks=args.top_k_blocks, rl_training=False,
)
method2.task = TASK

policy_correct = 0
policy_total_blocks = 0
policy_selected_blocks = 0
for i, item in enumerate(data):
    result = method2.run_batch_vllm([item])[0]
    if result.get('correct', False):
        policy_correct += 1
    stats = method2.get_efficiency_stats()
    policy_total_blocks += stats['total_blocks']
    policy_selected_blocks += stats['selected_blocks']
    if (i + 1) % 50 == 0:
        print(f'  Policy: {i+1}/{N_SAMPLES}, acc={policy_correct}/{i+1}')
        sys.stdout.flush()

policy_acc = policy_correct / N_SAMPLES
policy_read_ratio = policy_selected_blocks / max(policy_total_blocks, 1)
print(f'\nTrained Policy: {policy_correct}/{N_SAMPLES} = {policy_acc*100:.2f}%')
print(f'Policy latent: {policy_selected_blocks}/{policy_total_blocks} ({policy_read_ratio*100:.2f}%)')
sys.stdout.flush()

# Summary
acc_change = (policy_acc - baseline_acc) * 100
latent_saved_pct = (1 - policy_read_ratio / max(baseline_read_ratio, 0.001)) * 100 if baseline_read_ratio > 0 else 0

print('\n' + '='*60)
print('TEST SET RESULTS (Overfitting Check)')
print('='*60)
print(f'Model: {MODEL_NAME}')
print(f'Task: {TASK}')
print(f'Split: {DATA_SPLIT} (unseen during training)')
print(f'Samples: {N_SAMPLES}')
print(f'')
print(f'ACCURACY:')
print(f'  Baseline:       {baseline_acc*100:.2f}%')
print(f'  Trained Policy: {policy_acc*100:.2f}%')
print(f'  Change: {acc_change:+.2f}%')
print(f'')
print(f'LATENT REDUCTION:')
print(f'  Baseline read ratio:       {baseline_read_ratio*100:.2f}%')
print(f'  Trained Policy read ratio: {policy_read_ratio*100:.2f}%')
print(f'  Latent reduction: {latent_saved_pct:.1f}%')
print('='*60)

# Save results
results = {
    'model': MODEL_NAME,
    'task': TASK,
    'data_split': DATA_SPLIT,
    'n_samples': N_SAMPLES,
    'policy_checkpoint': POLICY_CHECKPOINT,
    'baseline_acc': baseline_acc,
    'policy_acc': policy_acc,
    'accuracy_change': acc_change,
    'baseline_read_ratio': baseline_read_ratio,
    'policy_read_ratio': policy_read_ratio,
    'latent_reduction_pct': latent_saved_pct,
    'baseline_total_blocks': baseline_total_blocks,
    'policy_total_blocks': policy_total_blocks,
    'timestamp': datetime.now().isoformat(),
}
result_file = f'logs/test_results_{MODEL_NAME}_{TASK}.json'
with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to {result_file}')
EOF

echo "Test evaluation completed!"
