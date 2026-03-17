#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_v4_winogrande_k50
#SBATCH --output=logs/eval_v4_winogrande_topk50_%j.out
#SBATCH --error=logs/eval_v4_winogrande_topk50_%j.err

# V4: Evaluate pure task optimization
# winogrande with top_k=50

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

TASK = "winogrande"
TOP_K = 50
TEST_SPLIT = "validation"
TEST_SAMPLES = 500
CHECKPOINT_PATH = "checkpoints/ablation_v4/winogrande_topk50_policy.pt"

print("="*60)
print(f"EVALUATION V4: {TASK} with top_k={TOP_K}")
print(f"Test: {TEST_SPLIT} split ({TEST_SAMPLES} samples)")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print("="*60)
sys.stdout.flush()

from data import load_gsm8k, load_arc_challenge, load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy import ReadingPolicyNetwork
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
    task = TASK

args = Args()

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)
hidden_dim = model.HF_model.config.hidden_size

# Load TEST data
print(f'\nLoading TEST data...')
if TASK == "gsm8k":
    test_data = list(load_gsm8k(split=TEST_SPLIT))[:TEST_SAMPLES]
elif TASK == "arc_challenge":
    test_data = list(load_arc_challenge(split=TEST_SPLIT))[:TEST_SAMPLES]
elif TASK == "winogrande":
    test_data = list(load_winogrande(split=TEST_SPLIT))[:TEST_SAMPLES]
print(f'Loaded {len(test_data)} test samples')
sys.stdout.flush()

# ========== BASELINE (read ALL blocks) ==========
print('\n' + '='*60)
print('BASELINE (read ALL blocks)')
print('='*60)

baseline_method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=None, top_k_blocks=TOP_K, rl_training=False,
)
baseline_method.task = TASK

baseline_correct = 0
baseline_total_blocks = 0
for i, item in enumerate(test_data):
    result = baseline_method.run_batch_vllm([item])[0]
    if result.get('correct', False):
        baseline_correct += 1
    stats = baseline_method.get_efficiency_stats()
    baseline_total_blocks += stats['total_blocks']
    if (i + 1) % 100 == 0:
        print(f'  Baseline: {i+1}/{TEST_SAMPLES}, acc={baseline_correct}/{i+1}')
        sys.stdout.flush()

baseline_acc = baseline_correct / TEST_SAMPLES
print(f'\nBaseline: {baseline_correct}/{TEST_SAMPLES} = {baseline_acc*100:.2f}%')

# ========== POLICY (read top_k blocks) ==========
print('\n' + '='*60)
print(f'POLICY (top_k={TOP_K})')
print('='*60)

# Load trained policy
policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)
policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=args.device))
policy.eval()
print(f'Loaded checkpoint: {CHECKPOINT_PATH}')

policy_method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=policy, top_k_blocks=TOP_K, rl_training=False,
)
policy_method.task = TASK

policy_correct = 0
policy_selected_blocks = 0
policy_total_blocks = 0
for i, item in enumerate(test_data):
    result = policy_method.run_batch_vllm([item])[0]
    if result.get('correct', False):
        policy_correct += 1
    stats = policy_method.get_efficiency_stats()
    policy_total_blocks += stats['total_blocks']
    policy_selected_blocks += stats['selected_blocks']
    if (i + 1) % 100 == 0:
        print(f'  Policy: {i+1}/{TEST_SAMPLES}, acc={policy_correct}/{i+1}')
        sys.stdout.flush()

policy_acc = policy_correct / TEST_SAMPLES
policy_read_ratio = policy_selected_blocks / max(policy_total_blocks, 1)
latent_reduction = (1 - policy_read_ratio) * 100

acc_change = (policy_acc - baseline_acc) * 100

print('\n' + '='*60)
print('RESULTS')
print('='*60)
print(f'Task: {TASK}')
print(f'top_k: {TOP_K}')
print(f'')
print(f'Baseline: {baseline_acc*100:.2f}% (read 100%)')
print(f'Policy:   {policy_acc*100:.2f}% (read {policy_read_ratio*100:.2f}%)')
print(f'')
print(f'Accuracy change: {acc_change:+.2f}%')
print(f'Latent reduction: {latent_reduction:.1f}%')
print('='*60)

# Save results
results = {
    'task': TASK,
    'top_k': TOP_K,
    'test_split': TEST_SPLIT,
    'test_samples': TEST_SAMPLES,
    'baseline_acc': baseline_acc,
    'policy_acc': policy_acc,
    'policy_read_ratio': policy_read_ratio,
    'accuracy_change': acc_change,
    'latent_reduction_pct': latent_reduction,
    'version': 'v4',
    'hyperparams': {
        'train_samples': 500,
        'lr': 1e-05,
        'alpha': 1.0,
        'beta': 0.3,
        'gamma': 0.0,
        'epochs': 5,
    },
    'timestamp': datetime.now().isoformat(),
}
result_file = f'logs/ablation_v4_{TASK}_topk{TOP_K}.json'
with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to {result_file}')
EOF

echo "Evaluation job completed!"
