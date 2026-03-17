#!/usr/bin/env python3
"""
Top-K Ablation Experiment for Qwen3-4B

Compare baseline (all blocks) vs trained policies with different top_k values.
For each dataset (GSM8K, ARC-Challenge, Winogrande):
1. Baseline: read ALL blocks
2. Policy with top_k=5
3. Policy with top_k=10

IMPORTANT: Proper train/test split:
- Training: use TRAIN split
- Evaluation: use TEST split (completely unseen)
"""

import os

SCRIPT_DIR = "/p/scratch/westai0052/liu52/LatentMAS/scripts"
os.makedirs(SCRIPT_DIR, exist_ok=True)

# Experiment configurations
MODEL_NAME = "Qwen3-4B"
MODEL_PATH = "/p/scratch/westai0052/liu52/models/Qwen3-4B"
SEGMENT_LAYER_IDX = 16

# Proper train/test splits
DATASETS = {
    "gsm8k": {
        "train_split": "train",
        "test_split": "test",
        "train_samples": 300,
        "test_samples": 500,
    },
    "arc_challenge": {
        "train_split": "train",
        "test_split": "test",
        "train_samples": 300,
        "test_samples": 500,
    },
    "winogrande": {
        "train_split": "train",
        "test_split": "validation",  # winogrande has no test labels
        "train_samples": 300,
        "test_samples": 500,
    },
}

TOP_K_VALUES = [5, 10]

# Template for evaluation script
EVAL_TEMPLATE = '''#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=ablation_{task}_topk{top_k}
#SBATCH --output=logs/ablation_{task}_topk{top_k}_%j.out
#SBATCH --error=logs/ablation_{task}_topk{top_k}_%j.err

# Top-K Ablation: {task} with top_k={top_k}
# PROPER SPLIT: Train on TRAIN, Evaluate on TEST (unseen)

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

MODEL_NAME = "{model_name}"
MODEL_PATH = "{model_path}"
TASK = "{task}"
TOP_K = {top_k}
TRAIN_SPLIT = "{train_split}"
TEST_SPLIT = "{test_split}"
TRAIN_SAMPLES = {train_samples}
TEST_SAMPLES = {test_samples}

print("="*60)
print(f"TOP-K ABLATION: {{MODEL_NAME}} on {{TASK}}")
print(f"top_k_blocks = {{TOP_K}}")
print(f"Train: {{TRAIN_SPLIT}} split ({{TRAIN_SAMPLES}} samples)")
print(f"Test: {{TEST_SPLIT}} split ({{TEST_SAMPLES}} samples) - UNSEEN")
print("="*60)
sys.stdout.flush()

from data import load_gsm8k, load_arc_challenge, load_winogrande
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
    top_k_blocks = TOP_K
    similarity_threshold = 0.85
    min_block_size = 4
    max_block_size = 64
    segment_layer_idx = {segment_layer_idx}
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

# Load TRAIN data (for training policy)
print(f'\\nLoading TRAIN data: {{TRAIN_SPLIT}} split...')
if TASK == "gsm8k":
    train_data = list(load_gsm8k(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
elif TASK == "arc_challenge":
    train_data = list(load_arc_challenge(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
elif TASK == "winogrande":
    train_data = list(load_winogrande(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
print(f'Loaded {{len(train_data)}} training samples')

# Load TEST data (for evaluation - completely unseen)
print(f'Loading TEST data: {{TEST_SPLIT}} split...')
if TASK == "gsm8k":
    test_data = list(load_gsm8k(split=TEST_SPLIT))[:TEST_SAMPLES]
elif TASK == "arc_challenge":
    test_data = list(load_arc_challenge(split=TEST_SPLIT))[:TEST_SAMPLES]
elif TASK == "winogrande":
    test_data = list(load_winogrande(split=TEST_SPLIT))[:TEST_SAMPLES]
print(f'Loaded {{len(test_data)}} test samples (UNSEEN)')
sys.stdout.flush()

# ========================================
# BASELINE: Read ALL blocks on TEST set
# ========================================
print('\\n' + '='*60)
print('BASELINE (read ALL blocks) on TEST set')
print('='*60)
sys.stdout.flush()

baseline_method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=None, top_k_blocks=args.top_k_blocks, rl_training=False,
)
baseline_method.task = TASK

baseline_correct = 0
baseline_total_blocks = 0
baseline_selected_blocks = 0
for i, item in enumerate(test_data):
    result = baseline_method.run_batch_vllm([item])[0]
    if result.get('correct', False):
        baseline_correct += 1
    stats = baseline_method.get_efficiency_stats()
    baseline_total_blocks += stats['total_blocks']
    baseline_selected_blocks += stats['selected_blocks']
    if (i + 1) % 50 == 0:
        print(f'  Baseline: {{i+1}}/{{TEST_SAMPLES}}, acc={{baseline_correct}}/{{i+1}}')
        sys.stdout.flush()

baseline_acc = baseline_correct / TEST_SAMPLES
baseline_read_ratio = baseline_selected_blocks / max(baseline_total_blocks, 1)
print(f'\\nBaseline: {{baseline_correct}}/{{TEST_SAMPLES}} = {{baseline_acc*100:.2f}}%')
print(f'Baseline read ratio: {{baseline_read_ratio*100:.2f}}%')
sys.stdout.flush()

# ========================================
# TRAIN policy on TRAIN set
# ========================================
print('\\n' + '='*60)
print(f'TRAINING POLICY (top_k={{TOP_K}}) on TRAIN set')
print('='*60)
sys.stdout.flush()

# Initialize policy
policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)
policy.train()

# Create method with policy
method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=policy, top_k_blocks=TOP_K, rl_training=True,
)
method.task = TASK

# Training setup
from training.reward import RewardCalculator
from training.grpo import GRPOTrainer

optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)
reward_calculator = RewardCalculator(alpha=1.0, beta=0.3, gamma=0.1)
trainer = GRPOTrainer(
    policy_net=policy, optimizer=optimizer, reward_calculator=reward_calculator,
    group_size=4, clip_epsilon=0.2, kl_coef=0.05, entropy_coef=0.02,
    max_grad_norm=1.0, device=args.device,
)

# Train for 3 epochs on TRAIN data
for epoch in range(3):
    print(f'\\nEpoch {{epoch+1}}/3')
    method.rl_training = True
    policy.train()

    current_group = []
    epoch_rewards = []

    for i, item in enumerate(train_data):
        result = method.run_batch_vllm([item])[0]
        trajectory_info = method.get_trajectory_info()
        if trajectory_info is None:
            continue

        total_reward, components = reward_calculator.compute_total_reward(
            prediction=result.get("prediction"),
            gold=result.get("gold"),
            task_type=TASK,
            num_selected_blocks=TOP_K,
            total_blocks=trajectory_info.get("num_blocks", 0),
            correct_override=result.get("correct"),
        )

        transition = method.create_transition(
            reward=total_reward,
            task_reward=components["task_reward"],
            consistency_reward=components["consistency_reward"],
            cost_penalty=components["cost_penalty"],
        )

        if transition is not None:
            current_group.append(transition)
            epoch_rewards.append(total_reward)

        if len(current_group) >= 4:
            trainer.buffer.groups.append(current_group)
            current_group = []

            if len(trainer.buffer.groups) >= 2:
                trainer.update_policy(trainer.buffer.groups)
                trainer.buffer.groups = []

        if (i + 1) % 50 == 0:
            avg_reward = sum(epoch_rewards[-50:]) / max(len(epoch_rewards[-50:]), 1)
            print(f'  Train: {{i+1}}/{{len(train_data)}}, avg_reward={{avg_reward:.3f}}')
            sys.stdout.flush()

    # Handle remaining
    if current_group:
        trainer.buffer.groups.append(current_group)
        if trainer.buffer.groups:
            trainer.update_policy(trainer.buffer.groups)
            trainer.buffer.groups = []

# ========================================
# EVALUATE trained policy on TEST set (UNSEEN)
# ========================================
print('\\n' + '='*60)
print(f'EVALUATING POLICY (top_k={{TOP_K}}) on TEST set (UNSEEN)')
print('='*60)
sys.stdout.flush()

method.rl_training = False
policy.eval()

policy_correct = 0
policy_total_blocks = 0
policy_selected_blocks = 0
for i, item in enumerate(test_data):
    result = method.run_batch_vllm([item])[0]
    if result.get('correct', False):
        policy_correct += 1
    stats = method.get_efficiency_stats()
    policy_total_blocks += stats['total_blocks']
    policy_selected_blocks += stats['selected_blocks']
    if (i + 1) % 50 == 0:
        print(f'  Policy: {{i+1}}/{{TEST_SAMPLES}}, acc={{policy_correct}}/{{i+1}}')
        sys.stdout.flush()

policy_acc = policy_correct / TEST_SAMPLES
policy_read_ratio = policy_selected_blocks / max(policy_total_blocks, 1)
latent_reduction = (1 - policy_read_ratio / max(baseline_read_ratio, 0.001)) * 100 if baseline_read_ratio > 0 else 0

print(f'\\nPolicy: {{policy_correct}}/{{TEST_SAMPLES}} = {{policy_acc*100:.2f}}%')
print(f'Policy read ratio: {{policy_read_ratio*100:.2f}}%')
sys.stdout.flush()

# Summary
acc_change = (policy_acc - baseline_acc) * 100

print('\\n' + '='*60)
print('ABLATION RESULTS (TEST SET - UNSEEN)')
print('='*60)
print(f'Model: {{MODEL_NAME}}')
print(f'Task: {{TASK}}')
print(f'top_k_blocks: {{TOP_K}}')
print(f'Train samples: {{TRAIN_SAMPLES}} ({{TRAIN_SPLIT}} split)')
print(f'Test samples: {{TEST_SAMPLES}} ({{TEST_SPLIT}} split)')
print(f'')
print(f'BASELINE (all blocks):')
print(f'  Accuracy: {{baseline_acc*100:.2f}}%')
print(f'  Read ratio: {{baseline_read_ratio*100:.2f}}%')
print(f'')
print(f'POLICY (top_k={{TOP_K}}):')
print(f'  Accuracy: {{policy_acc*100:.2f}}%')
print(f'  Read ratio: {{policy_read_ratio*100:.2f}}%')
print(f'  Accuracy change: {{acc_change:+.2f}}%')
print(f'  Latent reduction: {{latent_reduction:.1f}}%')
print('='*60)

# Save results
results = {{
    'model': MODEL_NAME,
    'task': TASK,
    'top_k': TOP_K,
    'train_split': TRAIN_SPLIT,
    'test_split': TEST_SPLIT,
    'train_samples': TRAIN_SAMPLES,
    'test_samples': TEST_SAMPLES,
    'baseline_acc': baseline_acc,
    'baseline_read_ratio': baseline_read_ratio,
    'baseline_total_blocks': baseline_total_blocks,
    'policy_acc': policy_acc,
    'policy_read_ratio': policy_read_ratio,
    'policy_total_blocks': policy_total_blocks,
    'accuracy_change': acc_change,
    'latent_reduction_pct': latent_reduction,
    'timestamp': datetime.now().isoformat(),
}}
result_file = f'logs/ablation_{{TASK}}_topk{{TOP_K}}.json'
with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to {{result_file}}')
EOF

echo "Ablation experiment completed!"
'''

# Generate scripts
scripts = []
for task, task_cfg in DATASETS.items():
    for top_k in TOP_K_VALUES:
        script_content = EVAL_TEMPLATE.format(
            task=task,
            top_k=top_k,
            model_name=MODEL_NAME,
            model_path=MODEL_PATH,
            train_split=task_cfg["train_split"],
            test_split=task_cfg["test_split"],
            train_samples=task_cfg["train_samples"],
            test_samples=task_cfg["test_samples"],
            segment_layer_idx=SEGMENT_LAYER_IDX,
        )

        script_path = os.path.join(SCRIPT_DIR, f"ablation_{task}_topk{top_k}.sh")
        with open(script_path, "w") as f:
            f.write(script_content)

        scripts.append(script_path)
        print(f"Generated: {script_path}")

# Generate submission script
submit_script = '''#!/bin/bash
# Submit all top-k ablation experiments
# PROPER SPLIT: Train on TRAIN, Evaluate on TEST (unseen)

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "Submitting Top-K Ablation Experiments..."
echo "========================================="
echo "Train: TRAIN split (300 samples)"
echo "Test: TEST split (500 samples) - UNSEEN"
echo ""

'''

for script in scripts:
    script_name = os.path.basename(script)
    submit_script += f'echo "Submitting {script_name}..."\n'
    submit_script += f'sbatch {script}\n'
    submit_script += '\n'

submit_script += '''
echo "========================================="
echo "All ablation experiments submitted!"
echo "Check status: squeue -u $USER"
echo "Results will be in: logs/ablation_*.json"
'''

submit_path = os.path.join(SCRIPT_DIR, "submit_topk_ablation.sh")
with open(submit_path, "w") as f:
    f.write(submit_script)
os.chmod(submit_path, 0o755)
print(f"\nGenerated submission script: {submit_path}")

print(f"\nTotal scripts: {len(scripts)}")
print("\nData splits:")
print("  - Training: TRAIN split (300 samples)")
print("  - Evaluation: TEST split (500 samples) - completely unseen")
print("\nTo run: bash scripts/submit_topk_ablation.sh")
