#!/usr/bin/env python3
"""
Top-K Ablation Experiment v4 for Qwen3-8B

Same as v4 but using Qwen3-8B instead of Qwen3-4B.
Goal: Compare model size impact on selective reading performance.
"""

import os

SCRIPT_DIR = "/p/scratch/westai0052/liu52/LatentMAS/scripts"
os.makedirs(SCRIPT_DIR, exist_ok=True)

MODEL_NAME = "Qwen3-8B"
MODEL_PATH = "/p/scratch/westai0052/liu52/models/Qwen3-8B"
SEGMENT_LAYER_IDX = 20  # 8B has more layers, adjust accordingly

DATASETS = {
    "gsm8k": {"train_split": "train", "test_split": "test", "train_samples": 500, "test_samples": 500},
    "arc_challenge": {"train_split": "train", "test_split": "test", "train_samples": 500, "test_samples": 500},
    "winogrande": {"train_split": "train", "test_split": "validation", "train_samples": 500, "test_samples": 500},
}

# Test key top_k values based on v4 findings: 10, 40 (best), 50
TOP_K_VALUES = [10, 40, 50]

# V4 hyperparameters - pure task optimization
LEARNING_RATE = 1e-5
KL_COEF = 0.0
ENTROPY_COEF = 0.01
ALPHA = 1.0
BETA = 0.3
GAMMA = 0.0  # NO cost penalty
NUM_EPOCHS = 5

# ==================== Training Script ====================
TRAIN_TEMPLATE = '''#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=train_8b_{task}_k{top_k}
#SBATCH --output=logs/train_v4_8b_{task}_topk{top_k}_%j.out
#SBATCH --error=logs/train_v4_8b_{task}_topk{top_k}_%j.err

# V4 8B: Qwen3-8B evaluation
# {task} with top_k={top_k}

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation_v4_8b

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 << 'EOF'
import torch
import sys
import json
from datetime import datetime

TASK = "{task}"
TOP_K = {top_k}
TRAIN_SPLIT = "{train_split}"
TRAIN_SAMPLES = {train_samples}
CHECKPOINT_PATH = "checkpoints/ablation_v4_8b/{task}_topk{top_k}_policy.pt"

LR = {lr}
KL_COEF = {kl}
ENTROPY_COEF = {entropy}
ALPHA = {alpha}
BETA = {beta}
GAMMA = {gamma}
NUM_EPOCHS = {epochs}

print("="*60)
print(f"TRAINING V4 8B: {{TASK}} with top_k={{TOP_K}}")
print(f"Model: Qwen3-8B")
print(f"Train: {{TRAIN_SPLIT}} split ({{TRAIN_SAMPLES}} samples)")
print(f"Reward: R = {{ALPHA}}*R1 + {{BETA}}*R2 (NO COST PENALTY)")
print(f"LR={{LR}}, epochs={{NUM_EPOCHS}}")
print(f"Checkpoint: {{CHECKPOINT_PATH}}")
print("="*60)
sys.stdout.flush()

from data import load_gsm8k, load_arc_challenge, load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy import ReadingPolicyNetwork
from training.reward import RewardCalculator
from training.rl_trainer import GRPOTrainer
from utils import set_seed

set_seed(42)

class Args:
    pass

args = Args()
args.model_name = "{model_path}"
args.prompt = 'sequential'
args.method = 'latent_mas'
args.device = 'cuda:0'
args.device2 = 'cuda:2'  # Use GPU 2 for HF_model (GPU 0,1 used by vLLM TP=2)
args.think = False
args.latent_steps = 3
args.top_k_blocks = TOP_K
args.similarity_threshold = 0.85
args.min_block_size = 4
args.max_block_size = 64
args.segment_layer_idx = {segment_layer_idx}
args.tensor_parallel_size = 2  # Use 2 GPUs (0,1) for 8B vLLM
args.gpu_memory_utilization = 0.80  # Slightly lower to leave room
args.enable_prefix_caching = True
args.use_second_HF_model = True
args.latent_space_realign = False
args.max_new_tokens = 512
args.task = TASK

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)
hidden_dim = model.HF_model.config.hidden_size

print(f'\\nLoading TRAIN data...')
if TASK == "gsm8k":
    train_data = list(load_gsm8k(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
elif TASK == "arc_challenge":
    train_data = list(load_arc_challenge(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
elif TASK == "winogrande":
    train_data = list(load_winogrande(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
print(f'Loaded {{len(train_data)}} training samples')
sys.stdout.flush()

# Initialize policy (use same architecture, just hidden_dim changes)
policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)

method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=policy, top_k_blocks=TOP_K, rl_training=True,
)
method.task = TASK

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
reward_calculator = RewardCalculator(alpha=ALPHA, beta=BETA, gamma=GAMMA)
trainer = GRPOTrainer(
    policy_net=policy, optimizer=optimizer, reward_calculator=reward_calculator,
    group_size=4, clip_epsilon=0.2, kl_coef=KL_COEF, entropy_coef=ENTROPY_COEF,
    max_grad_norm=1.0, device=args.device,
)

best_acc = 0.0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    print(f'\\nEpoch {{epoch+1}}/{{NUM_EPOCHS}}')
    method.rl_training = True
    policy.train()

    current_group = []
    epoch_rewards = []
    epoch_correct = 0

    for i, item in enumerate(train_data):
        result = method.run_batch_vllm([item])[0]
        if result.get('correct', False):
            epoch_correct += 1

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

        if (i + 1) % 100 == 0:
            acc = epoch_correct / (i + 1)
            avg_r = sum(epoch_rewards[-100:]) / max(len(epoch_rewards[-100:]), 1)
            print(f'  {{i+1}}/{{len(train_data)}}: acc={{acc:.2%}}, reward={{avg_r:.3f}}')
            sys.stdout.flush()

    if current_group:
        trainer.buffer.groups.append(current_group)
        if trainer.buffer.groups:
            trainer.update_policy(trainer.buffer.groups)
            trainer.buffer.groups = []

    epoch_acc = epoch_correct / len(train_data)
    print(f'Epoch {{epoch+1}} done: acc={{epoch_acc:.2%}}')

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch + 1
        torch.save(policy.state_dict(), CHECKPOINT_PATH)
        print(f'  -> New best! Saved checkpoint.')

print(f'\\nBest epoch: {{best_epoch}} with acc={{best_acc:.2%}}')
print(f'Checkpoint saved: {{CHECKPOINT_PATH}}')
print("Training completed!")
EOF

echo "Training job completed!"
'''

# ==================== Evaluation Script ====================
EVAL_TEMPLATE = '''#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=eval_8b_{task}_k{top_k}
#SBATCH --output=logs/eval_v4_8b_{task}_topk{top_k}_%j.out
#SBATCH --error=logs/eval_v4_8b_{task}_topk{top_k}_%j.err

# V4 8B: Evaluate Qwen3-8B
# {task} with top_k={top_k}

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 << 'EOF'
import torch
import sys
import json
from datetime import datetime

TASK = "{task}"
TOP_K = {top_k}
TEST_SPLIT = "{test_split}"
TEST_SAMPLES = {test_samples}
CHECKPOINT_PATH = "checkpoints/ablation_v4_8b/{task}_topk{top_k}_policy.pt"

print("="*60)
print(f"EVALUATION V4 8B: {{TASK}} with top_k={{TOP_K}}")
print(f"Model: Qwen3-8B")
print(f"Test: {{TEST_SPLIT}} split ({{TEST_SAMPLES}} samples)")
print(f"Checkpoint: {{CHECKPOINT_PATH}}")
print("="*60)
sys.stdout.flush()

from data import load_gsm8k, load_arc_challenge, load_winogrande
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy import ReadingPolicyNetwork
from utils import set_seed

set_seed(42)

class Args:
    pass

args = Args()
args.model_name = "{model_path}"
args.prompt = 'sequential'
args.method = 'latent_mas'
args.device = 'cuda:0'
args.device2 = 'cuda:2'  # Use GPU 2 for HF_model (GPU 0,1 used by vLLM TP=2)
args.think = False
args.latent_steps = 3
args.top_k_blocks = TOP_K
args.similarity_threshold = 0.85
args.min_block_size = 4
args.max_block_size = 64
args.segment_layer_idx = {segment_layer_idx}
args.tensor_parallel_size = 2  # Use 2 GPUs (0,1) for 8B vLLM
args.gpu_memory_utilization = 0.80  # Slightly lower to leave room
args.enable_prefix_caching = True
args.use_second_HF_model = True
args.latent_space_realign = False
args.max_new_tokens = 512
args.task = TASK

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)
hidden_dim = model.HF_model.config.hidden_size

print(f'\\nLoading TEST data...')
if TASK == "gsm8k":
    test_data = list(load_gsm8k(split=TEST_SPLIT))[:TEST_SAMPLES]
elif TASK == "arc_challenge":
    test_data = list(load_arc_challenge(split=TEST_SPLIT))[:TEST_SAMPLES]
elif TASK == "winogrande":
    test_data = list(load_winogrande(split=TEST_SPLIT))[:TEST_SAMPLES]
print(f'Loaded {{len(test_data)}} test samples')
sys.stdout.flush()

# BASELINE
print('\\n' + '='*60)
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
        print(f'  Baseline: {{i+1}}/{{TEST_SAMPLES}}, acc={{baseline_correct}}/{{i+1}}')
        sys.stdout.flush()

baseline_acc = baseline_correct / TEST_SAMPLES
print(f'\\nBaseline: {{baseline_correct}}/{{TEST_SAMPLES}} = {{baseline_acc*100:.2f}}%')

# POLICY
print('\\n' + '='*60)
print(f'POLICY (top_k={{TOP_K}})')
print('='*60)

policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)
policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=args.device))
policy.eval()
print(f'Loaded checkpoint: {{CHECKPOINT_PATH}}')

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
        print(f'  Policy: {{i+1}}/{{TEST_SAMPLES}}, acc={{policy_correct}}/{{i+1}}')
        sys.stdout.flush()

policy_acc = policy_correct / TEST_SAMPLES
policy_read_ratio = policy_selected_blocks / max(policy_total_blocks, 1)
latent_reduction = (1 - policy_read_ratio) * 100
acc_change = (policy_acc - baseline_acc) * 100

print('\\n' + '='*60)
print('RESULTS')
print('='*60)
print(f'Task: {{TASK}}')
print(f'Model: Qwen3-8B')
print(f'top_k: {{TOP_K}}')
print(f'')
print(f'Baseline: {{baseline_acc*100:.2f}}% (read 100%)')
print(f'Policy:   {{policy_acc*100:.2f}}% (read {{policy_read_ratio*100:.2f}}%)')
print(f'')
print(f'Accuracy change: {{acc_change:+.2f}}%')
print(f'Latent reduction: {{latent_reduction:.1f}}%')
print('='*60)

results = {{
    'task': TASK,
    'model': 'Qwen3-8B',
    'top_k': TOP_K,
    'test_split': TEST_SPLIT,
    'test_samples': TEST_SAMPLES,
    'baseline_acc': baseline_acc,
    'policy_acc': policy_acc,
    'policy_read_ratio': policy_read_ratio,
    'accuracy_change': acc_change,
    'latent_reduction_pct': latent_reduction,
    'version': 'v4_8b',
    'timestamp': datetime.now().isoformat(),
}}
result_file = f'logs/ablation_v4_8b_{{TASK}}_topk{{TOP_K}}.json'
with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to {{result_file}}')
EOF

echo "Evaluation job completed!"
'''

# Generate scripts
train_scripts = []
eval_scripts = []

for task, cfg in DATASETS.items():
    for top_k in TOP_K_VALUES:
        # Training script
        train_content = TRAIN_TEMPLATE.format(
            task=task, top_k=top_k,
            model_path=MODEL_PATH,
            train_split=cfg["train_split"],
            train_samples=cfg["train_samples"],
            segment_layer_idx=SEGMENT_LAYER_IDX,
            lr=LEARNING_RATE,
            kl=KL_COEF,
            entropy=ENTROPY_COEF,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
            epochs=NUM_EPOCHS,
        )
        train_path = os.path.join(SCRIPT_DIR, f"train_v4_8b_{task}_topk{top_k}.sh")
        with open(train_path, "w") as f:
            f.write(train_content)
        train_scripts.append(train_path)

        # Evaluation script
        eval_content = EVAL_TEMPLATE.format(
            task=task, top_k=top_k,
            model_path=MODEL_PATH,
            train_samples=cfg["train_samples"],
            test_split=cfg["test_split"],
            test_samples=cfg["test_samples"],
            segment_layer_idx=SEGMENT_LAYER_IDX,
        )
        eval_path = os.path.join(SCRIPT_DIR, f"eval_v4_8b_{task}_topk{top_k}.sh")
        with open(eval_path, "w") as f:
            f.write(eval_content)
        eval_scripts.append(eval_path)

print("=" * 60)
print("ABLATION V4 - Qwen3-8B")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"top_k values: {TOP_K_VALUES}")
print(f"Datasets: {list(DATASETS.keys())}")
print("=" * 60)

# Submission scripts
submit_train = '''#!/bin/bash
# V4 8B Training - Qwen3-8B experiments

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation_v4_8b

echo "=============================================="
echo "ABLATION V4 - Qwen3-8B"
echo "=============================================="
echo "Training with key top_k values: 10, 40, 50"
echo ""

'''
for s in train_scripts:
    name = os.path.basename(s)
    submit_train += f'echo "Submitting {name}..."\nsbatch {s}\n'

submit_train += '''
echo ""
echo "Training jobs submitted!"
echo "After training, run: bash scripts/submit_eval_ablation_v4_8b.sh"
'''

with open(os.path.join(SCRIPT_DIR, "submit_train_ablation_v4_8b.sh"), "w") as f:
    f.write(submit_train)
os.chmod(os.path.join(SCRIPT_DIR, "submit_train_ablation_v4_8b.sh"), 0o755)

submit_eval = '''#!/bin/bash
# V4 8B Evaluation

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "Submitting V4 8B evaluation jobs..."

'''
for s in eval_scripts:
    name = os.path.basename(s)
    submit_eval += f'echo "Submitting {name}..."\nsbatch {s}\n'

submit_eval += '''
echo ""
echo "Evaluation jobs submitted!"
echo "Results: logs/ablation_v4_8b_*.json"
'''

with open(os.path.join(SCRIPT_DIR, "submit_eval_ablation_v4_8b.sh"), "w") as f:
    f.write(submit_eval)
os.chmod(os.path.join(SCRIPT_DIR, "submit_eval_ablation_v4_8b.sh"), 0o755)

print("\nGenerated scripts:")
for s in train_scripts:
    print(f"  {os.path.basename(s)}")
print("\nTo run:")
print("  bash scripts/submit_train_ablation_v4_8b.sh")
