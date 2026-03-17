#!/usr/bin/env python3
"""
Top-K Ablation Experiment v3 - Practical improvements

Key insight: LatentMem uses 40K trajectories to train a full LLM (Memory Composer),
but our Reading Policy Network is much smaller (cross-attention network).
We don't need as much data, but we need better training signal.

Changes from v2:
1. Training samples: 300 -> 500 (reasonable increase)
2. Learning rate: 5e-5 -> 1e-5 (more stable)
3. KL coefficient: 0.05 -> 0.0 (remove KL penalty per LatentMem)
4. Cost penalty gamma: 0.1 -> 0.02 (much smaller - prioritize accuracy!)
5. Training epochs: 3 -> 5
6. Alpha (task reward): 1.0 -> 2.0 (emphasize correctness more)
"""

import os

SCRIPT_DIR = "/p/scratch/westai0052/liu52/LatentMAS/scripts"
os.makedirs(SCRIPT_DIR, exist_ok=True)

MODEL_NAME = "Qwen3-4B"
MODEL_PATH = "/p/scratch/westai0052/liu52/models/Qwen3-4B"
SEGMENT_LAYER_IDX = 16

DATASETS = {
    "gsm8k": {"train_split": "train", "test_split": "test", "train_samples": 500, "test_samples": 500},
    "arc_challenge": {"train_split": "train", "test_split": "test", "train_samples": 500, "test_samples": 500},
    "winogrande": {"train_split": "train", "test_split": "validation", "train_samples": 500, "test_samples": 500},
}

TOP_K_VALUES = [5, 10]

# Improved hyperparameters - focus on maintaining accuracy
LEARNING_RATE = 1e-5      # More stable
KL_COEF = 0.0             # No KL penalty
ENTROPY_COEF = 0.01
ALPHA = 2.0               # Increased task reward weight (was 1.0)
BETA = 0.3                # Evidence consistency
GAMMA = 0.02              # Much smaller cost penalty (was 0.1)
NUM_EPOCHS = 5            # More epochs

# ==================== STEP 1: Training Script ====================
TRAIN_TEMPLATE = '''#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=train_v3_{task}_k{top_k}
#SBATCH --output=logs/train_v3_{task}_topk{top_k}_%j.out
#SBATCH --error=logs/train_v3_{task}_topk{top_k}_%j.err

# V3: Focus on accuracy - smaller cost penalty, larger task reward
# {task} with top_k={top_k}

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation_v3

export CUDA_VISIBLE_DEVICES=0,1
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
CHECKPOINT_PATH = "checkpoints/ablation_v3/{task}_topk{top_k}_policy.pt"

# Hyperparameters focused on accuracy
LR = {lr}
KL_COEF = {kl}
ENTROPY_COEF = {entropy}
ALPHA = {alpha}  # Task reward weight
BETA = {beta}    # Consistency weight
GAMMA = {gamma}  # Cost penalty (small!)
NUM_EPOCHS = {epochs}

print("="*60)
print(f"TRAINING V3: {{TASK}} with top_k={{TOP_K}}")
print(f"Train: {{TRAIN_SPLIT}} split ({{TRAIN_SAMPLES}} samples)")
print(f"Reward: R = {{ALPHA}}*R1 + {{BETA}}*R2 - {{GAMMA}}*R3")
print(f"LR={{LR}}, KL={{KL_COEF}}, epochs={{NUM_EPOCHS}}")
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
    model_name = "{model_path}"
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

# Load TRAIN data
print(f'\\nLoading TRAIN data...')
if TASK == "gsm8k":
    train_data = list(load_gsm8k(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
elif TASK == "arc_challenge":
    train_data = list(load_arc_challenge(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
elif TASK == "winogrande":
    train_data = list(load_winogrande(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
print(f'Loaded {{len(train_data)}} training samples')
sys.stdout.flush()

# Initialize policy
policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)

# Create method
method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=policy, top_k_blocks=TOP_K, rl_training=True,
)
method.task = TASK

# Training setup - emphasize task correctness!
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
reward_calculator = RewardCalculator(alpha=ALPHA, beta=BETA, gamma=GAMMA)
trainer = GRPOTrainer(
    policy_net=policy, optimizer=optimizer, reward_calculator=reward_calculator,
    group_size=4, clip_epsilon=0.2, kl_coef=KL_COEF, entropy_coef=ENTROPY_COEF,
    max_grad_norm=1.0, device=args.device,
)

best_acc = 0.0
best_epoch = 0

# Train for NUM_EPOCHS epochs
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

    # Save best checkpoint
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

# ==================== STEP 2: Evaluation Script ====================
EVAL_TEMPLATE = '''#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_v3_{task}_k{top_k}
#SBATCH --output=logs/eval_v3_{task}_topk{top_k}_%j.out
#SBATCH --error=logs/eval_v3_{task}_topk{top_k}_%j.err

# V3: Evaluate with improved policy
# {task} with top_k={top_k}

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

TASK = "{task}"
TOP_K = {top_k}
TEST_SPLIT = "{test_split}"
TEST_SAMPLES = {test_samples}
CHECKPOINT_PATH = "checkpoints/ablation_v3/{task}_topk{top_k}_policy.pt"

print("="*60)
print(f"EVALUATION V3: {{TASK}} with top_k={{TOP_K}}")
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
    model_name = "{model_path}"
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

# Load TEST data
print(f'\\nLoading TEST data...')
if TASK == "gsm8k":
    test_data = list(load_gsm8k(split=TEST_SPLIT))[:TEST_SAMPLES]
elif TASK == "arc_challenge":
    test_data = list(load_arc_challenge(split=TEST_SPLIT))[:TEST_SAMPLES]
elif TASK == "winogrande":
    test_data = list(load_winogrande(split=TEST_SPLIT))[:TEST_SAMPLES]
print(f'Loaded {{len(test_data)}} test samples')
sys.stdout.flush()

# ========== BASELINE (read ALL blocks) ==========
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

# ========== POLICY (read top_k blocks) ==========
print('\\n' + '='*60)
print(f'POLICY (top_k={{TOP_K}})')
print('='*60)

# Load trained policy
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
print(f'top_k: {{TOP_K}}')
print(f'')
print(f'Baseline: {{baseline_acc*100:.2f}}% (read 100%)')
print(f'Policy:   {{policy_acc*100:.2f}}% (read {{policy_read_ratio*100:.2f}}%)')
print(f'')
print(f'Accuracy change: {{acc_change:+.2f}}%')
print(f'Latent reduction: {{latent_reduction:.1f}}%')
print('='*60)

# Save results
results = {{
    'task': TASK,
    'top_k': TOP_K,
    'test_split': TEST_SPLIT,
    'test_samples': TEST_SAMPLES,
    'baseline_acc': baseline_acc,
    'policy_acc': policy_acc,
    'policy_read_ratio': policy_read_ratio,
    'accuracy_change': acc_change,
    'latent_reduction_pct': latent_reduction,
    'version': 'v3',
    'hyperparams': {{
        'train_samples': {train_samples},
        'lr': {lr},
        'alpha': {alpha},
        'beta': {beta},
        'gamma': {gamma},
        'epochs': {epochs},
    }},
    'timestamp': datetime.now().isoformat(),
}}
result_file = f'logs/ablation_v3_{{TASK}}_topk{{TOP_K}}.json'
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
        train_path = os.path.join(SCRIPT_DIR, f"train_v3_{task}_topk{top_k}.sh")
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
            lr=LEARNING_RATE,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
            epochs=NUM_EPOCHS,
        )
        eval_path = os.path.join(SCRIPT_DIR, f"eval_v3_{task}_topk{top_k}.sh")
        with open(eval_path, "w") as f:
            f.write(eval_content)
        eval_scripts.append(eval_path)

print("="*60)
print("ABLATION V3 - Focus on Accuracy")
print("="*60)
print(f"Key insight: Our Reading Policy is a small network,")
print(f"not a full LLM like LatentMem's Memory Composer.")
print(f"So we focus on better reward design, not more data.")
print("="*60)
print(f"Training samples: 500 (was 300)")
print(f"Learning rate: {LEARNING_RATE} (was 5e-5)")
print(f"Task reward alpha: {ALPHA} (was 1.0) <- emphasize correctness!")
print(f"Cost penalty gamma: {GAMMA} (was 0.1) <- much smaller!")
print(f"KL coefficient: {KL_COEF} (was 0.05)")
print(f"Training epochs: {NUM_EPOCHS} (was 3)")
print("="*60)

print("\nGenerated training scripts:")
for s in train_scripts:
    print(f"  {s}")

print("\nGenerated evaluation scripts:")
for s in eval_scripts:
    print(f"  {s}")

# Submission script
submit_content = '''#!/bin/bash
# V3 Ablation - Focus on Accuracy
# Key changes: alpha=2.0, gamma=0.02

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation_v3

echo "=============================================="
echo "ABLATION V3 - Focus on Accuracy"
echo "=============================================="
echo "Reward: R = 2.0*R1 + 0.3*R2 - 0.02*R3"
echo "(Task reward doubled, cost penalty reduced 5x)"
echo ""
echo "Training samples: 500"
echo "Learning rate: 1e-5"
echo "Epochs: 5"
echo "=============================================="
echo ""
echo "STEP 1: Submitting training jobs..."
echo ""

'''

for s in train_scripts:
    name = os.path.basename(s)
    submit_content += f'echo "Submitting {name}..."\n'
    submit_content += f'sbatch {s}\n'

submit_content += '''
echo ""
echo "======================================"
echo "Training jobs submitted!"
echo ""
echo "After training completes, run:"
echo "  bash scripts/submit_eval_ablation_v3.sh"
'''

submit_train_path = os.path.join(SCRIPT_DIR, "submit_train_ablation_v3.sh")
with open(submit_train_path, "w") as f:
    f.write(submit_content)
os.chmod(submit_train_path, 0o755)

# Eval submission script
eval_submit = '''#!/bin/bash
# V3 Ablation - Submit evaluation jobs

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "STEP 2: Submitting V3 evaluation jobs..."
echo "======================================"

'''

for s in eval_scripts:
    name = os.path.basename(s)
    eval_submit += f'echo "Submitting {name}..."\n'
    eval_submit += f'sbatch {s}\n'

eval_submit += '''
echo ""
echo "======================================"
echo "Evaluation jobs submitted!"
echo "Results will be in: logs/ablation_v3_*.json"
'''

submit_eval_path = os.path.join(SCRIPT_DIR, "submit_eval_ablation_v3.sh")
with open(submit_eval_path, "w") as f:
    f.write(eval_submit)
os.chmod(submit_eval_path, 0o755)

print(f"\nSubmission scripts:")
print(f"  {submit_train_path}")
print(f"  {submit_eval_path}")
print(f"\nTo run:")
print(f"  1. bash scripts/submit_train_ablation_v3.sh")
print(f"  2. (wait for training to complete)")
print(f"  3. bash scripts/submit_eval_ablation_v3.sh")
