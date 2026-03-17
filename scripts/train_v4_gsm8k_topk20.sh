#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=train_v4_gsm8k_k20
#SBATCH --output=logs/train_v4_gsm8k_topk20_%j.out
#SBATCH --error=logs/train_v4_gsm8k_topk20_%j.err

# V4: Pure task optimization - find optimal accuracy-efficiency trade-off
# gsm8k with top_k=20

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation_v4

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 << 'EOF'
import torch
import sys
import json
from datetime import datetime

TASK = "gsm8k"
TOP_K = 20
TRAIN_SPLIT = "train"
TRAIN_SAMPLES = 500
CHECKPOINT_PATH = "checkpoints/ablation_v4/gsm8k_topk20_policy.pt"

# V4 Hyperparameters - pure task optimization
LR = 1e-05
KL_COEF = 0.0
ENTROPY_COEF = 0.01
ALPHA = 1.0  # Task reward weight
BETA = 0.3    # Consistency weight
GAMMA = 0.0  # NO cost penalty!
NUM_EPOCHS = 5

print("="*60)
print(f"TRAINING V4: {TASK} with top_k={TOP_K}")
print(f"Train: {TRAIN_SPLIT} split ({TRAIN_SAMPLES} samples)")
print(f"Reward: R = {ALPHA}*R1 + {BETA}*R2 (NO COST PENALTY!)")
print(f"Goal: Accuracy drop < 5%, find smallest top_k")
print(f"LR={LR}, KL={KL_COEF}, epochs={NUM_EPOCHS}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
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

# Load TRAIN data
print(f'\nLoading TRAIN data...')
if TASK == "gsm8k":
    train_data = list(load_gsm8k(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
elif TASK == "arc_challenge":
    train_data = list(load_arc_challenge(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
elif TASK == "winogrande":
    train_data = list(load_winogrande(split=TRAIN_SPLIT))[:TRAIN_SAMPLES]
print(f'Loaded {len(train_data)} training samples')
sys.stdout.flush()

# Initialize policy
policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)

# Create method
method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=policy, top_k_blocks=TOP_K, rl_training=True,
)
method.task = TASK

# Training setup - pure task optimization
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
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
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
            print(f'  {i+1}/{len(train_data)}: acc={acc:.2%}, reward={avg_r:.3f}')
            sys.stdout.flush()

    if current_group:
        trainer.buffer.groups.append(current_group)
        if trainer.buffer.groups:
            trainer.update_policy(trainer.buffer.groups)
            trainer.buffer.groups = []

    epoch_acc = epoch_correct / len(train_data)
    print(f'Epoch {epoch+1} done: acc={epoch_acc:.2%}')

    # Save best checkpoint
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch + 1
        torch.save(policy.state_dict(), CHECKPOINT_PATH)
        print(f'  -> New best! Saved checkpoint.')

print(f'\nBest epoch: {best_epoch} with acc={best_acc:.2%}')
print(f'Checkpoint saved: {CHECKPOINT_PATH}')
print("Training completed!")
EOF

echo "Training job completed!"
