#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=reward_abl
#SBATCH --output=logs/reward_ablation_%j.out
#SBATCH --error=logs/reward_ablation_%j.err

# Reward Ablation: Compare different α, β, γ combinations on GSM8K

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/reward_ablation checkpoints/reward_ablation

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 << 'EOF'
import torch
import sys
import json
import os
from datetime import datetime

# Reward configurations to test
# (name, alpha, beta, gamma)
REWARD_CONFIGS = [
    ('task_only', 1.0, 0.0, 0.0),       # Only task reward
    ('consistency_only', 0.0, 1.0, 0.0), # Only consistency reward
    ('task_focused', 0.7, 0.3, 0.0),    # Focus on task
    ('with_cost', 0.5, 0.5, 0.1),       # Add cost penalty
]
# Note: balanced (0.5, 0.5, 0.0) is already in V4 results

TASK = "gsm8k"
TOP_K = 40
TRAIN_SAMPLES = 500
TEST_SAMPLES = 500
NUM_EPOCHS = 5
LR = 1e-5

print("="*60)
print("REWARD ABLATION STUDY")
print(f"Task: {TASK}, top_k={TOP_K}")
print("="*60)
sys.stdout.flush()

from data import load_gsm8k
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

print('Loading data...')
train_data = list(load_gsm8k(split="train"))[:TRAIN_SAMPLES]
test_data = list(load_gsm8k(split="test"))[:TEST_SAMPLES]
print(f'Train: {len(train_data)}, Test: {len(test_data)}')

os.makedirs('logs/reward_ablation', exist_ok=True)
os.makedirs('checkpoints/reward_ablation', exist_ok=True)

for config_name, alpha, beta, gamma in REWARD_CONFIGS:
    print('\n' + '='*60)
    print(f'TRAINING: {config_name} (α={alpha}, β={beta}, γ={gamma})')
    print('='*60)
    sys.stdout.flush()

    # Initialize policy
    policy = ReadingPolicyNetwork(
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=1
    ).to(args.device).to(torch.bfloat16)

    # Create method
    method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=policy, top_k_blocks=TOP_K, rl_training=True,
    )
    method.task = TASK

    # Training setup with specific reward weights
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    reward_calculator = RewardCalculator(alpha=alpha, beta=beta, gamma=gamma)
    trainer = GRPOTrainer(
        policy_net=policy, optimizer=optimizer, reward_calculator=reward_calculator,
        group_size=4, clip_epsilon=0.2, kl_coef=0.0, entropy_coef=0.01,
        max_grad_norm=1.0, device=args.device,
    )

    checkpoint_path = f'checkpoints/reward_ablation/gsm8k_{config_name}_policy.pt'
    best_acc = 0.0

    # Train
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

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(policy.state_dict(), checkpoint_path)
            print(f'  -> New best! Saved.')

    print(f'\nTraining done for {config_name}, best_acc={best_acc:.2%}')

    # Evaluate
    print('\n' + '='*60)
    print(f'EVALUATING: {config_name}')
    print('='*60)

    # Baseline
    print('Running baseline...')
    baseline_method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=None, top_k_blocks=TOP_K, rl_training=False,
    )
    baseline_method.task = TASK

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
    policy.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    policy.eval()

    test_method = LatentMASMethodRL(
        model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
        args=args, reading_policy=policy, top_k_blocks=TOP_K, rl_training=False,
    )
    test_method.task = TASK

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

    print(f'\nResults for {config_name}:')
    print(f'  Baseline: {baseline_acc*100:.2f}%')
    print(f'  Policy:   {policy_acc*100:.2f}% (Δ{acc_change:+.2f}%)')
    print(f'  Latent Reduction: {latent_reduction:.1f}%')

    results = {
        'task': TASK,
        'config_name': config_name,
        'reward_weights': {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
        },
        'top_k': TOP_K,
        'train_samples': TRAIN_SAMPLES,
        'test_samples': TEST_SAMPLES,
        'baseline_acc': baseline_acc,
        'policy_acc': policy_acc,
        'policy_read_ratio': read_ratio,
        'accuracy_change': acc_change,
        'latent_reduction_pct': latent_reduction,
        'timestamp': datetime.now().isoformat(),
    }

    result_file = f'logs/reward_ablation/gsm8k_{config_name}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved: {result_file}')

print('\n' + '='*60)
print('REWARD ABLATION COMPLETE!')
print('='*60)
EOF

echo "Done!"
