#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=latentmas_rl_test
#SBATCH --output=logs/rl_train_%j.out
#SBATCH --error=logs/rl_train_%j.err

# ============================================================
# Quick Test Script - 使用较小的模型和数据量进行测试
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

# Environment Setup
source ~/.bashrc
conda activate latentmas

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs
mkdir -p checkpoints/rl_policy

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Quick test with smaller model (use local path)
MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen3-8B"
TASK="gsm8k"
OUTPUT_DIR="checkpoints/rl_policy/test_${TASK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Quick Test Run..."
echo "Model: $MODEL_NAME"
echo "Task: $TASK"

python run_rl_train.py \
    --model_name $MODEL_NAME \
    --task $TASK \
    --prompt sequential \
    --device cuda:0 \
    --device2 cuda:1 \
    --use_vllm \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.8 \
    --max_new_tokens 2048 \
    --latent_steps 5 \
    --temperature 0.6 \
    --top_p 0.95 \
    --think \
    --top_k_blocks 4 \
    --policy_num_heads 4 \
    --policy_num_layers 1 \
    --policy_lr 1e-4 \
    --grpo_group_size 4 \
    --num_epochs 2 \
    --max_samples 100 \
    --eval_samples 20 \
    --seed 42 \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/console_output.log

echo ""
echo "Test Complete! Results: $OUTPUT_DIR"
