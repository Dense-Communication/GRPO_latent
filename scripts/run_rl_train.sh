#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=latentmas_rl
#SBATCH --output=logs/rl_train_%j.out
#SBATCH --error=logs/rl_train_%j.err

# ============================================================
# RL Training Script for LatentMAS Reading Policy
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

# ------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------

# Load conda (根据你的系统修改路径)
source ~/.bashrc
# 或者使用: source /path/to/conda/etc/profile.d/conda.sh

# Activate conda environment (修改为你的环境名称)
conda activate latentmas
# 或者: conda activate your_env_name

# Set working directory
cd /p/scratch/westai0052/liu52/LatentMAS

# Create logs directory if not exists
mkdir -p logs
mkdir -p checkpoints/rl_policy

# ------------------------------------------------------------
# CUDA Settings
# ------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print GPU info
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# ------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------

MODEL_NAME="Qwen/Qwen3-14B"
TASK="gsm8k"
OUTPUT_DIR="checkpoints/rl_policy/${TASK}_$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR

# ------------------------------------------------------------
# Run Training
# ------------------------------------------------------------

echo "Starting RL Training..."
echo "Model: $MODEL_NAME"
echo "Task: $TASK"
echo "Output: $OUTPUT_DIR"
echo ""

python run_rl_train.py \
    --model_name $MODEL_NAME \
    --task $TASK \
    --prompt sequential \
    --device cuda:0 \
    --device2 cuda:1 \
    --use_vllm \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.85 \
    --max_new_tokens 4096 \
    --latent_steps 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    --think \
    --latent_space_realign \
    --top_k_blocks 4 \
    --policy_num_heads 8 \
    --policy_num_layers 2 \
    --policy_dropout 0.1 \
    --similarity_threshold 0.85 \
    --min_block_size 4 \
    --max_block_size 64 \
    --segment_layer_idx 16 \
    --policy_lr 1e-5 \
    --grpo_group_size 8 \
    --grpo_clip_epsilon 0.2 \
    --grpo_kl_coef 0.1 \
    --entropy_coef 0.01 \
    --ref_policy_update_freq 100 \
    --update_every_n_groups 4 \
    --reward_alpha 1.0 \
    --reward_beta 0.5 \
    --reward_gamma 0.1 \
    --num_epochs 5 \
    --max_samples 500 \
    --eval_samples 50 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --output_dir $OUTPUT_DIR \
    --log_file $OUTPUT_DIR/training_log.json \
    2>&1 | tee $OUTPUT_DIR/console_output.log

# ------------------------------------------------------------
# Completion
# ------------------------------------------------------------

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Copy final results to a summary file
if [ -f "$OUTPUT_DIR/training_log.json" ]; then
    echo "Training log:"
    cat $OUTPUT_DIR/training_log.json
fi
