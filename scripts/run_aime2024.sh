#!/bin/bash
#SBATCH --job-name=rl_aime2024
#SBATCH --output=logs/rl_aime2024_%j.out
#SBATCH --error=logs/rl_aime2024_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=westai0052

# ============================================================
# AIME 2024 RL Training Script
#
# 提交方式:
#   sbatch scripts/run_aime2024.sh
#
# 查看状态:
#   squeue -u $USER
#   tail -f logs/rl_aime2024_<JOB_ID>.out
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Task: AIME 2024"
echo "Start Time: $(date)"
echo "=========================================="

# Environment
source ~/.bashrc
conda activate latentmas

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/rl_policy

# CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# GPU Info
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Configuration
MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen3-14B"
TASK="aime2024"
OUTPUT_DIR="checkpoints/rl_policy/${TASK}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Task: $TASK"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run Training
echo "Starting RL Training on AIME 2024..."
echo ""

python run_rl_train.py \
    --model_name $MODEL_NAME \
    --task $TASK \
    --prompt sequential \
    --method latent_mas \
    --device cuda:0 \
    --device2 cuda:2 \
    --use_vllm \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.85 \
    --max_new_tokens 4096 \
    --latent_steps 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    --think \
    --top_k_blocks 4 \
    --policy_num_heads 8 \
    --policy_num_layers 2 \
    --policy_dropout 0.1 \
    --similarity_threshold 0.85 \
    --min_block_size 4 \
    --max_block_size 64 \
    --segment_layer_idx 16 \
    --policy_lr 1e-5 \
    --grpo_group_size 4 \
    --grpo_clip_epsilon 0.2 \
    --grpo_kl_coef 0.1 \
    --entropy_coef 0.01 \
    --ref_policy_update_freq 50 \
    --update_every_n_groups 2 \
    --reward_alpha 1.0 \
    --reward_beta 0.5 \
    --reward_gamma 0.1 \
    --num_epochs 3 \
    --max_samples -1 \
    --eval_samples 5 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --output_dir $OUTPUT_DIR \
    --log_file $OUTPUT_DIR/training_log.json \
    2>&1 | tee $OUTPUT_DIR/console_output.log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
