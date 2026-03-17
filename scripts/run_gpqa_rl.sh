#!/bin/bash
#SBATCH --job-name=rl_gpqa
#SBATCH --output=logs/rl_gpqa_%j.out
#SBATCH --error=logs/rl_gpqa_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=westai0052

# ============================================================
# GPQA Diamond RL Training Script
#
# 目标: 证明 reading policy 能提升 LatentMAS 性能
# 数据集: GPQA Diamond (448 样本, 科学问答)
# 原论文 baseline: 53.1%
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Task: GPQA Diamond RL Training"
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
TASK="gpqa"
OUTPUT_DIR="checkpoints/rl_policy/${TASK}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Task: $TASK (GPQA Diamond)"
echo "  Dataset size: 448 samples"
echo "  Original baseline: 53.1%"
echo "  Output: $OUTPUT_DIR"
echo ""

# ============================================================
# Train Reading Policy with GRPO
# 使用 350 样本训练, 98 样本评估 (约 78% / 22% 划分)
# ============================================================
echo "Training Reading Policy on GPQA..."
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
    --max_new_tokens 2048 \
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
    --num_epochs 5 \
    --max_samples -1 \
    --eval_samples 98 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --output_dir $OUTPUT_DIR \
    --log_file $OUTPUT_DIR/training_log.json \
    2>&1 | tee $OUTPUT_DIR/training_output.log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Original baseline: 53.1%"
echo "Check training_log.json for new accuracy"
echo "=========================================="
