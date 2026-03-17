#!/bin/bash
# ============================================================
# 完整版 RL 训练脚本
#
# 使用方法:
#   bash scripts/run_full_training.sh [TASK]
#
# 支持的数据集:
#   gsm8k         - 数学推理 (默认, 已缓存)
#   medqa         - 医学问答 (本地数据)
#   arc_easy      - ARC Easy 科学问答
#   arc_challenge - ARC Challenge 科学问答
#   gpqa          - GPQA Diamond 研究生问答
#   aime2024      - AIME 2024 数学竞赛
#   aime2025      - AIME 2025 数学竞赛
#
# 示例:
#   bash scripts/run_full_training.sh gsm8k
#   bash scripts/run_full_training.sh medqa
#   bash scripts/run_full_training.sh arc_challenge
# ============================================================

TASK=${1:-gsm8k}

echo "=========================================="
echo "RL Training - Full Run"
echo "Node: $(hostname)"
echo "Task: $TASK"
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

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Configuration
MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen3-14B"
OUTPUT_DIR="checkpoints/rl_policy/${TASK}_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Task: $TASK"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run Training
echo "Starting Full RL Training..."
echo ""

python run_rl_train.py \
    --model_name $MODEL_NAME \
    --task $TASK \
    --prompt sequential \
    --method latent_mas \
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

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
