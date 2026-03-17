#!/bin/bash
# ============================================================
# 交互式节点本地运行 - 快速测试版本
#
# 使用方法:
#   cd /p/scratch/westai0052/liu52/LatentMAS
#   bash scripts/run_local_test.sh
# ============================================================

echo "=========================================="
echo "RL Training - Quick Test Run"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

# Environment
source ~/.bashrc
conda activate latentmas

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/rl_policy

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo ""
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

# Quick test with smaller settings
# 使用本地模型路径 (避免网络下载)
MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen3-8B"
TASK="gsm8k"
OUTPUT_DIR="checkpoints/rl_policy/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Quick Test Config:"
echo "  Model: $MODEL_NAME (using cached model)"
echo "  Samples: 50"
echo "  Output: $OUTPUT_DIR"
echo ""

python run_rl_train.py \
    --model_name $MODEL_NAME \
    --task $TASK \
    --prompt sequential \
    --method latent_mas \
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
    --num_epochs 1 \
    --max_samples 50 \
    --eval_samples 10 \
    --seed 42 \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/console.log

echo ""
echo "Done! Results: $OUTPUT_DIR"
