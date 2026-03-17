#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=westai0052

# ============================================================
# Baseline Comparison: 原 LatentMAS (无 Reading Policy)
#
# 目的: 获取对比基准数据
# 测试: GSM8K, GPQA, AIME2024
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Task: Baseline Comparison (No Reading Policy)"
echo "Start Time: $(date)"
echo "=========================================="

source ~/.bashrc
conda activate latentmas

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/baseline

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen3-14B"
OUTPUT_DIR="checkpoints/baseline/baseline_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# ============================================================
# Test 1: GSM8K Baseline (200 samples)
# ============================================================
echo "=========================================="
echo "Test 1: GSM8K Baseline"
echo "=========================================="

python run.py \
    --method latent_mas \
    --model_name $MODEL_NAME \
    --task gsm8k \
    --prompt sequential \
    --max_samples 200 \
    --max_new_tokens 2048 \
    --latent_steps 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    2>&1 | tee $OUTPUT_DIR/gsm8k_baseline.log

echo ""

# ============================================================
# Test 2: GPQA Baseline (全部 ~198 samples)
# ============================================================
echo "=========================================="
echo "Test 2: GPQA Baseline"
echo "=========================================="

python run.py \
    --method latent_mas \
    --model_name $MODEL_NAME \
    --task gpqa \
    --prompt sequential \
    --max_samples -1 \
    --max_new_tokens 2048 \
    --latent_steps 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    2>&1 | tee $OUTPUT_DIR/gpqa_baseline.log

echo ""

# ============================================================
# Test 3: AIME 2024 Baseline (30 samples)
# ============================================================
echo "=========================================="
echo "Test 3: AIME 2024 Baseline"
echo "=========================================="

python run.py \
    --method latent_mas \
    --model_name $MODEL_NAME \
    --task aime2024 \
    --prompt sequential \
    --max_samples -1 \
    --max_new_tokens 4096 \
    --latent_steps 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    2>&1 | tee $OUTPUT_DIR/aime2024_baseline.log

echo ""
echo "=========================================="
echo "Baseline Comparison Complete!"
echo "End Time: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
