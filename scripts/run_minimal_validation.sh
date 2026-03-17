#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=min_val
#SBATCH --output=logs/minimal_val_%j.out
#SBATCH --error=logs/minimal_val_%j.err

# ============================================================
# TIME BUDGET: 3 hours
#
# Actual runtime (1.5B model, 200 samples per task):
#   - GSM8K: ~3 min (train) + ~1 min (baseline) + ~1 min (eval) = ~5 min
#   - ARC:   ~3 min + ~1 min + ~1 min = ~5 min
#   - Wino:  ~2 min + ~1 min + ~1 min = ~4 min
#   - Total: ~15 min actual
#   - With loading overhead: ~30 min
#   - Safety margin (6x): 3 hours
# ============================================================

# ============================================================
# Minimal Validation Script - Comprehensive Dataset Test
#
# Purpose: Quick validation with smallest model (1.5B) to verify
#          Reading Policy optimization is working correctly
#
# Uses 3 diverse datasets with enough samples for generalization:
#   1. GSM8K (math, 7473 train) - numerical reasoning
#   2. ARC-Challenge (science, 1119 train) - A/B/C/D choice
#   3. Winogrande (commonsense, 9248 train) - 1/2 choice
#
# Avoids tiny datasets: GPQA (198), AIME (30)
# ============================================================

echo "=========================================="
echo "Minimal Validation Test - 3 Datasets"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

source ~/.bashrc
conda activate latentmas

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs
mkdir -p checkpoints/rl_policy

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

# Smallest model for fast testing
MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

# Common parameters for minimal test
COMMON_ARGS="
    --model_name $MODEL_NAME
    --prompt sequential
    --method latent_mas
    --device cuda:0
    --device2 cuda:1
    --use_vllm
    --tensor_parallel_size 1
    --gpu_memory_utilization 0.85
    --max_new_tokens 512
    --latent_steps 3
    --temperature 0.6
    --top_p 0.95
    --top_k_blocks 3
    --policy_num_heads 4
    --policy_num_layers 1
    --policy_lr 1e-4
    --grpo_group_size 4
    --num_epochs 1
    --seed 42
"

# ============================================================
# Test 1: GSM8K (Math - 7473 train samples)
# ============================================================
echo ""
echo "=========================================="
echo "[1/3] GSM8K (Math - Numerical Reasoning)"
echo "  Train: 150 samples, Eval: 50 samples"
echo "=========================================="

TASK="gsm8k"
OUTPUT_DIR="checkpoints/rl_policy/minval_${TASK}_${JOB_ID}"
mkdir -p $OUTPUT_DIR

python run_rl_train.py \
    $COMMON_ARGS \
    --task $TASK \
    --max_samples 200 \
    --eval_samples 50 \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/console_output.log

echo "[1/3] GSM8K Done -> $OUTPUT_DIR"

# ============================================================
# Test 2: ARC-Challenge (Science - 1119 train samples)
# ============================================================
echo ""
echo "=========================================="
echo "[2/3] ARC-Challenge (Science - A/B/C/D)"
echo "  Train: 150 samples, Eval: 50 samples"
echo "=========================================="

TASK="arc_challenge"
OUTPUT_DIR="checkpoints/rl_policy/minval_${TASK}_${JOB_ID}"
mkdir -p $OUTPUT_DIR

python run_rl_train.py \
    $COMMON_ARGS \
    --task $TASK \
    --max_samples 200 \
    --eval_samples 50 \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/console_output.log

echo "[2/3] ARC-Challenge Done -> $OUTPUT_DIR"

# ============================================================
# Test 3: Winogrande (Commonsense - 9248 train samples)
# ============================================================
echo ""
echo "=========================================="
echo "[3/3] Winogrande (Commonsense - 1/2 Choice)"
echo "  Train: 150 samples, Eval: 50 samples"
echo "=========================================="

TASK="winogrande"
OUTPUT_DIR="checkpoints/rl_policy/minval_${TASK}_${JOB_ID}"
mkdir -p $OUTPUT_DIR

python run_rl_train.py \
    $COMMON_ARGS \
    --task $TASK \
    --max_samples 200 \
    --eval_samples 50 \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/console_output.log

echo "[3/3] Winogrande Done -> $OUTPUT_DIR"

# ============================================================
# Summary Report with Baseline Comparison
# ============================================================
echo ""
echo "=========================================="
echo "Minimal Validation Complete!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  1. GSM8K:         checkpoints/rl_policy/minval_gsm8k_${JOB_ID}/"
echo "  2. ARC-Challenge: checkpoints/rl_policy/minval_arc_challenge_${JOB_ID}/"
echo "  3. Winogrande:    checkpoints/rl_policy/minval_winogrande_${JOB_ID}/"
echo ""

# Print comparison table
echo "============================================================"
echo "                 BASELINE vs READING POLICY COMPARISON"
echo "============================================================"
echo ""
printf "%-15s | %-12s | %-12s | %-10s | %-12s\n" "Dataset" "Baseline" "With Policy" "Change" "Reduction"
echo "--------------- | ------------ | ------------ | ---------- | ------------"

for TASK in gsm8k arc_challenge winogrande; do
    LOG_FILE="checkpoints/rl_policy/minval_${TASK}_${JOB_ID}/training_log.json"
    if [ -f "$LOG_FILE" ]; then
        BASELINE=$(python -c "import json; d=json.load(open('$LOG_FILE')); print(f\"{d['baseline_results']['accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
        FINAL=$(python -c "import json; d=json.load(open('$LOG_FILE')); print(f\"{d['comparison']['final_accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
        CHANGE=$(python -c "import json; d=json.load(open('$LOG_FILE')); print(f\"{d['comparison']['accuracy_change']:+.2%}\")" 2>/dev/null || echo "N/A")
        REDUCTION=$(python -c "import json; d=json.load(open('$LOG_FILE')); print(f\"{d['comparison']['latent_reduction_pct']:.1f}%\")" 2>/dev/null || echo "N/A")
        printf "%-15s | %-12s | %-12s | %-10s | %-12s\n" "$TASK" "$BASELINE" "$FINAL" "$CHANGE" "$REDUCTION"
    else
        printf "%-15s | %-12s | %-12s | %-10s | %-12s\n" "$TASK" "N/A" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "============================================================"
echo "Key: Baseline = No reading policy (100% latent blocks)"
echo "     With Policy = Trained reading policy (selective blocks)"
echo "     Change = Accuracy difference (positive = improvement)"
echo "     Reduction = Latent communication saved"
echo "============================================================"
echo ""
echo "Detailed results: cat checkpoints/rl_policy/minval_*_${JOB_ID}/training_log.json | jq '.comparison'"
