#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=full_val
#SBATCH --output=logs/full_val_%j.out
#SBATCH --error=logs/full_val_%j.err

# ============================================================
# TIME BUDGET CALCULATION:
#
# Per sample time (estimated):
#   - 1.5B: ~1s/sample
#   - 8B:   ~3s/sample
#   - 14B:  ~6s/sample
#
# Per task (1500 samples * 3 epochs + 300 eval * 2 [baseline+final]):
#   - 1.5B: (1500*3 + 300*2) * 1s = 5100s = 1.4h
#   - 8B:   (1500*3 + 300*2) * 3s = 15300s = 4.3h
#   - 14B:  (1500*3 + 300*2) * 6s = 30600s = 8.5h
#
# With 2 models (1.5B, 8B) x 3 datasets:
#   Total = (1.4 + 4.3) * 3 = 17.1h
#   With 30% safety margin = 22h
#
# Current setting: 24:00:00 (sufficient for 1.5B + 8B)
# ============================================================

# ============================================================
# Full Validation Script - Multiple Models & Complete Datasets
#
# Purpose: Comprehensive validation of Reading Policy across:
#   - Multiple model sizes (1.5B, 8B, 14B)
#   - Complete datasets (not subsampled)
#   - Baseline vs With Policy comparison
#
# Datasets:
#   - GSM8K: 7473 train, use 1000 train + 500 eval
#   - ARC-Challenge: 1119 train, use all
#   - Winogrande: 9248 train, use 1000 train + 500 eval
#
# Models:
#   - Qwen2.5-1.5B-Instruct (small, fast)
#   - Qwen3-8B (medium)
#   - Qwen3-14B (large, best quality)
# ============================================================

echo "============================================================"
echo "Full Validation - Multiple Models & Complete Datasets"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "============================================================"

source ~/.bashrc
conda activate latentmas

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs
mkdir -p checkpoints/rl_policy

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

# ============================================================
# Configuration
# ============================================================

# Models to test (comment out to skip)
MODELS=(
    "/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct"
    "/p/scratch/westai0052/liu52/models/Qwen3-8B"
    # "/p/scratch/westai0052/liu52/models/Qwen3-14B"  # Uncomment for 14B (needs more GPU memory)
)

# Datasets with their sample sizes
# Format: "task_name:max_samples:eval_samples"
DATASETS=(
    "gsm8k:1500:300"
    "arc_challenge:1119:200"  # Use all available
    "winogrande:1500:300"
)

# Common training parameters
NUM_EPOCHS=3
LATENT_STEPS=5
TOP_K_BLOCKS=4
POLICY_LR="1e-4"

# ============================================================
# Helper function to get model short name
# ============================================================
get_model_name() {
    basename "$1"
}

# ============================================================
# Helper function to get GPU config based on model size
# ============================================================
get_gpu_config() {
    local model_path="$1"
    local model_name=$(basename "$model_path")

    case "$model_name" in
        *"1.5B"*)
            echo "cuda:0|cuda:1|1|0.85"  # device|device2|tp_size|gpu_util
            ;;
        *"8B"*)
            echo "cuda:0|cuda:2|2|0.85"
            ;;
        *"14B"*)
            echo "cuda:0|cuda:2|4|0.90"
            ;;
        *)
            echo "cuda:0|cuda:1|1|0.85"
            ;;
    esac
}

# ============================================================
# Run experiments
# ============================================================

RESULTS_DIR="checkpoints/rl_policy/full_val_${JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Full Validation Summary - $JOB_ID" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

total_experiments=$((${#MODELS[@]} * ${#DATASETS[@]}))
current_exp=0

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(get_model_name "$MODEL_PATH")
    GPU_CONFIG=$(get_gpu_config "$MODEL_PATH")

    IFS='|' read -r DEVICE DEVICE2 TP_SIZE GPU_UTIL <<< "$GPU_CONFIG"

    echo ""
    echo "============================================================"
    echo "Model: $MODEL_NAME"
    echo "  Device: $DEVICE, Device2: $DEVICE2"
    echo "  TP Size: $TP_SIZE, GPU Util: $GPU_UTIL"
    echo "============================================================"

    for DATASET_CONFIG in "${DATASETS[@]}"; do
        IFS=':' read -r TASK MAX_SAMPLES EVAL_SAMPLES <<< "$DATASET_CONFIG"

        current_exp=$((current_exp + 1))

        echo ""
        echo "------------------------------------------------------------"
        echo "[$current_exp/$total_experiments] $MODEL_NAME + $TASK"
        echo "  Max samples: $MAX_SAMPLES, Eval samples: $EVAL_SAMPLES"
        echo "------------------------------------------------------------"

        OUTPUT_DIR="$RESULTS_DIR/${MODEL_NAME}_${TASK}"
        mkdir -p "$OUTPUT_DIR"

        # Run training with baseline comparison
        python run_rl_train.py \
            --model_name "$MODEL_PATH" \
            --task "$TASK" \
            --prompt sequential \
            --method latent_mas \
            --device "$DEVICE" \
            --device2 "$DEVICE2" \
            --use_vllm \
            --tensor_parallel_size "$TP_SIZE" \
            --gpu_memory_utilization "$GPU_UTIL" \
            --max_new_tokens 1024 \
            --latent_steps "$LATENT_STEPS" \
            --temperature 0.6 \
            --top_p 0.95 \
            --top_k_blocks "$TOP_K_BLOCKS" \
            --policy_num_heads 8 \
            --policy_num_layers 2 \
            --policy_lr "$POLICY_LR" \
            --grpo_group_size 8 \
            --num_epochs "$NUM_EPOCHS" \
            --max_samples "$MAX_SAMPLES" \
            --eval_samples "$EVAL_SAMPLES" \
            --seed 42 \
            --output_dir "$OUTPUT_DIR" \
            2>&1 | tee "$OUTPUT_DIR/console_output.log"

        # Extract and log results
        if [ -f "$OUTPUT_DIR/training_log.json" ]; then
            echo "[$current_exp/$total_experiments] $MODEL_NAME + $TASK - DONE" >> "$SUMMARY_FILE"
            python -c "
import json
with open('$OUTPUT_DIR/training_log.json') as f:
    d = json.load(f)
comp = d.get('comparison', {})
baseline = comp.get('baseline_accuracy', 0)
final = comp.get('final_accuracy', 0)
change = comp.get('accuracy_change', 0)
reduction = comp.get('latent_reduction_pct', 0)
print(f'  Baseline: {baseline:.2%}, Final: {final:.2%}, Change: {change:+.2%}, Reduction: {reduction:.1f}%')
" >> "$SUMMARY_FILE"
        else
            echo "[$current_exp/$total_experiments] $MODEL_NAME + $TASK - FAILED" >> "$SUMMARY_FILE"
        fi

        echo "Completed: $OUTPUT_DIR"
    done
done

# ============================================================
# Generate final comparison table
# ============================================================

echo ""
echo "============================================================"
echo "                    FULL VALIDATION COMPLETE"
echo "============================================================"
echo ""

# Print comparison table
echo "=============================================================================================="
echo "                           BASELINE vs READING POLICY COMPARISON"
echo "=============================================================================================="
echo ""
printf "%-25s | %-15s | %-10s | %-10s | %-10s | %-10s\n" "Model" "Dataset" "Baseline" "WithPolicy" "Change" "Reduction"
echo "------------------------- | --------------- | ---------- | ---------- | ---------- | ----------"

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(get_model_name "$MODEL_PATH")

    for DATASET_CONFIG in "${DATASETS[@]}"; do
        IFS=':' read -r TASK _ _ <<< "$DATASET_CONFIG"

        LOG_FILE="$RESULTS_DIR/${MODEL_NAME}_${TASK}/training_log.json"

        if [ -f "$LOG_FILE" ]; then
            BASELINE=$(python -c "import json; d=json.load(open('$LOG_FILE')); print(f\"{d['comparison']['baseline_accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
            FINAL=$(python -c "import json; d=json.load(open('$LOG_FILE')); print(f\"{d['comparison']['final_accuracy']:.2%}\")" 2>/dev/null || echo "N/A")
            CHANGE=$(python -c "import json; d=json.load(open('$LOG_FILE')); print(f\"{d['comparison']['accuracy_change']:+.2%}\")" 2>/dev/null || echo "N/A")
            REDUCTION=$(python -c "import json; d=json.load(open('$LOG_FILE')); print(f\"{d['comparison']['latent_reduction_pct']:.1f}%\")" 2>/dev/null || echo "N/A")
        else
            BASELINE="N/A"
            FINAL="N/A"
            CHANGE="N/A"
            REDUCTION="N/A"
        fi

        printf "%-25s | %-15s | %-10s | %-10s | %-10s | %-10s\n" "$MODEL_NAME" "$TASK" "$BASELINE" "$FINAL" "$CHANGE" "$REDUCTION"
    done
done

echo ""
echo "=============================================================================================="
echo ""
echo "Results directory: $RESULTS_DIR"
echo "Summary file: $SUMMARY_FILE"
echo ""
echo "To view detailed results:"
echo "  cat $RESULTS_DIR/*/training_log.json | jq '.comparison'"
echo ""
echo "End Time: $(date)"
echo "============================================================"

# Append final table to summary
echo "" >> "$SUMMARY_FILE"
echo "Completed: $(date)" >> "$SUMMARY_FILE"
