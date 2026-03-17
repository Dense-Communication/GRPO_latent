#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=single_val
#SBATCH --output=logs/single_val_%j.out
#SBATCH --error=logs/single_val_%j.err

# ============================================================
# TIME BUDGET CALCULATION:
#
# Per sample time (estimated):
#   - 1.5B: ~1s/sample
#   - 8B:   ~3s/sample
#   - 14B:  ~6s/sample
#
# Per task (1500 samples * 3 epochs + 300 eval * 2):
#   - 1.5B: ~1.4h per task, ~4.2h for all 3 tasks
#   - 8B:   ~4.3h per task, ~12.9h for all 3 tasks
#   - 14B:  ~8.5h per task, ~25.5h for all 3 tasks
#
# Current setting: 12:00:00
#   - Sufficient for: 1.5B (all), 8B (all), 14B (1 task)
#   - For 14B all tasks, use --time=36:00:00
# ============================================================

# ============================================================
# Single Model Validation Script
#
# Usage:
#   sbatch run_single_validation.sh [model_size] [dataset]
#
# Arguments:
#   model_size: 1.5b, 8b, or 14b (default: 8b)
#   dataset: gsm8k, arc_challenge, winogrande, or all (default: all)
#
# Examples:
#   sbatch run_single_validation.sh 8b gsm8k
#   sbatch run_single_validation.sh 14b all
#   sbatch run_single_validation.sh 1.5b arc_challenge
# ============================================================

# Parse arguments
MODEL_SIZE="${1:-8b}"
DATASET_ARG="${2:-all}"

echo "============================================================"
echo "Single Model Validation"
echo "Job ID: $SLURM_JOB_ID"
echo "Model Size: $MODEL_SIZE"
echo "Dataset: $DATASET_ARG"
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
# Model configuration
# ============================================================
case "$MODEL_SIZE" in
    "1.5b"|"1.5B")
        MODEL_PATH="/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct"
        DEVICE="cuda:0"
        DEVICE2="cuda:1"
        TP_SIZE=1
        GPU_UTIL=0.85
        ;;
    "8b"|"8B")
        MODEL_PATH="/p/scratch/westai0052/liu52/models/Qwen3-8B"
        DEVICE="cuda:0"
        DEVICE2="cuda:2"
        TP_SIZE=2
        GPU_UTIL=0.85
        ;;
    "14b"|"14B")
        MODEL_PATH="/p/scratch/westai0052/liu52/models/Qwen3-14B"
        DEVICE="cuda:0"
        DEVICE2="cuda:2"
        TP_SIZE=4
        GPU_UTIL=0.90
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Valid options: 1.5b, 8b, 14b"
        exit 1
        ;;
esac

MODEL_NAME=$(basename "$MODEL_PATH")

# ============================================================
# Dataset configuration
# ============================================================
declare -A DATASET_CONFIGS
DATASET_CONFIGS["gsm8k"]="1500:300"
DATASET_CONFIGS["arc_challenge"]="1119:200"
DATASET_CONFIGS["winogrande"]="1500:300"

if [ "$DATASET_ARG" = "all" ]; then
    DATASETS=("gsm8k" "arc_challenge" "winogrande")
else
    if [ -z "${DATASET_CONFIGS[$DATASET_ARG]}" ]; then
        echo "Unknown dataset: $DATASET_ARG"
        echo "Valid options: gsm8k, arc_challenge, winogrande, all"
        exit 1
    fi
    DATASETS=("$DATASET_ARG")
fi

# ============================================================
# Training parameters
# ============================================================
NUM_EPOCHS=3
LATENT_STEPS=5
TOP_K_BLOCKS=4
POLICY_LR="1e-4"

# ============================================================
# Run experiments
# ============================================================

RESULTS_DIR="checkpoints/rl_policy/single_${MODEL_SIZE}_${JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME ($MODEL_PATH)"
echo "  Device: $DEVICE, Device2: $DEVICE2"
echo "  TP Size: $TP_SIZE, GPU Util: $GPU_UTIL"
echo "  Datasets: ${DATASETS[*]}"
echo "  Results: $RESULTS_DIR"
echo ""

total_experiments=${#DATASETS[@]}
current_exp=0

for TASK in "${DATASETS[@]}"; do
    IFS=':' read -r MAX_SAMPLES EVAL_SAMPLES <<< "${DATASET_CONFIGS[$TASK]}"
    current_exp=$((current_exp + 1))

    echo ""
    echo "============================================================"
    echo "[$current_exp/$total_experiments] $MODEL_NAME + $TASK"
    echo "  Max samples: $MAX_SAMPLES, Eval samples: $EVAL_SAMPLES"
    echo "============================================================"

    OUTPUT_DIR="$RESULTS_DIR/${TASK}"
    mkdir -p "$OUTPUT_DIR"

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
        --think \
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

    echo "Completed: $OUTPUT_DIR"
done

# ============================================================
# Print comparison table
# ============================================================

echo ""
echo "============================================================"
echo "           VALIDATION COMPLETE: $MODEL_NAME"
echo "============================================================"
echo ""
printf "%-15s | %-10s | %-10s | %-10s | %-10s\n" "Dataset" "Baseline" "WithPolicy" "Change" "Reduction"
echo "--------------- | ---------- | ---------- | ---------- | ----------"

for TASK in "${DATASETS[@]}"; do
    LOG_FILE="$RESULTS_DIR/${TASK}/training_log.json"

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

    printf "%-15s | %-10s | %-10s | %-10s | %-10s\n" "$TASK" "$BASELINE" "$FINAL" "$CHANGE" "$REDUCTION"
done

echo ""
echo "============================================================"
echo "Results: $RESULTS_DIR"
echo "View details: cat $RESULTS_DIR/*/training_log.json | jq '.comparison'"
echo "End Time: $(date)"
echo "============================================================"
