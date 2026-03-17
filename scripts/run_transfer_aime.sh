#!/bin/bash
#SBATCH --job-name=transfer_aime
#SBATCH --output=logs/transfer_aime_%j.out
#SBATCH --error=logs/transfer_aime_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=westai0052

# ============================================================
# Transfer Learning: GSM8K -> AIME 2024
#
# 目标:
#   1. 保持 AIME 2024 准确率 (~66.7%)
#   2. 减少 latent 通信量 (通过 reading policy 选择性阅读)
#
# 实验设计:
#   Step 1: 在 GSM8K 上训练 reading policy
#   Step 2: 在 AIME 2024 上评估 (对比有无 policy)
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Task: Transfer Learning GSM8K -> AIME 2024"
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
OUTPUT_DIR="checkpoints/rl_policy/transfer_aime_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Training Task: GSM8K (7473 samples)"
echo "  Evaluation Task: AIME 2024 (30 samples)"
echo "  Goal: Maintain accuracy, reduce latent communication"
echo "  Output: $OUTPUT_DIR"
echo ""

# ============================================================
# Step 0: Run Baseline (NO reading policy) on AIME 2024
# ============================================================
echo "=========================================="
echo "Step 0: Baseline Evaluation on AIME 2024 (NO reading policy)"
echo "=========================================="
echo ""

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
    2>&1 | tee $OUTPUT_DIR/baseline_aime_output.log

echo ""
echo "Baseline AIME 2024 complete! Check baseline_aime_output.log"
echo ""

# ============================================================
# Step 1: Train Reading Policy on GSM8K
# ============================================================
echo "=========================================="
echo "Step 1: Training Reading Policy on GSM8K"
echo "=========================================="
echo ""

python run_rl_train.py \
    --model_name $MODEL_NAME \
    --task gsm8k \
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
    --grpo_group_size 8 \
    --grpo_clip_epsilon 0.2 \
    --grpo_kl_coef 0.1 \
    --entropy_coef 0.01 \
    --ref_policy_update_freq 100 \
    --update_every_n_groups 4 \
    --reward_alpha 1.0 \
    --reward_beta 0.3 \
    --reward_gamma 0.2 \
    --num_epochs 3 \
    --max_samples 2000 \
    --eval_samples 200 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --output_dir $OUTPUT_DIR \
    --log_file $OUTPUT_DIR/gsm8k_training_log.json \
    2>&1 | tee $OUTPUT_DIR/gsm8k_training.log

echo ""
echo "GSM8K training complete!"
echo ""

# ============================================================
# Step 2: Evaluate on AIME 2024 with trained policy
# ============================================================
echo "=========================================="
echo "Step 2: Evaluating on AIME 2024"
echo "=========================================="
echo ""

# Find the best policy checkpoint
POLICY_CKPT="$OUTPUT_DIR/policy_best.pt"

if [ -f "$POLICY_CKPT" ]; then
    echo "Using trained policy: $POLICY_CKPT"

    # Evaluate with reading policy
    python run_rl_train.py \
        --model_name $MODEL_NAME \
        --task aime2024 \
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
        --policy_checkpoint $POLICY_CKPT \
        --num_epochs 1 \
        --max_samples -1 \
        --eval_samples 30 \
        --seed 42 \
        --output_dir $OUTPUT_DIR/aime_eval \
        --log_file $OUTPUT_DIR/aime_eval_log.json \
        2>&1 | tee $OUTPUT_DIR/aime_eval.log
else
    echo "Warning: Policy checkpoint not found at $POLICY_CKPT"
fi

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results Summary:"
echo "  1. Baseline (no policy): $OUTPUT_DIR/baseline_aime_output.log"
echo "  2. GSM8K Training Log:   $OUTPUT_DIR/gsm8k_training_log.json"
echo "  3. AIME 2024 with Policy: $OUTPUT_DIR/aime_eval_log.json"
echo ""
echo "Key Metrics to Compare:"
echo "  - Accuracy: Should maintain ~66.7%"
echo "  - Latent Reduction: Check 'latent_reduction_pct' in eval_results"
echo "  - Selection Ratio: Check 'avg_selection_ratio' in eval_results"
echo "=========================================="
