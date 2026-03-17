#!/bin/bash
#SBATCH --job-name=model_cmp
#SBATCH --output=logs/model_comparison_%j.out
#SBATCH --error=logs/model_comparison_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=westai0052

# ============================================================
# 多模型对比实验
#
# 目的: 证明 Reading Policy 在不同规模模型上的通用性
# 模型: Qwen3-8B, Qwen3-14B
# 数据集: GPQA (样本适中，测试快)
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Task: Multi-Model Comparison"
echo "Start Time: $(date)"
echo "=========================================="

source ~/.bashrc
conda activate latentmas

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/model_comparison

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

OUTPUT_DIR="checkpoints/model_comparison/comparison_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# ============================================================
# Model 1: Qwen3-8B
# ============================================================
echo "=========================================="
echo "Model 1: Qwen3-8B"
echo "=========================================="

MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen3-8B"

python run_rl_train.py \
    --model_name $MODEL_NAME \
    --task gpqa \
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
    --policy_lr 1e-5 \
    --grpo_group_size 4 \
    --num_epochs 3 \
    --max_samples 150 \
    --eval_samples 48 \
    --seed 42 \
    --output_dir $OUTPUT_DIR/qwen3_8b \
    --log_file $OUTPUT_DIR/qwen3_8b_log.json \
    2>&1 | tee $OUTPUT_DIR/qwen3_8b_output.log

echo "Qwen3-8B complete!"
echo ""

# ============================================================
# Model 2: Qwen3-14B
# ============================================================
echo "=========================================="
echo "Model 2: Qwen3-14B"
echo "=========================================="

MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen3-14B"

python run_rl_train.py \
    --model_name $MODEL_NAME \
    --task gpqa \
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
    --policy_lr 1e-5 \
    --grpo_group_size 4 \
    --num_epochs 3 \
    --max_samples 150 \
    --eval_samples 48 \
    --seed 42 \
    --output_dir $OUTPUT_DIR/qwen3_14b \
    --log_file $OUTPUT_DIR/qwen3_14b_log.json \
    2>&1 | tee $OUTPUT_DIR/qwen3_14b_output.log

echo "Qwen3-14B complete!"
echo ""

echo "=========================================="
echo "Model Comparison Complete!"
echo "End Time: $(date)"
echo ""
echo "Results:"
echo "  Qwen3-8B:  $OUTPUT_DIR/qwen3_8b_log.json"
echo "  Qwen3-14B: $OUTPUT_DIR/qwen3_14b_log.json"
echo "=========================================="
