#!/bin/bash
#SBATCH --job-name=ablation_rand
#SBATCH --output=logs/ablation_random_%j.out
#SBATCH --error=logs/ablation_random_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=westai0052

# ============================================================
# 消融实验: Random Selection vs Learned Policy
#
# 目的: 证明学习的 policy 比随机选择更好
# 方法: 使用未训练的随机初始化 policy 进行评估
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Task: Ablation Study - Random vs Learned Policy"
echo "Start Time: $(date)"
echo "=========================================="

source ~/.bashrc
conda activate latentmas

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen3-14B"
OUTPUT_DIR="checkpoints/ablation/random_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# ============================================================
# Test 1: Random Policy (只运行 1 epoch，不真正训练)
# ============================================================
echo "=========================================="
echo "Test 1: Random Policy (Untrained)"
echo "=========================================="

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
    --policy_lr 0.0 \
    --num_epochs 1 \
    --max_samples 50 \
    --eval_samples 98 \
    --seed 42 \
    --output_dir $OUTPUT_DIR/random_policy \
    --log_file $OUTPUT_DIR/random_policy_log.json \
    2>&1 | tee $OUTPUT_DIR/random_policy_output.log

echo ""

# ============================================================
# Test 2: Trained Policy (使用之前训练好的 checkpoint)
# ============================================================
echo "=========================================="
echo "Test 2: Trained Policy"
echo "=========================================="

# 查找最新的训练好的 policy
TRAINED_POLICY=$(find /p/scratch/westai0052/liu52/LatentMAS/checkpoints/rl_policy -name "policy_best.pt" -type f 2>/dev/null | head -1)

if [ -n "$TRAINED_POLICY" ]; then
    echo "Using trained policy: $TRAINED_POLICY"

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
        --policy_checkpoint $TRAINED_POLICY \
        --num_epochs 1 \
        --max_samples 50 \
        --eval_samples 98 \
        --seed 42 \
        --output_dir $OUTPUT_DIR/trained_policy \
        --log_file $OUTPUT_DIR/trained_policy_log.json \
        2>&1 | tee $OUTPUT_DIR/trained_policy_output.log
else
    echo "No trained policy found! Please run training first."
fi

echo ""
echo "=========================================="
echo "Ablation Study Complete!"
echo "End Time: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
