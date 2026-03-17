#!/bin/bash
#SBATCH --job-name=ablation_k
#SBATCH --output=logs/ablation_topk_%j.out
#SBATCH --error=logs/ablation_topk_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --account=westai0052

# ============================================================
# 消融实验: Top-K 值对效果的影响
#
# 目的: 分析不同 k 值 (1, 2, 4, 8) 对准确率和效率的影响
# 数据集: GPQA (样本量适中，测试速度快)
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Task: Ablation Study - Top-K Values"
echo "Start Time: $(date)"
echo "=========================================="

source ~/.bashrc
conda activate latentmas

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

MODEL_NAME="/p/scratch/westai0052/liu52/models/Qwen3-14B"
OUTPUT_DIR="checkpoints/ablation/topk_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# 测试不同的 top_k 值
for K in 1 2 4 8; do
    echo "=========================================="
    echo "Testing top_k = $K"
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
        --top_k_blocks $K \
        --policy_num_heads 8 \
        --policy_num_layers 2 \
        --policy_dropout 0.1 \
        --similarity_threshold 0.85 \
        --policy_lr 1e-5 \
        --grpo_group_size 4 \
        --num_epochs 2 \
        --max_samples 150 \
        --eval_samples 48 \
        --seed 42 \
        --output_dir $OUTPUT_DIR/topk_$K \
        --log_file $OUTPUT_DIR/topk_${K}_log.json \
        2>&1 | tee $OUTPUT_DIR/topk_${K}_output.log

    echo "top_k=$K complete!"
    echo ""
done

echo "=========================================="
echo "Ablation Study Complete!"
echo "End Time: $(date)"
echo ""
echo "Results Summary:"
for K in 1 2 4 8; do
    echo "  top_k=$K: $OUTPUT_DIR/topk_${K}_log.json"
done
echo "=========================================="
