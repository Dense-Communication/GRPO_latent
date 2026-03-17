#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=8b_wino
#SBATCH --output=logs/8b_wino_%j.out
#SBATCH --error=logs/8b_wino_%j.err

# Qwen3-8B + Winogrande
# Time: ~4.5h actual, 8h budget

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

MODEL="/p/scratch/westai0052/liu52/models/Qwen3-8B"
TASK="winogrande"
OUTPUT_DIR="checkpoints/rl_policy/val_8b_${TASK}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "Starting: 8B + $TASK"
echo "Job ID: $SLURM_JOB_ID"
echo "Output: $OUTPUT_DIR"

python run_rl_train.py \
    --model_name $MODEL \
    --task $TASK \
    --prompt sequential \
    --method latent_mas \
    --device cuda:0 \
    --device2 cuda:2 \
    --use_vllm \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.85 \
    --max_new_tokens 1024 \
    --latent_steps 5 \
    --temperature 0.6 \
    --top_p 0.95 \
    --think \
    --top_k_blocks 4 \
    --policy_num_heads 8 \
    --policy_num_layers 2 \
    --policy_lr 1e-4 \
    --grpo_group_size 8 \
    --num_epochs 3 \
    --max_samples 1500 \
    --eval_samples 300 \
    --seed 42 \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/console_output.log

echo "Done: $OUTPUT_DIR"
