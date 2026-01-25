#!/bin/bash
#SBATCH --job-name=gsm8k_grpo
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/westai0052/liu52/verl-agent/outputs/slurm_%j.out
#SBATCH --error=/p/scratch/westai0052/liu52/verl-agent/outputs/slurm_%j.err

# GSM8K GRPO 完整训练 - SLURM 批处理
# 提交: sbatch submit_gsm8k_full.slurm
# 查看: squeue -u $USER
# 取消: scancel <job_id>

echo "========================================="
echo "SLURM Job Info"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

cd /p/scratch/westai0052/liu52/verl-agent/test_script
bash gsm8k_full_train.sh

echo "========================================="
echo "Job finished: $(date)"
echo "========================================="
