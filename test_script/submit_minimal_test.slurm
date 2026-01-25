#!/bin/bash
#SBATCH --job-name=grpo_minimal
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/westai0052/liu52/verl-agent/outputs/slurm_%j.out
#SBATCH --error=/p/scratch/westai0052/liu52/verl-agent/outputs/slurm_%j.err

# GRPO 最小训练测试 - SLURM 批处理版本
# 提交方式: sbatch submit_minimal_test.slurm
# 查看状态: squeue -u $USER
# 取消作业: scancel <job_id>

echo "========================================="
echo "SLURM Job Info"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# 切换到脚本目录
cd /p/scratch/westai0052/liu52/verl-agent/test_script

# 运行训练
bash minimal_train_test.sh

echo "========================================="
echo "Job finished: $(date)"
echo "========================================="
