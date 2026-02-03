#!/bin/bash
#SBATCH --job-name=gsm8k_long
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/p/scratch/westai0052/liu52/verl-agent/outputs/slurm_%j.out
#SBATCH --error=/p/scratch/westai0052/liu52/verl-agent/outputs/slurm_%j.err

# GSM8K GRPO 长时间训练 - SLURM 批处理
# 用于观察训练过程中 reward 和 accuracy 的变化趋势
#
# 提交: sbatch submit_long_train.slurm
# 查看: squeue -u $USER
# 取消: scancel <job_id>
#
# 可选环境变量:
#   TOTAL_STEPS=500       总训练步数
#   TEST_FREQ=5           验证频率
#   SAVE_FREQ=100         保存频率
#   USE_VAL_SUBSET=true   是否使用验证子集

echo "========================================="
echo "SLURM Job Info"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

cd /p/scratch/westai0052/liu52/verl-agent/test_script

# 默认配置 (可通过环境变量覆盖)
export TOTAL_STEPS=${TOTAL_STEPS:-500}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-100}
export USE_VAL_SUBSET=${USE_VAL_SUBSET:-true}

echo "训练配置:"
echo "  总步数: $TOTAL_STEPS"
echo "  验证频率: $TEST_FREQ"
echo "  保存频率: $SAVE_FREQ"
echo "  使用验证子集: $USE_VAL_SUBSET"

bash gsm8k_long_train.sh

echo "========================================="
echo "Job finished: $(date)"
echo "========================================="
