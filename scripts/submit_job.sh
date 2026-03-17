#!/bin/bash
# ============================================================
# 提交脚本 - 用于提交 SLURM 作业
# ============================================================
#
# 使用方法:
#   ./scripts/submit_job.sh [full|test]
#
# 示例:
#   ./scripts/submit_job.sh test   # 提交测试作业 (小模型, 少量数据)
#   ./scripts/submit_job.sh full   # 提交完整训练作业
#   ./scripts/submit_job.sh        # 默认提交测试作业
#
# ============================================================

cd /p/scratch/westai0052/liu52/LatentMAS

# 创建必要的目录
mkdir -p logs
mkdir -p checkpoints/rl_policy

# 确保脚本有执行权限
chmod +x scripts/run_rl_train.sh
chmod +x scripts/run_rl_train_small.sh

# 选择要运行的脚本
MODE=${1:-test}

if [ "$MODE" == "full" ]; then
    echo "Submitting FULL training job..."
    JOB_ID=$(sbatch scripts/run_rl_train.sh | awk '{print $4}')
    SCRIPT="run_rl_train.sh"
elif [ "$MODE" == "test" ]; then
    echo "Submitting TEST job (small model, limited data)..."
    JOB_ID=$(sbatch scripts/run_rl_train_small.sh | awk '{print $4}')
    SCRIPT="run_rl_train_small.sh"
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [full|test]"
    exit 1
fi

echo ""
echo "=========================================="
echo "Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo "Script: $SCRIPT"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  查看作业状态:  squeue -j $JOB_ID"
echo "  查看输出日志:  tail -f logs/rl_train_${JOB_ID}.out"
echo "  查看错误日志:  tail -f logs/rl_train_${JOB_ID}.err"
echo "  取消作业:      scancel $JOB_ID"
echo ""
echo "实时监控:"
echo "  watch -n 5 squeue -j $JOB_ID"
echo ""
