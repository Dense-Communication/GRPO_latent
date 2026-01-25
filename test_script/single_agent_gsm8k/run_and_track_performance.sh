#!/bin/bash
# 运行训练并跟踪性能提升的脚本

echo "========================================="
echo "开始 GSM8K 训练任务"
echo "========================================="
echo ""

# 设置日志文件
LOG_FILE="gsm8k_training_$(date +%Y%m%d_%H%M%S).log"
METRICS_FILE="training_metrics_$(date +%Y%m%d_%H%M%S).txt"

echo "日志文件: $LOG_FILE"
echo "指标文件: $METRICS_FILE"
echo ""

# 运行训练
cd /p/scratch/westai0052/liu52/verl-agent/test_script
bash run_gsm8k_with_custom_reward.sh 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================="
echo "训练完成！正在提取性能指标..."
echo "========================================="
echo ""

# 提取性能指标
echo "=== 性能提升报告 ===" > "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

# 提取初始验证分数（训练前）
INITIAL_SCORE=$(grep "Initial validation metrics" "$LOG_FILE" | grep -oP "test_score['\"]:\s*np\.float64\(\K[0-9.]+")
if [ -z "$INITIAL_SCORE" ]; then
    INITIAL_SCORE=$(grep "step:0.*val/.*test_score" "$LOG_FILE" | head -1 | grep -oP "test_score:\K[0-9.]+")
fi

# 提取最终验证分数（训练后）
FINAL_SCORE=$(grep "step:[0-9].*val/.*test_score" "$LOG_FILE" | tail -1 | grep -oP "test_score:\K[0-9.]+")

# 输出结果
echo "训练前初始分数: $INITIAL_SCORE" | tee -a "$METRICS_FILE"
echo "训练后最终分数: $FINAL_SCORE" | tee -a "$METRICS_FILE"

if [ -n "$INITIAL_SCORE" ] && [ -n "$FINAL_SCORE" ]; then
    # 计算提升
    IMPROVEMENT=$(echo "scale=4; $FINAL_SCORE - $INITIAL_SCORE" | bc)
    IMPROVEMENT_PERCENT=$(echo "scale=2; ($FINAL_SCORE - $INITIAL_SCORE) * 100 / ($INITIAL_SCORE + 0.0001)" | bc)

    echo "" | tee -a "$METRICS_FILE"
    echo "性能提升: $IMPROVEMENT (绝对值)" | tee -a "$METRICS_FILE"
    echo "性能提升: $IMPROVEMENT_PERCENT% (百分比)" | tee -a "$METRICS_FILE"

    if (( $(echo "$IMPROVEMENT > 0" | bc -l) )); then
        echo "" | tee -a "$METRICS_FILE"
        echo "✓ 训练有效！模型性能提升了！" | tee -a "$METRICS_FILE"
    else
        echo "" | tee -a "$METRICS_FILE"
        echo "✗ 性能未提升，可能需要调整超参数或增加训练数据" | tee -a "$METRICS_FILE"
    fi
fi

echo "" | tee -a "$METRICS_FILE"
echo "所有训练步骤的验证分数:" >> "$METRICS_FILE"
grep "step:[0-9].*val/.*test_score" "$LOG_FILE" >> "$METRICS_FILE"

echo "" | tee -a "$METRICS_FILE"
echo "模型保存路径:" | tee -a "$METRICS_FILE"
echo "checkpoints/verl_agent_GSM8K_custom_reward/rollout_only_grpo_qwen2p5_1p5b_custom_reward/" | tee -a "$METRICS_FILE"

echo ""
echo "========================================="
echo "完整指标已保存到: $METRICS_FILE"
echo "完整日志已保存到: $LOG_FILE"
echo "========================================="
