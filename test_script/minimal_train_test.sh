#!/bin/bash
# 最小训练测试 - 只跑2个step验证整个流程
# 预计时间: 2-3分钟
#
# 关键修复: reward_model.reward_manager=naive (使用规则奖励而不是环境交互)

set -x

source /p/home/jusers/liu52/jureca/miniforge3/etc/profile.d/conda.sh
conda activate verl-agent

# ========== JURECA 离线环境配置 ==========
SCRATCH_BASE="/p/scratch/westai0052/liu52"

# 离线模式 (计算节点无网络)
export HF_HOME="$SCRATCH_BASE/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export VLLM_NO_USAGE_STATS=1

# Ray 临时目录放在 scratch (避免 /tmp 问题)
export RAY_TMPDIR="$SCRATCH_BASE/.tmp/ray"
mkdir -p "$RAY_TMPDIR"

# 清理之前的 Ray session
rm -rf /tmp/ray/session_* 2>/dev/null
rm -rf "$RAY_TMPDIR"/session_* 2>/dev/null

# 路径
LOCAL_QWEN="$SCRATCH_BASE/models/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="$SCRATCH_BASE/verl-agent/outputs/minimal_test_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

echo "========================================="
echo "最小训练测试 - 验证GRPO奖励循环"
echo "========================================="
echo "输出: $OUTPUT_DIR"
echo "关键: reward_model.reward_manager=naive"
echo "RAY_TMPDIR: $RAY_TMPDIR"
echo "========================================="

# 超小配置: batch=8, 只跑2个step
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=naive \
    data.train_files="/p/scratch/westai0052/liu52/verl-agent/test_script/data/train.parquet" \
    data.val_files="/p/scratch/westai0052/liu52/verl-agent/test_script/data/test.parquet" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$LOCAL_QWEN" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_model_len=640 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=True \
    env.env_name="" \
    trainer.logger=[console] \
    trainer.project_name='minimal_test' \
    trainer.experiment_name='minimal_test' \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_training_steps=2 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

echo ""
echo "========================================="
echo "分析结果"
echo "========================================="

# 检查分数
echo "--- 分数统计 ---"
echo "test_score 值:"
grep -oE "test_score:[0-9.]+" "${OUTPUT_DIR}/training.log" | sort | uniq -c
echo ""
echo "critic/score/mean 值:"
grep -oE "critic/score/mean:[0-9.]+" "${OUTPUT_DIR}/training.log" | sort | uniq -c

# 检查是否有非零分数
echo ""
echo "--- 奖励验证 ---"
NON_ZERO_SCORE=$(grep -oE "critic/score/mean:0\.[0-9]*[1-9]|critic/score/mean:[1-9]" "${OUTPUT_DIR}/training.log" | head -1)
NON_ZERO_TEST=$(grep -oE "test_score:0\.[0-9]*[1-9]|test_score:[1-9]" "${OUTPUT_DIR}/training.log" | head -1)

if [ -n "$NON_ZERO_SCORE" ] || [ -n "$NON_ZERO_TEST" ]; then
    echo "✅ 发现非零奖励!"
    [ -n "$NON_ZERO_SCORE" ] && echo "   critic/score: $NON_ZERO_SCORE"
    [ -n "$NON_ZERO_TEST" ] && echo "   test_score: $NON_ZERO_TEST"
else
    echo "⚠️ 所有奖励都是0，检查reward配置"
fi

# 检查 ground_truth 和 score 打印（naive reward manager 会打印这些）
echo ""
echo "--- Reward Manager 验证 ---"
if grep -q "\[ground_truth\]" "${OUTPUT_DIR}/training.log"; then
    echo "✅ 使用 naive reward manager (规则奖励)"
    echo "示例输出:"
    grep -A1 "\[response\]" "${OUTPUT_DIR}/training.log" | head -6
elif grep -q "\[openai/gsm8k\]\[score\]" "${OUTPUT_DIR}/training.log"; then
    echo "⚠️ 使用 episode reward manager (环境交互) - 这可能是问题!"
else
    echo "? 无法确定 reward manager 类型"
fi

echo ""
echo "========================================="
echo "测试完成！"
echo "输出目录: $OUTPUT_DIR"
echo "日志: ${OUTPUT_DIR}/training.log"
echo "========================================="
