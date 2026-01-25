#!/bin/bash
# 最小训练测试 - 快速验证训练管道是否能正常学习
# 只训练1个epoch，小batch size，预计10-15分钟完成

set -x

# 激活 conda 环境
source /p/home/jusers/liu52/jureca/miniforge3/etc/profile.d/conda.sh
conda activate verl-agent

ENGINE=${1:-vllm}

# ====== 离线模式设置 ======
export HF_HOME="/p/scratch/westai0052/liu52/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0
export WANDB_DISABLED=true
export WANDB_MODE=offline

# ====== 路径配置 ======
LOCAL_QWEN="/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="/p/scratch/westai0052/liu52/verl-agent/outputs/mini_test_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
CUSTOM_REWARD_PATH="/p/scratch/westai0052/liu52/verl-agent/test_script/custom_gsm8k_reward.py"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "========================================="
echo "最小训练测试 - 验证学习能力"
echo "========================================="
echo "目标: 验证模型能否从新的奖励函数中学习"
echo "配置: 1 epoch, 小 batch, 启用 KL loss"
echo "预计时间: 10-15 分钟"
echo "输出目录: $OUTPUT_DIR"
echo "========================================="

# 使用更小的配置进行快速测试
# train_batch_size=32 (比正常的64小)
# total_epochs=1 (只训练1轮)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/p/scratch/westai0052/liu52/verl-agent/test_script/data/train.parquet" \
    data.val_files="/p/scratch/westai0052/liu52/verl-agent/test_script/data/test.parquet" \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$LOCAL_QWEN" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=True \
    trainer.logger=[console] \
    trainer.project_name='mini_test' \
    trainer.experiment_name='mini_test_grpo' \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    env.env_name="" \
    env.resources_per_worker.num_gpus=0 \
    env.rollout.n=1 \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name="compute_score" \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/training.log"

# ========================================
# 分析结果
# ========================================
echo ""
echo "========================================="
echo "分析训练结果..."
echo "========================================="

LOG_FILE="${OUTPUT_DIR}/training.log"

# 提取关键指标
echo ""
echo "--- 奖励分数变化 ---"
grep -E "score.*:|reward" "$LOG_FILE" | head -20

echo ""
echo "--- 训练前后对比 ---"
BEFORE=$(grep -E "step.*0.*test_score|val.*step.*0" "$LOG_FILE" | head -1)
AFTER=$(grep -E "step.*test_score" "$LOG_FILE" | tail -1)
echo "训练前: $BEFORE"
echo "训练后: $AFTER"

# 检查是否有非零分数
echo ""
echo "--- 检查学习信号 ---"
NON_ZERO_SCORES=$(grep -oE "score['\"]?:?\s*[0-9]+\.[0-9]+" "$LOG_FILE" | grep -v "0\.0" | head -10)
if [ -n "$NON_ZERO_SCORES" ]; then
    echo "✓ 发现非零分数，说明奖励函数正在工作！"
    echo "$NON_ZERO_SCORES"
else
    echo "⚠ 没有发现非零分数，需要检查配置"
fi

# 检查模型输出是否正常
echo ""
echo "--- 检查模型输出质量 ---"
grep -A2 "\[response\]" "$LOG_FILE" | head -20

echo ""
echo "========================================="
echo "测试完成！"
echo "完整日志: ${LOG_FILE}"
echo "========================================="
