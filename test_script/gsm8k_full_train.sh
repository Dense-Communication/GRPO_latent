#!/bin/bash
# GSM8K 完整训练 - GRPO算法
# 预计时间: 约3-4小时 (取决于epoch数)

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

# Ray 临时目录放在 scratch
export RAY_TMPDIR="$SCRATCH_BASE/.tmp/ray"
mkdir -p "$RAY_TMPDIR"

# 清理之前的 Ray session
rm -rf /tmp/ray/session_* 2>/dev/null
rm -rf "$RAY_TMPDIR"/session_* 2>/dev/null

# ========== 训练配置 ==========
# 模型路径
LOCAL_QWEN="$SCRATCH_BASE/models/Qwen2.5-1.5B-Instruct"

# 输出目录
EXPERIMENT_NAME="gsm8k_grpo_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$SCRATCH_BASE/verl-agent/outputs/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# ========== 超参数 ==========
# Batch size
TRAIN_BATCH_SIZE=64        # 每步训练的样本数
VAL_BATCH_SIZE=64          # 验证batch大小
PPO_MINI_BATCH_SIZE=64     # PPO mini-batch
PPO_MICRO_BATCH_SIZE=4     # 每GPU micro-batch

# 序列长度
MAX_PROMPT_LENGTH=512      # 最大prompt长度
MAX_RESPONSE_LENGTH=512    # 最大response长度 (GSM8K需要较长输出)

# 训练步数
TOTAL_STEPS=200            # 总训练步数 (7473样本 / 64 batch ≈ 117步/epoch)
SAVE_FREQ=50               # 保存checkpoint频率
TEST_FREQ=10               # 验证频率

# 学习率
LR=1e-6                    # 学习率 (GRPO通常用较小的lr)

# KL惩罚
KL_COEF=0.05               # KL损失系数

echo "========================================="
echo "GSM8K GRPO 完整训练"
echo "========================================="
echo "实验名称: $EXPERIMENT_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "模型: $LOCAL_QWEN"
echo "训练步数: $TOTAL_STEPS"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "学习率: $LR"
echo "========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=naive \
    data.train_files="$SCRATCH_BASE/verl-agent/test_script/data/train.parquet" \
    data.val_files="$SCRATCH_BASE/verl-agent/test_script/data/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$LOCAL_QWEN" \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=True \
    env.env_name="" \
    trainer.logger=[console] \
    trainer.project_name='gsm8k_grpo' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.val_before_train=True \
    trainer.val_only=False \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

echo ""
echo "========================================="
echo "训练完成!"
echo "========================================="
echo "输出目录: $OUTPUT_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "日志: ${OUTPUT_DIR}/training.log"

# 打印最终结果
echo ""
echo "--- 最终结果 ---"
grep -E "step:[0-9]+ .* test_score" "${OUTPUT_DIR}/training.log" | tail -5
echo "========================================="
