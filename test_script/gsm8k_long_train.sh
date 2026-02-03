#!/bin/bash
# GSM8K 长时间训练 - GRPO算法
# 支持固定验证子集进行频繁测试
# 用于观察训练过程中 reward 和 accuracy 的变化趋势

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
EXPERIMENT_NAME="gsm8k_long_train_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$SCRATCH_BASE/verl-agent/outputs/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# ========== 数据配置 ==========
# 训练数据
TRAIN_DATA="$SCRATCH_BASE/verl-agent/test_script/data/train.parquet"
# 完整验证集 (用于最终评估)
FULL_VAL_DATA="$SCRATCH_BASE/verl-agent/test_script/data/test.parquet"
# 固定验证子集 (用于频繁测试) - 使用固定子集加速验证
VAL_SUBSET_DATA="$SCRATCH_BASE/verl-agent/test_script/data/val_subset.parquet"

# 选择使用哪个验证集 (子集更快，完整更准确)
# 默认使用子集进行频繁验证
USE_VAL_SUBSET=${USE_VAL_SUBSET:-true}

if [ "$USE_VAL_SUBSET" = true ]; then
    VAL_DATA="$VAL_SUBSET_DATA"
    echo "使用固定验证子集: $VAL_DATA"
else
    VAL_DATA="$FULL_VAL_DATA"
    echo "使用完整验证集: $VAL_DATA"
fi

# ========== 超参数 ==========
# Batch size
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-4}

# 序列长度
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# 训练步数 - 长时间训练
TOTAL_STEPS=${TOTAL_STEPS:-500}        # 增加总训练步数
SAVE_FREQ=${SAVE_FREQ:-100}            # 保存checkpoint频率
TEST_FREQ=${TEST_FREQ:-5}              # 验证频率 (更频繁的验证以观察趋势)

# 学习率
LR=${LR:-1e-6}

# KL惩罚
KL_COEF=${KL_COEF:-0.05}

echo "========================================="
echo "GSM8K GRPO 长时间训练 (趋势分析)"
echo "========================================="
echo "实验名称: $EXPERIMENT_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "模型: $LOCAL_QWEN"
echo "训练步数: $TOTAL_STEPS"
echo "验证频率: 每 $TEST_FREQ 步"
echo "保存频率: 每 $SAVE_FREQ 步"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "学习率: $LR"
echo "验证数据: $VAL_DATA"
echo "========================================="

# 记录实验配置
cat > "${OUTPUT_DIR}/config.txt" << EOF
实验名称: $EXPERIMENT_NAME
开始时间: $(date)
模型: $LOCAL_QWEN
训练数据: $TRAIN_DATA
验证数据: $VAL_DATA
使用验证子集: $USE_VAL_SUBSET

超参数:
- 总训练步数: $TOTAL_STEPS
- 验证频率: $TEST_FREQ
- 保存频率: $SAVE_FREQ
- 训练批量大小: $TRAIN_BATCH_SIZE
- 验证批量大小: $VAL_BATCH_SIZE
- 学习率: $LR
- KL系数: $KL_COEF
- 最大prompt长度: $MAX_PROMPT_LENGTH
- 最大response长度: $MAX_RESPONSE_LENGTH
EOF

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=naive \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
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
    trainer.project_name='gsm8k_grpo_long' \
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

# 运行分析脚本
echo ""
echo "--- 运行训练指标分析 ---"
python3 "$SCRATCH_BASE/verl-agent/test_script/analyze_training_metrics.py" \
    "${OUTPUT_DIR}/training.log" --plot

# 打印最终结果摘要
echo ""
echo "--- 验证集结果 (每 $TEST_FREQ 步) ---"
grep -oP "step:\d+ .*?val/openai/gsm8k/test_score:\d+\.\d+" "${OUTPUT_DIR}/training.log" | \
    sed 's/.*step:\([0-9]*\).*test_score:\([0-9.]*\).*/step:\1 test_score:\2/'

echo "========================================="
