#!/bin/bash
# GSM8K GRPO 训练脚本 - 4 GPU H100 配置
# 包含模型保存和训练前后结果对比

set -x

# 激活 conda 环境
source /p/home/jusers/liu52/jureca/miniforge3/etc/profile.d/conda.sh
conda activate verl-agent

ENGINE=${1:-vllm}

# ====== 离线模式设置（工作节点无网）======
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
SCRIPT_DIR="/p/scratch/westai0052/liu52/verl-agent/test_script/single_agent_gsm8k"
OUTPUT_DIR="/p/scratch/westai0052/liu52/verl-agent/outputs/gsm8k_grpo_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "========================================="
echo "GSM8K GRPO 训练"
echo "========================================="
echo "模型: Qwen2.5-1.5B-Instruct"
echo "GPU数量: 4 x H100"
echo "输出目录: $OUTPUT_DIR"
echo "Checkpoint目录: $CHECKPOINT_DIR"
echo "========================================="

# ====== 自定义 Reward 函数 ======
CUSTOM_REWARD_PATH="/p/scratch/westai0052/liu52/verl-agent/test_script/custom_gsm8k_reward.py"
REWARD_FUNCTION_NAME="compute_score"

# ====== 训练数据配置（4 GPU）======
# train_data_size 必须能被 (micro_batch_size * n_gpus) 整除
# 16 * 4 = 64
train_data_size=64
val_data_size=128
group_size=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/p/scratch/westai0052/liu52/verl-agent/test_script/data/train.parquet" \
    data.val_files="/p/scratch/westai0052/liu52/verl-agent/test_script/data/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$LOCAL_QWEN" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=True \
    trainer.logger=[console] \
    trainer.project_name='verl_agent_GSM8K_grpo' \
    trainer.experiment_name='grpo_qwen2p5_1p5b_4gpu_h100' \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    env.env_name="" \
    env.resources_per_worker.num_gpus=0 \
    env.rollout.n=1 \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name="${REWARD_FUNCTION_NAME}" \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/training.log"

# ========================================
# 训练完成后提取结果对比
# ========================================
echo ""
echo "========================================="
echo "训练完成！正在生成结果对比报告..."
echo "========================================="

LOG_FILE="${OUTPUT_DIR}/training.log"
REPORT_FILE="${OUTPUT_DIR}/performance_report.txt"

{
    echo "========================================="
    echo "GSM8K GRPO 训练结果报告"
    echo "========================================="
    echo "时间: $(date)"
    echo "模型: Qwen2.5-1.5B-Instruct"
    echo "算法: GRPO"
    echo ""

    # 提取训练前验证分数
    echo "--- 训练前后对比 ---"
    BEFORE_SCORE=$(grep -E "step:\s*0.*test_score" "$LOG_FILE" | head -1 | grep -oP "test_score['\"]?:?\s*[\w.]*\(?\K[0-9.]+" | head -1)
    if [ -z "$BEFORE_SCORE" ]; then
        BEFORE_SCORE=$(grep -E "val.*test_score|test_score.*val" "$LOG_FILE" | head -1 | grep -oP "[0-9]+\.[0-9]+")
    fi

    # 提取训练后最终分数
    AFTER_SCORE=$(grep -E "step:.*test_score" "$LOG_FILE" | tail -1 | grep -oP "test_score['\"]?:?\s*[\w.]*\(?\K[0-9.]+" | head -1)
    if [ -z "$AFTER_SCORE" ]; then
        AFTER_SCORE=$(grep -E "val.*test_score|test_score.*val" "$LOG_FILE" | tail -1 | grep -oP "[0-9]+\.[0-9]+")
    fi

    echo "训练前 (Initial) Score: ${BEFORE_SCORE:-N/A}"
    echo "训练后 (Final) Score: ${AFTER_SCORE:-N/A}"

    if [ -n "$BEFORE_SCORE" ] && [ -n "$AFTER_SCORE" ]; then
        IMPROVEMENT=$(echo "scale=4; $AFTER_SCORE - $BEFORE_SCORE" | bc 2>/dev/null || echo "N/A")
        echo "绝对提升: $IMPROVEMENT"

        if [ "$IMPROVEMENT" != "N/A" ]; then
            if (( $(echo "$IMPROVEMENT > 0" | bc -l 2>/dev/null) )); then
                echo "结论: 训练有效，模型性能提升！"
            elif (( $(echo "$IMPROVEMENT == 0" | bc -l 2>/dev/null) )); then
                echo "结论: 性能持平"
            else
                echo "结论: 性能下降，需要检查训练配置"
            fi
        fi
    fi

    echo ""
    echo "--- 所有验证步骤的分数 ---"
    grep -E "step:.*val.*test_score|val.*step:.*test_score" "$LOG_FILE" | head -20

    echo ""
    echo "--- 保存的Checkpoint ---"
    ls -la "$CHECKPOINT_DIR" 2>/dev/null || echo "无checkpoint"

    echo ""
    echo "========================================="
    echo "完整日志: ${LOG_FILE}"
    echo "Checkpoint目录: ${CHECKPOINT_DIR}"
    echo "========================================="
} | tee "$REPORT_FILE"

echo ""
echo "报告已保存到: $REPORT_FILE"
