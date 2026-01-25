#!/bin/bash
# GRPO 最小训练脚本 - 4×H100 GPU
# 目标：验证完整训练流程，展示性能提升

set -x

# 激活环境
source /p/home/jusers/liu52/jureca/miniforge3/etc/profile.d/conda.sh
conda activate verl-agent

# ====== 离线环境配置 ======
export HF_HOME="/p/scratch/westai0052/liu52/.cache/huggingface"
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0
export WANDB_DISABLED=true
export WANDB_MODE=offline

# ====== 模型和数据路径 ======
MODEL_PATH="/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct"
DATA_DIR="/p/scratch/westai0052/liu52/verl-agent/test_script/data"
REWARD_PATH="/p/scratch/westai0052/liu52/verl-agent/test_script/custom_gsm8k_reward.py"
OUTPUT_DIR="/p/scratch/westai0052/liu52/verl-agent/outputs/grpo_gsm8k_minimal"

# ====== 训练参数 (4×H100 优化) ======
# 批次大小：64 = 16 * 4 GPUs
TRAIN_BATCH=64
VAL_BATCH=64
MICRO_BATCH=8
N_GPUS=4
EPOCHS=3

echo "=============================================="
echo "GRPO 最小训练 - GSM8K"
echo "模型: Qwen2.5-1.5B-Instruct"
echo "GPU: ${N_GPUS}×H100"
echo "训练批次: ${TRAIN_BATCH}, 验证批次: ${VAL_BATCH}"
echo "训练轮数: ${EPOCHS}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=============================================="

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${TRAIN_BATCH} \
    data.val_batch_size=${VAL_BATCH} \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.3 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH} \
    algorithm.use_kl_in_reward=False \
    trainer.logger=[console] \
    trainer.project_name='grpo_gsm8k_minimal' \
    trainer.experiment_name='qwen2p5_1p5b_4gpu' \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=${EPOCHS} \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    env.env_name="" \
    env.resources_per_worker.num_gpus=0 \
    env.rollout.n=1 \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name="compute_score" \
    "$@"

echo "=============================================="
echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
echo "=============================================="
