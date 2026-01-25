#!/bin/bash
# GSM8K PPO 训练脚本 - 使用自定义 Reward 函数

set -x

# 激活 conda 环境
source /p/home/jusers/liu52/jureca/miniforge3/etc/profile.d/conda.sh
conda activate verl-agent

ENGINE=${1:-vllm}

#会attention 降级到V0，暂时 comment
#export VLLM_ATTENTION_BACKEND=XFORMERS

# ====== 新增：强制全链路离线 & 关掉外部遥测/日志上报 ======
export HF_HOME="/p/scratch/westai0052/liu52/.cache/huggingface"
# 关键：必须设置为0以避免 repo id 验证，即使是离线环境
# transformers 会从本地缓存加载，不会真正联网
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
# vLLM 自身的遥测/网络
export VLLM_NO_USAGE_STATS=1
# Ray 分布式框架的遥测/网络（关键：离线环境必须禁用）
export RAY_USAGE_STATS_ENABLED=0
# 脚本里启用了 wandb logger，但无网会卡，建议禁用
export WANDB_DISABLED=true
export WANDB_MODE=offline

# ====== 关键：把模型路径改成本地绝对路径 ======
LOCAL_QWEN="/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct"

# ========================================
# 配置自定义 Reward 函数
# ========================================
CUSTOM_REWARD_PATH="/p/scratch/westai0052/liu52/verl-agent/test_script/custom_gsm8k_reward.py"
REWARD_FUNCTION_NAME="compute_score"  # 严格模式
# REWARD_FUNCTION_NAME="compute_score_flexible"  # 灵活模式

# 训练数据配置
# 注意：train_data_size 必须能被 (micro_batch_size * n_gpus) 整除
train_data_size=32      # 使用 32 个训练样本（2 GPUs，16*2=32）
val_data_size=128       # 验证数据保持 128
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
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    trainer.logger=[console] \
    trainer.project_name='verl_agent_GSM8K_custom_reward' \
    trainer.experiment_name='rollout_only_grpo_qwen2p5_1p5b_custom_reward' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    env.env_name=""\
    env.resources_per_worker.num_gpus=0\
    env.rollout.n=1 \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name="${REWARD_FUNCTION_NAME}" \
    "$@"
