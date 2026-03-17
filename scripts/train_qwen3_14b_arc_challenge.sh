#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=train_qwen3_14b_arc_challenge
#SBATCH --output=logs/train_qwen3_14b_arc_challenge_%j.out
#SBATCH --error=logs/train_qwen3_14b_arc_challenge_%j.err

# Train reading policy: Qwen3-14B on arc_challenge

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 run_rl_train.py \
    --model_name /p/scratch/westai0052/liu52/models/Qwen3-14B \
    --task arc_challenge \
    --prompt sequential \
    --use_vllm \
    --device cuda:0 \
    --device2 cuda:1 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --enable_prefix_caching \
    --use_second_HF_model \
    --max_new_tokens 512 \
    --latent_steps 3 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k_blocks 3 \
    --policy_num_heads 4 \
    --policy_num_layers 1 \
    --policy_dropout 0.1 \
    --similarity_threshold 0.85 \
    --min_block_size 4 \
    --max_block_size 64 \
    --segment_layer_idx 24 \
    --policy_lr 5e-5 \
    --grpo_group_size 4 \
    --grpo_clip_epsilon 0.2 \
    --grpo_kl_coef 0.05 \
    --entropy_coef 0.02 \
    --ref_policy_update_freq 50 \
    --update_every_n_groups 2 \
    --reward_alpha 1.0 \
    --reward_beta 0.3 \
    --reward_gamma 0.5 \
    --num_epochs 5 \
    --max_samples 500 \
    --eval_samples 100 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --output_dir checkpoints/rl_policy/qwen3_14b_arc_challenge_$SLURM_JOB_ID

echo "Training completed!"
