#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=rew_consistency_only
#SBATCH --output=logs/reward_gsm8k_consistency_only_%j.out
#SBATCH --error=logs/reward_gsm8k_consistency_only_%j.err

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/reward_ablation

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 scripts/run_reward_ablation.py --task gsm8k --config consistency_only --top_k 40
