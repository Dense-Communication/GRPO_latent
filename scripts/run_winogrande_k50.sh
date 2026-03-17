#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=wino_k50
#SBATCH --output=logs/winogrande_k50_%j.out
#SBATCH --error=logs/winogrande_k50_%j.err

# Complete missing Winogrande k=50 evaluation

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

echo "=========================================="
echo "Winogrande k=50 Evaluation (Missing)"
echo "=========================================="

python3 scripts/run_topk_ablation_v4.py --task winogrande --top_k 50

echo "Done!"
