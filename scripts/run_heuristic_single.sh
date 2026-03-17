#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=heur_%1_%2
#SBATCH --output=logs/heuristic_%1_%2_%j.out
#SBATCH --error=logs/heuristic_%1_%2_%j.err

# Single heuristic evaluation
# Usage: sbatch run_heuristic_single.sh <task> <method>

TASK=$1
METHOD=$2
TOP_K=10

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/heuristics

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

echo "=========================================="
echo "Heuristic: $TASK - $METHOD - k=$TOP_K"
echo "=========================================="

python3 scripts/eval_heuristics.py --task $TASK --method $METHOD --top_k $TOP_K --test_samples 500

echo "Done!"
