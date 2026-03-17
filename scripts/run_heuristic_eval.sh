#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=heuristic_eval
#SBATCH --output=logs/heuristic_eval_%j.out
#SBATCH --error=logs/heuristic_eval_%j.err

# Evaluate all heuristic baselines on all datasets

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/heuristics

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

echo "=========================================="
echo "Evaluating Heuristic Baselines"
echo "=========================================="
echo ""

# GSM8K
echo "=== GSM8K k=10 ==="
for method in random recency similarity time_weighted; do
    echo "Method: $method"
    python3 scripts/eval_heuristics.py --task gsm8k --method $method --top_k 10
done

# ARC
echo ""
echo "=== ARC k=10 ==="
for method in random recency similarity time_weighted; do
    echo "Method: $method"
    python3 scripts/eval_heuristics.py --task arc_challenge --method $method --top_k 10
done

# Winogrande
echo ""
echo "=== Winogrande k=10 ==="
for method in random recency similarity time_weighted; do
    echo "Method: $method"
    python3 scripts/eval_heuristics.py --task winogrande --method $method --top_k 10
done

echo ""
echo "=========================================="
echo "Heuristic evaluation completed!"
echo "Results in: logs/heuristics/"
echo "=========================================="
