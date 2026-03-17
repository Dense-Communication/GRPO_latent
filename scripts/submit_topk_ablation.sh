#!/bin/bash
# Submit all top-k ablation experiments
# PROPER SPLIT: Train on TRAIN, Evaluate on TEST (unseen)

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "Submitting Top-K Ablation Experiments..."
echo "========================================="
echo "Train: TRAIN split (300 samples)"
echo "Test: TEST split (500 samples) - UNSEEN"
echo ""

echo "Submitting ablation_gsm8k_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/ablation_gsm8k_topk5.sh

echo "Submitting ablation_gsm8k_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/ablation_gsm8k_topk10.sh

echo "Submitting ablation_arc_challenge_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/ablation_arc_challenge_topk5.sh

echo "Submitting ablation_arc_challenge_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/ablation_arc_challenge_topk10.sh

echo "Submitting ablation_winogrande_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/ablation_winogrande_topk5.sh

echo "Submitting ablation_winogrande_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/ablation_winogrande_topk10.sh


echo "========================================="
echo "All ablation experiments submitted!"
echo "Check status: squeue -u $USER"
echo "Results will be in: logs/ablation_*.json"
