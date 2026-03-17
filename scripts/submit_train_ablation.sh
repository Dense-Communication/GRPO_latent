#!/bin/bash
# Step 1: Submit training jobs
# Step 2: After training completes, submit evaluation jobs

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation

echo "STEP 1: Submitting training jobs..."
echo "======================================"

echo "Submitting train_gsm8k_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_gsm8k_topk5.sh
echo "Submitting train_gsm8k_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_gsm8k_topk10.sh
echo "Submitting train_arc_challenge_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_arc_challenge_topk5.sh
echo "Submitting train_arc_challenge_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_arc_challenge_topk10.sh
echo "Submitting train_winogrande_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_winogrande_topk5.sh
echo "Submitting train_winogrande_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_winogrande_topk10.sh

echo ""
echo "======================================"
echo "Training jobs submitted!"
echo ""
echo "After training completes, run:"
echo "  bash scripts/submit_eval_ablation.sh"
