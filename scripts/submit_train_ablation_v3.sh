#!/bin/bash
# V3 Ablation - Focus on Accuracy
# Key changes: alpha=2.0, gamma=0.02

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation_v3

echo "=============================================="
echo "ABLATION V3 - Focus on Accuracy"
echo "=============================================="
echo "Reward: R = 2.0*R1 + 0.3*R2 - 0.02*R3"
echo "(Task reward doubled, cost penalty reduced 5x)"
echo ""
echo "Training samples: 500"
echo "Learning rate: 1e-5"
echo "Epochs: 5"
echo "=============================================="
echo ""
echo "STEP 1: Submitting training jobs..."
echo ""

echo "Submitting train_v3_gsm8k_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v3_gsm8k_topk5.sh
echo "Submitting train_v3_gsm8k_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v3_gsm8k_topk10.sh
echo "Submitting train_v3_arc_challenge_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v3_arc_challenge_topk5.sh
echo "Submitting train_v3_arc_challenge_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v3_arc_challenge_topk10.sh
echo "Submitting train_v3_winogrande_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v3_winogrande_topk5.sh
echo "Submitting train_v3_winogrande_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v3_winogrande_topk10.sh

echo ""
echo "======================================"
echo "Training jobs submitted!"
echo ""
echo "After training completes, run:"
echo "  bash scripts/submit_eval_ablation_v3.sh"
