#!/bin/bash
# V4 Ablation - Find Optimal Accuracy-Efficiency Trade-off

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation_v4

echo "=============================================="
echo "ABLATION V4 - Find Optimal Trade-off"
echo "=============================================="
echo "Goal: Accuracy drop < 5%, maximize latent reduction"
echo "Reward: R = 1.0*R1 + 0.3*R2 (NO cost penalty!)"
echo "top_k range: [10, 20, 30, 40, 50]"
echo ""
echo "Training samples: 500"
echo "Learning rate: 1e-5"
echo "Epochs: 5"
echo "=============================================="
echo ""
echo "STEP 1: Submitting training jobs..."
echo ""

echo "Submitting train_v4_gsm8k_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_gsm8k_topk10.sh
echo "Submitting train_v4_gsm8k_topk20.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_gsm8k_topk20.sh
echo "Submitting train_v4_gsm8k_topk30.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_gsm8k_topk30.sh
echo "Submitting train_v4_gsm8k_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_gsm8k_topk40.sh
echo "Submitting train_v4_gsm8k_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_gsm8k_topk50.sh
echo "Submitting train_v4_arc_challenge_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_arc_challenge_topk10.sh
echo "Submitting train_v4_arc_challenge_topk20.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_arc_challenge_topk20.sh
echo "Submitting train_v4_arc_challenge_topk30.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_arc_challenge_topk30.sh
echo "Submitting train_v4_arc_challenge_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_arc_challenge_topk40.sh
echo "Submitting train_v4_arc_challenge_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_arc_challenge_topk50.sh
echo "Submitting train_v4_winogrande_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_winogrande_topk10.sh
echo "Submitting train_v4_winogrande_topk20.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_winogrande_topk20.sh
echo "Submitting train_v4_winogrande_topk30.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_winogrande_topk30.sh
echo "Submitting train_v4_winogrande_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_winogrande_topk40.sh
echo "Submitting train_v4_winogrande_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_winogrande_topk50.sh

echo ""
echo "======================================"
echo "Training jobs submitted!"
echo ""
echo "After training completes, run:"
echo "  bash scripts/submit_eval_ablation_v4.sh"
