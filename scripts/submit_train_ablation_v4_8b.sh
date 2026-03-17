#!/bin/bash
# V4 8B Training - Qwen3-8B experiments

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs checkpoints/ablation_v4_8b

echo "=============================================="
echo "ABLATION V4 - Qwen3-8B"
echo "=============================================="
echo "Training with key top_k values: 10, 40, 50"
echo ""

echo "Submitting train_v4_8b_gsm8k_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_8b_gsm8k_topk10.sh
echo "Submitting train_v4_8b_gsm8k_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_8b_gsm8k_topk40.sh
echo "Submitting train_v4_8b_gsm8k_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_8b_gsm8k_topk50.sh
echo "Submitting train_v4_8b_arc_challenge_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_8b_arc_challenge_topk10.sh
echo "Submitting train_v4_8b_arc_challenge_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_8b_arc_challenge_topk40.sh
echo "Submitting train_v4_8b_arc_challenge_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_8b_arc_challenge_topk50.sh
echo "Submitting train_v4_8b_winogrande_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_8b_winogrande_topk10.sh
echo "Submitting train_v4_8b_winogrande_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_8b_winogrande_topk40.sh
echo "Submitting train_v4_8b_winogrande_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/train_v4_8b_winogrande_topk50.sh

echo ""
echo "Training jobs submitted!"
echo "After training, run: bash scripts/submit_eval_ablation_v4_8b.sh"
