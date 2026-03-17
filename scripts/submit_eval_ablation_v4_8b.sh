#!/bin/bash
# V4 8B Evaluation

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "Submitting V4 8B evaluation jobs..."

echo "Submitting eval_v4_8b_gsm8k_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_8b_gsm8k_topk10.sh
echo "Submitting eval_v4_8b_gsm8k_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_8b_gsm8k_topk40.sh
echo "Submitting eval_v4_8b_gsm8k_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_8b_gsm8k_topk50.sh
echo "Submitting eval_v4_8b_arc_challenge_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_8b_arc_challenge_topk10.sh
echo "Submitting eval_v4_8b_arc_challenge_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_8b_arc_challenge_topk40.sh
echo "Submitting eval_v4_8b_arc_challenge_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_8b_arc_challenge_topk50.sh
echo "Submitting eval_v4_8b_winogrande_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_8b_winogrande_topk10.sh
echo "Submitting eval_v4_8b_winogrande_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_8b_winogrande_topk40.sh
echo "Submitting eval_v4_8b_winogrande_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_8b_winogrande_topk50.sh

echo ""
echo "Evaluation jobs submitted!"
echo "Results: logs/ablation_v4_8b_*.json"
