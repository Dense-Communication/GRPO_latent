#!/bin/bash
# Submit evaluation jobs (run after training completes)

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "STEP 2: Submitting evaluation jobs..."
echo "======================================"

echo "Submitting eval_gsm8k_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_gsm8k_topk5.sh
echo "Submitting eval_gsm8k_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_gsm8k_topk10.sh
echo "Submitting eval_arc_challenge_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_arc_challenge_topk5.sh
echo "Submitting eval_arc_challenge_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_arc_challenge_topk10.sh
echo "Submitting eval_winogrande_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_winogrande_topk5.sh
echo "Submitting eval_winogrande_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_winogrande_topk10.sh

echo ""
echo "======================================"
echo "Evaluation jobs submitted!"
echo "Results will be in: logs/ablation_*.json"
