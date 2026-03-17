#!/bin/bash
# V3 Ablation - Submit evaluation jobs

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "STEP 2: Submitting V3 evaluation jobs..."
echo "======================================"

echo "Submitting eval_v3_gsm8k_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v3_gsm8k_topk5.sh
echo "Submitting eval_v3_gsm8k_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v3_gsm8k_topk10.sh
echo "Submitting eval_v3_arc_challenge_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v3_arc_challenge_topk5.sh
echo "Submitting eval_v3_arc_challenge_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v3_arc_challenge_topk10.sh
echo "Submitting eval_v3_winogrande_topk5.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v3_winogrande_topk5.sh
echo "Submitting eval_v3_winogrande_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v3_winogrande_topk10.sh

echo ""
echo "======================================"
echo "Evaluation jobs submitted!"
echo "Results will be in: logs/ablation_v3_*.json"
