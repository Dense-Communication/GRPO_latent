#!/bin/bash
# V4 Ablation - Submit evaluation jobs

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "STEP 2: Submitting V4 evaluation jobs..."
echo "======================================"

echo "Submitting eval_v4_gsm8k_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_gsm8k_topk10.sh
echo "Submitting eval_v4_gsm8k_topk20.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_gsm8k_topk20.sh
echo "Submitting eval_v4_gsm8k_topk30.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_gsm8k_topk30.sh
echo "Submitting eval_v4_gsm8k_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_gsm8k_topk40.sh
echo "Submitting eval_v4_gsm8k_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_gsm8k_topk50.sh
echo "Submitting eval_v4_arc_challenge_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_arc_challenge_topk10.sh
echo "Submitting eval_v4_arc_challenge_topk20.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_arc_challenge_topk20.sh
echo "Submitting eval_v4_arc_challenge_topk30.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_arc_challenge_topk30.sh
echo "Submitting eval_v4_arc_challenge_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_arc_challenge_topk40.sh
echo "Submitting eval_v4_arc_challenge_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_arc_challenge_topk50.sh
echo "Submitting eval_v4_winogrande_topk10.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_winogrande_topk10.sh
echo "Submitting eval_v4_winogrande_topk20.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_winogrande_topk20.sh
echo "Submitting eval_v4_winogrande_topk30.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_winogrande_topk30.sh
echo "Submitting eval_v4_winogrande_topk40.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_winogrande_topk40.sh
echo "Submitting eval_v4_winogrande_topk50.sh..."
sbatch /p/scratch/westai0052/liu52/LatentMAS/scripts/eval_v4_winogrande_topk50.sh

echo ""
echo "======================================"
echo "Evaluation jobs submitted!"
echo "Results will be in: logs/ablation_v4_*.json"
