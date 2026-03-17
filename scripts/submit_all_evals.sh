#!/bin/bash
# Submit all evaluation jobs in parallel

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "Submitting all evaluation jobs..."
echo "================================="

# Qwen3-4B jobs
echo "Submitting Qwen3-4B jobs..."
JOB1=$(sbatch scripts/eval_qwen3_4b_gsm8k.sh | awk '{print $4}')
JOB2=$(sbatch scripts/eval_qwen3_4b_arc_challenge.sh | awk '{print $4}')
JOB3=$(sbatch scripts/eval_qwen3_4b_winogrande.sh | awk '{print $4}')
echo "  Qwen3-4B + gsm8k: $JOB1"
echo "  Qwen3-4B + arc_challenge: $JOB2"
echo "  Qwen3-4B + winogrande: $JOB3"

# Qwen3-8B jobs
echo "Submitting Qwen3-8B jobs..."
JOB4=$(sbatch scripts/eval_qwen3_8b_gsm8k.sh | awk '{print $4}')
JOB5=$(sbatch scripts/eval_qwen3_8b_arc_challenge.sh | awk '{print $4}')
JOB6=$(sbatch scripts/eval_qwen3_8b_winogrande.sh | awk '{print $4}')
echo "  Qwen3-8B + gsm8k: $JOB4"
echo "  Qwen3-8B + arc_challenge: $JOB5"
echo "  Qwen3-8B + winogrande: $JOB6"

# Qwen3-14B jobs
echo "Submitting Qwen3-14B jobs..."
JOB7=$(sbatch scripts/eval_qwen3_14b_gsm8k.sh | awk '{print $4}')
JOB8=$(sbatch scripts/eval_qwen3_14b_arc_challenge.sh | awk '{print $4}')
JOB9=$(sbatch scripts/eval_qwen3_14b_winogrande.sh | awk '{print $4}')
echo "  Qwen3-14B + gsm8k: $JOB7"
echo "  Qwen3-14B + arc_challenge: $JOB8"
echo "  Qwen3-14B + winogrande: $JOB9"

echo ""
echo "================================="
echo "All 9 jobs submitted!"
echo ""
echo "Check status with: squeue -u $USER"
echo "View results in: logs/results_*.json"
