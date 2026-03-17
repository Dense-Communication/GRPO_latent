#!/bin/bash
# Submit all TEST SET evaluation jobs
# These evaluate TRAINED policies on UNSEEN test data to check for overfitting

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

echo "Submitting TEST SET evaluation jobs..."
echo "======================================="
echo "These use UNSEEN test data to verify no overfitting"
echo ""

# Qwen3-4B jobs
echo "Submitting Qwen3-4B test evaluations..."
JOB1=$(sbatch scripts/test_qwen3_4b_gsm8k.sh 2>&1)
JOB2=$(sbatch scripts/test_qwen3_4b_arc_challenge.sh 2>&1)
JOB3=$(sbatch scripts/test_qwen3_4b_winogrande.sh 2>&1)
echo "  Qwen3-4B + gsm8k (test): $JOB1"
echo "  Qwen3-4B + arc_challenge (test): $JOB2"
echo "  Qwen3-4B + winogrande (validation): $JOB3"

# Qwen3-8B jobs
echo "Submitting Qwen3-8B test evaluations..."
JOB4=$(sbatch scripts/test_qwen3_8b_gsm8k.sh 2>&1)
JOB5=$(sbatch scripts/test_qwen3_8b_arc_challenge.sh 2>&1)
JOB6=$(sbatch scripts/test_qwen3_8b_winogrande.sh 2>&1)
echo "  Qwen3-8B + gsm8k (test): $JOB4"
echo "  Qwen3-8B + arc_challenge (test): $JOB5"
echo "  Qwen3-8B + winogrande (validation): $JOB6"

# Qwen3-14B jobs
echo "Submitting Qwen3-14B test evaluations..."
JOB7=$(sbatch scripts/test_qwen3_14b_gsm8k.sh 2>&1)
JOB8=$(sbatch scripts/test_qwen3_14b_arc_challenge.sh 2>&1)
JOB9=$(sbatch scripts/test_qwen3_14b_winogrande.sh 2>&1)
echo "  Qwen3-14B + gsm8k (test): $JOB7"
echo "  Qwen3-14B + arc_challenge (test): $JOB8"
echo "  Qwen3-14B + winogrande (validation): $JOB9"

echo ""
echo "======================================="
echo "All 9 test evaluation jobs submitted!"
echo ""
echo "Check status with: squeue -u $USER"
echo "Results will be saved to: logs/test_results_*.json"
