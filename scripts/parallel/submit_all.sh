#!/bin/bash
# ============================================================
# Submit all parallel validation jobs
#
# This submits 6 jobs in parallel:
#   - 3 jobs for 1.5B model (gsm8k, arc, winogrande)
#   - 3 jobs for 8B model (gsm8k, arc, winogrande)
#
# Total GPU hours: 3*4h + 3*8h = 36 GPU-hours
# But runs in parallel, so wall time = max(4h, 8h) = 8h
# ============================================================

cd /p/scratch/westai0052/liu52/LatentMAS/scripts/parallel

echo "Submitting 6 parallel validation jobs..."
echo ""

# 1.5B model jobs (2 GPUs each, 4h)
echo "=== 1.5B Model Jobs ==="
JOB1=$(sbatch run_1.5b_gsm8k.sh | awk '{print $4}')
echo "  GSM8K:      Job $JOB1"

JOB2=$(sbatch run_1.5b_arc.sh | awk '{print $4}')
echo "  ARC:        Job $JOB2"

JOB3=$(sbatch run_1.5b_wino.sh | awk '{print $4}')
echo "  Winogrande: Job $JOB3"

echo ""

# 8B model jobs (4 GPUs each, 8h)
echo "=== 8B Model Jobs ==="
JOB4=$(sbatch run_8b_gsm8k.sh | awk '{print $4}')
echo "  GSM8K:      Job $JOB4"

JOB5=$(sbatch run_8b_arc.sh | awk '{print $4}')
echo "  ARC:        Job $JOB5"

JOB6=$(sbatch run_8b_wino.sh | awk '{print $4}')
echo "  Winogrande: Job $JOB6"

echo ""
echo "============================================================"
echo "All 6 jobs submitted!"
echo ""
echo "Check status: squeue -u \$USER"
echo ""
echo "Results will be in:"
echo "  checkpoints/rl_policy/val_1.5b_*/"
echo "  checkpoints/rl_policy/val_8b_*/"
echo ""
echo "View comparison after completion:"
echo "  cat checkpoints/rl_policy/val_*/training_log.json | jq '.comparison'"
echo "============================================================"
