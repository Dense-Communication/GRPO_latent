#!/bin/bash
# Submit all heuristic baseline evaluations as separate jobs

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/heuristics

echo "Submitting heuristic baseline jobs..."

# Tasks and methods
TASKS=("gsm8k" "arc_challenge" "winogrande")
METHODS=("random" "recency" "similarity" "time_weighted")

for task in "${TASKS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo "Submitting: $task - $method"
        sbatch --job-name="heur_${task}_${method}" \
               --output="logs/heuristic_${task}_${method}_%j.out" \
               --error="logs/heuristic_${task}_${method}_%j.err" \
               scripts/run_heuristic_single.sh $task $method
    done
done

echo ""
echo "Submitted 12 heuristic jobs"
echo "Check status: squeue -u liu52"
