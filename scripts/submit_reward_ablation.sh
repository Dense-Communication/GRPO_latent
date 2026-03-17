#!/bin/bash
# Submit reward ablation experiments
# Compare different reward weight configurations on GSM8K

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/reward_ablation

echo "Submitting reward ablation jobs..."

# Configurations to test (balanced is already tested in V4)
CONFIGS=("task_only" "consistency_only" "task_focused" "with_cost")

for config in "${CONFIGS[@]}"; do
    echo "Submitting: GSM8K $config"

    cat > scripts/run_reward_gsm8k_${config}.sh << EOF
#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=rew_${config}
#SBATCH --output=logs/reward_gsm8k_${config}_%j.out
#SBATCH --error=logs/reward_gsm8k_${config}_%j.err

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/reward_ablation

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 scripts/run_reward_ablation.py --task gsm8k --config $config --top_k 40
EOF

    sbatch scripts/run_reward_gsm8k_${config}.sh
done

echo ""
echo "Submitted 4 reward ablation jobs"
echo "Configs: task_only, consistency_only, task_focused, with_cost"
echo "(balanced is already in V4 results)"
