#!/bin/bash
# Submit architecture ablation experiments
# Compare num_layers = [1, 2, 4] on GSM8K

cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/architecture

echo "Submitting architecture ablation jobs..."

# Test on GSM8K with different num_layers
# Note: num_layers=2 is our default, so we test 1 and 4 as alternatives
for layers in 1 4; do
    echo "Submitting: GSM8K layers=$layers"

    cat > scripts/run_arch_gsm8k_L${layers}.sh << EOF
#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=arch_L${layers}
#SBATCH --output=logs/arch_gsm8k_L${layers}_%j.out
#SBATCH --error=logs/arch_gsm8k_L${layers}_%j.err

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs/architecture

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 scripts/run_architecture_ablation.py --task gsm8k --num_layers $layers --top_k 40
EOF

    sbatch scripts/run_arch_gsm8k_L${layers}.sh
done

echo ""
echo "Submitted 2 architecture ablation jobs"
echo "This will compare: layers=1 vs layers=2 (default) vs layers=4"
