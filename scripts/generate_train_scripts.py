#!/usr/bin/env python3
"""Generate training scripts for all model-dataset combinations."""

import os

SCRIPT_TEMPLATE = '''#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}G
#SBATCH --time={time}
#SBATCH --job-name=train_{job_name}
#SBATCH --output=logs/train_{job_name}_%j.out
#SBATCH --error=logs/train_{job_name}_%j.err

# Train reading policy: {model_name} on {task}

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

export CUDA_VISIBLE_DEVICES={cuda_devices}
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 run_rl_train.py \\
    --model_name {model_path} \\
    --task {task} \\
    --prompt sequential \\
    --use_vllm \\
    --device cuda:0 \\
    --device2 cuda:1 \\
    --tensor_parallel_size 1 \\
    --gpu_memory_utilization 0.85 \\
    --enable_prefix_caching \\
    --use_second_HF_model \\
    --max_new_tokens 512 \\
    --latent_steps 3 \\
    --temperature 0.6 \\
    --top_p 0.95 \\
    --top_k_blocks 3 \\
    --policy_num_heads 4 \\
    --policy_num_layers 1 \\
    --policy_dropout 0.1 \\
    --similarity_threshold 0.85 \\
    --min_block_size 4 \\
    --max_block_size 64 \\
    --segment_layer_idx {segment_layer_idx} \\
    --policy_lr 5e-5 \\
    --grpo_group_size 4 \\
    --grpo_clip_epsilon 0.2 \\
    --grpo_kl_coef 0.05 \\
    --entropy_coef 0.02 \\
    --ref_policy_update_freq 50 \\
    --update_every_n_groups 2 \\
    --reward_alpha 1.0 \\
    --reward_beta 0.3 \\
    --reward_gamma 0.5 \\
    --num_epochs 5 \\
    --max_samples {max_samples} \\
    --eval_samples 100 \\
    --max_grad_norm 1.0 \\
    --seed 42 \\
    --output_dir checkpoints/rl_policy/{model_key}_{task}_$SLURM_JOB_ID

echo "Training completed!"
'''

# Model configurations
MODELS = {
    'qwen3_4b': {
        'name': 'Qwen3-4B',
        'path': '/p/scratch/westai0052/liu52/models/Qwen3-4B',
        'n_gpus': 2,
        'mem': 64,
        'segment_layer_idx': 16,
    },
    'qwen3_8b': {
        'name': 'Qwen3-8B',
        'path': '/p/scratch/westai0052/liu52/models/Qwen3-8B',
        'n_gpus': 2,
        'mem': 80,
        'segment_layer_idx': 20,
    },
    'qwen3_14b': {
        'name': 'Qwen3-14B',
        'path': '/p/scratch/westai0052/liu52/models/Qwen3-14B',
        'n_gpus': 4,
        'mem': 128,
        'segment_layer_idx': 24,
    },
}

# Dataset configurations for training
DATASETS = {
    'gsm8k': {
        'max_samples': 500,
        'time': '04:00:00',
    },
    'arc_challenge': {
        'max_samples': 500,
        'time': '04:00:00',
    },
    'winogrande': {
        'max_samples': 500,
        'time': '04:00:00',
    },
}

def main():
    scripts_dir = '/p/scratch/westai0052/liu52/LatentMAS/scripts'

    for model_key, model_cfg in MODELS.items():
        for task_key, task_cfg in DATASETS.items():
            job_name = f"{model_key}_{task_key}"

            # Generate CUDA devices string
            cuda_devices = ','.join(str(i) for i in range(model_cfg['n_gpus']))

            script_content = SCRIPT_TEMPLATE.format(
                n_gpus=model_cfg['n_gpus'],
                mem=model_cfg['mem'],
                time=task_cfg['time'],
                job_name=job_name,
                model_name=model_cfg['name'],
                model_path=model_cfg['path'],
                model_key=model_key,
                task=task_key,
                cuda_devices=cuda_devices,
                max_samples=task_cfg['max_samples'],
                segment_layer_idx=model_cfg['segment_layer_idx'],
            )

            script_path = os.path.join(scripts_dir, f'train_{job_name}.sh')
            with open(script_path, 'w') as f:
                f.write(script_content)

            print(f'Generated: {script_path}')

    print(f'\nTotal scripts generated: {len(MODELS) * len(DATASETS)}')

if __name__ == '__main__':
    main()
