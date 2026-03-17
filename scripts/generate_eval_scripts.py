#!/usr/bin/env python3
"""Generate evaluation scripts for all model-dataset combinations."""

import os

SCRIPT_TEMPLATE = '''#!/bin/bash
#SBATCH --account=westai0052
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}G
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --output=logs/eval_{job_name}_%j.out
#SBATCH --error=logs/eval_{job_name}_%j.err

# {model_name} + {task} evaluation

source ~/.bashrc
conda activate latentmas
cd /p/scratch/westai0052/liu52/LatentMAS
mkdir -p logs

export CUDA_VISIBLE_DEVICES={cuda_devices}
export HF_DATASETS_CACHE="/p/scratch/westai0052/liu52/.cache/huggingface/datasets"

python3 << 'EOF'
import torch
import sys
import json
from datetime import datetime

MODEL_NAME = "{model_name}"
MODEL_PATH = "{model_path}"
TASK = "{task}"
N_SAMPLES = {n_samples}
LOAD_FN = "{load_fn}"

print("="*60)
print(f"Evaluation: {{MODEL_NAME}} on {{TASK}}")
print(f"Samples: {{N_SAMPLES}}")
print("="*60)
sys.stdout.flush()

from data import {load_fn}
from models import ModelWrapper
from methods.latent_mas_rl import LatentMASMethodRL
from policy import ReadingPolicyNetwork
from utils import set_seed

set_seed(42)

class Args:
    model_name = MODEL_PATH
    prompt = 'sequential'
    method = 'latent_mas'
    device = 'cuda:0'
    device2 = 'cuda:1'
    think = False
    latent_steps = 3
    top_k_blocks = 3
    similarity_threshold = 0.85
    min_block_size = 4
    max_block_size = 64
    segment_layer_idx = {segment_layer_idx}
    tensor_parallel_size = 1
    gpu_memory_utilization = 0.85
    enable_prefix_caching = True
    use_second_HF_model = True
    latent_space_realign = False
    max_new_tokens = 512
    task = TASK

args = Args()

print('Loading model...')
sys.stdout.flush()
model = ModelWrapper(args.model_name, device=args.device, use_vllm=True, args=args)

hidden_dim = model.HF_model.config.hidden_size
policy = ReadingPolicyNetwork(hidden_dim=hidden_dim, num_heads=4, num_layers=1).to(args.device).to(torch.bfloat16)

# Load data
data = list({load_fn}(split='{data_split}'))[:N_SAMPLES]
print(f'Loaded {{len(data)}} samples')
sys.stdout.flush()

# Baseline evaluation
print('\\n' + '='*60)
print('BASELINE (random block selection)')
print('='*60)
sys.stdout.flush()

method = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=None, top_k_blocks=args.top_k_blocks, rl_training=False,
)
method.task = TASK

baseline_correct = 0
baseline_total_blocks = 0
baseline_selected_blocks = 0
for i, item in enumerate(data):
    result = method.run_batch_vllm([item])[0]
    if result.get('correct', False):
        baseline_correct += 1
    # Track block statistics
    stats = method.get_efficiency_stats()
    baseline_total_blocks += stats['total_blocks']
    baseline_selected_blocks += stats['selected_blocks']
    if (i + 1) % 20 == 0:
        print(f'  Baseline progress: {{i+1}}/{{N_SAMPLES}}, acc={{baseline_correct}}/{{i+1}}')
        sys.stdout.flush()

baseline_acc = baseline_correct / N_SAMPLES
baseline_read_ratio = baseline_selected_blocks / max(baseline_total_blocks, 1)
print(f'\\nBaseline accuracy: {{baseline_correct}}/{{N_SAMPLES}} = {{baseline_acc*100:.2f}}%')
print(f'Baseline latent read: {{baseline_selected_blocks}}/{{baseline_total_blocks}} blocks ({{baseline_read_ratio*100:.2f}}%)')
sys.stdout.flush()

# Policy evaluation
print('\\n' + '='*60)
print('WITH POLICY (untrained)')
print('='*60)
sys.stdout.flush()

method2 = LatentMASMethodRL(
    model, latent_steps=args.latent_steps, temperature=0.6, top_p=0.95,
    args=args, reading_policy=policy, top_k_blocks=args.top_k_blocks, rl_training=False,
)
method2.task = TASK

policy_correct = 0
policy_total_blocks = 0
policy_selected_blocks = 0
for i, item in enumerate(data):
    result = method2.run_batch_vllm([item])[0]
    if result.get('correct', False):
        policy_correct += 1
    # Track block statistics
    stats = method2.get_efficiency_stats()
    policy_total_blocks += stats['total_blocks']
    policy_selected_blocks += stats['selected_blocks']
    if (i + 1) % 20 == 0:
        print(f'  Policy progress: {{i+1}}/{{N_SAMPLES}}, acc={{policy_correct}}/{{i+1}}')
        sys.stdout.flush()

policy_acc = policy_correct / N_SAMPLES
policy_read_ratio = policy_selected_blocks / max(policy_total_blocks, 1)
print(f'\\nPolicy accuracy: {{policy_correct}}/{{N_SAMPLES}} = {{policy_acc*100:.2f}}%')
print(f'Policy latent read: {{policy_selected_blocks}}/{{policy_total_blocks}} blocks ({{policy_read_ratio*100:.2f}}%)')
sys.stdout.flush()

# Summary
print('\\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'Model: {{MODEL_NAME}}')
print(f'Task: {{TASK}}')
print(f'Samples: {{N_SAMPLES}}')
print(f'')
print(f'ACCURACY:')
print(f'  Baseline: {{baseline_correct}}/{{N_SAMPLES}} ({{baseline_acc*100:.2f}}%)')
print(f'  Policy:   {{policy_correct}}/{{N_SAMPLES}} ({{policy_acc*100:.2f}}%)')
print(f'')
print(f'LATENT READ EFFICIENCY:')
print(f'  Baseline: {{baseline_selected_blocks}}/{{baseline_total_blocks}} blocks ({{baseline_read_ratio*100:.2f}}%)')
print(f'  Policy:   {{policy_selected_blocks}}/{{policy_total_blocks}} blocks ({{policy_read_ratio*100:.2f}}%)')
latent_saved = baseline_selected_blocks - policy_selected_blocks
latent_saved_pct = (1 - policy_read_ratio / max(baseline_read_ratio, 0.001)) * 100 if baseline_read_ratio > 0 else 0
print(f'  Latent tokens saved: {{latent_saved}} ({{latent_saved_pct:.1f}}% reduction)')
print('='*60)

# Save results
results = {{
    'model': MODEL_NAME,
    'task': TASK,
    'n_samples': N_SAMPLES,
    'baseline_correct': baseline_correct,
    'policy_correct': policy_correct,
    'baseline_acc': baseline_acc,
    'policy_acc': policy_acc,
    'baseline_total_blocks': baseline_total_blocks,
    'baseline_selected_blocks': baseline_selected_blocks,
    'baseline_read_ratio': baseline_read_ratio,
    'policy_total_blocks': policy_total_blocks,
    'policy_selected_blocks': policy_selected_blocks,
    'policy_read_ratio': policy_read_ratio,
    'latent_tokens_saved': latent_saved,
    'latent_reduction_pct': latent_saved_pct,
    'timestamp': datetime.now().isoformat(),
}}
result_file = f'logs/results_{{MODEL_NAME}}_{{TASK}}.json'
with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to {{result_file}}')
EOF

echo "Evaluation completed!"
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

# Dataset configurations
DATASETS = {
    'gsm8k': {
        'load_fn': 'load_gsm8k',
        'split': 'train',
        'n_samples': 200,
        'time': '02:00:00',
    },
    'arc_challenge': {
        'load_fn': 'load_arc_challenge',
        'split': 'train',
        'n_samples': 200,
        'time': '02:00:00',
    },
    'winogrande': {
        'load_fn': 'load_winogrande',
        'split': 'train',
        'n_samples': 200,
        'time': '02:00:00',
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
                task=task_key,
                cuda_devices=cuda_devices,
                load_fn=task_cfg['load_fn'],
                data_split=task_cfg['split'],
                n_samples=task_cfg['n_samples'],
                segment_layer_idx=model_cfg['segment_layer_idx'],
            )

            script_path = os.path.join(scripts_dir, f'eval_{job_name}.sh')
            with open(script_path, 'w') as f:
                f.write(script_content)

            print(f'Generated: {script_path}')

    print(f'\nTotal scripts generated: {len(MODELS) * len(DATASETS)}')

if __name__ == '__main__':
    main()
