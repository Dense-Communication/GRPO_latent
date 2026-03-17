# GRPO Latent Reading

**GRPO Latent Reading** extends [LatentMAS](https://github.com/Gen-Verse/LatentMAS) with a learned **reading policy** trained via **GRPO** (Group Relative Policy Optimization). Instead of applying a fixed number of latent steps to every token position, the policy network dynamically selects which transformer blocks to apply latent reasoning — improving both accuracy and efficiency.

---

## Overview

This repository builds on top of LatentMAS and adds:

- **Reading Policy Network** (`policy/`) — a lightweight network that decides when and where to apply latent steps
- **GRPO Trainer** (`training/`) — trains the policy using group relative policy optimization with task reward signals
- **Heuristic Baselines** (`methods/heuristic_selection.py`) — rule-based block selection baselines for comparison
- **RL Training Script** (`run_rl_train.py`) — end-to-end GRPO training loop

---

## Repository Structure

```
GRPO_latent/
├── run.py                        # Evaluation entry point (baseline / text_mas / latent_mas)
├── run_rl_train.py               # RL training entry point (GRPO policy training)
├── models.py                     # ModelWrapper: HF + vLLM + latent realignment
├── methods/
│   ├── baseline.py               # Single-agent baseline
│   ├── text_mas.py               # Token-space multi-agent method
│   ├── latent_mas.py             # Latent-space multi-agent (LatentMAS)
│   ├── latent_mas_rl.py          # LatentMAS with RL-based reading policy
│   └── heuristic_selection.py    # Heuristic block selection baselines
├── policy/
│   ├── reading_policy.py         # Reading policy network
│   └── block_selector.py         # Block-level selection logic
├── training/
│   ├── rl_trainer.py             # GRPO trainer
│   ├── reward.py                 # Reward calculation
│   └── trajectory.py             # Transition / trajectory definitions
├── prompts.py                    # Prompt constructors
├── data.py                       # Dataset loaders
├── utils.py                      # Helpers: answer parsing, device setup, etc.
├── scripts/                      # Shell scripts for running experiments
├── data/                         # Dataset files
├── example_logs/                 # Example output logs
└── requirements.txt
```

---

## Setup

```bash
conda create -n grpo_latent python=3.10 -y
conda activate grpo_latent
pip install -r requirements.txt
```

Set HuggingFace cache directory:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

---

## Quick Start

### Evaluation (standard LatentMAS)

```bash
# Baseline
python run.py --method baseline --model_name Qwen/Qwen3-8B --task gsm8k --max_samples -1 --max_new_tokens 2048

# TextMAS
python run.py --method text_mas --model_name Qwen/Qwen3-8B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048

# LatentMAS
python run.py --method latent_mas --model_name Qwen/Qwen3-8B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
```

### RL Training (GRPO Reading Policy)

```bash
python run_rl_train.py \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --max_samples 1000 \
  --max_new_tokens 2048
```

See `scripts/run_rl_train.sh` for a full example with all arguments.

---

## Supported Tasks

`gsm8k`, `aime2024`, `aime2025`, `gpqa`, `arc_easy`, `arc_challenge`, `mbppplus`, `humanevalplus`, `medqa`, `winogrande`

---

## Based On

This work is built on top of [LatentMAS](https://github.com/Gen-Verse/LatentMAS):

```
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```
