# GRPO Latent Reading

This project extends [LatentMAS](https://github.com/Gen-Verse/LatentMAS) with a learned **reading policy** that dynamically selects which latent memory blocks to read, trained via **GRPO** (Group Relative Policy Optimization).

---

## Motivation

In the original LatentMAS, every agent's latent KV cache is fully passed to the judger, regardless of relevance. This project investigates: *can we train a lightweight policy to select only the most relevant latent blocks, reducing computational cost while maintaining accuracy?*

---

## New Contributions (on top of LatentMAS)

### 1. Semantic Memory Segmentation

After each non-judger agent completes its latent reasoning steps, the resulting KV cache is segmented into **semantic blocks** using cosine similarity between adjacent hidden states (`SemanticBlockSegmenter`). These blocks are stored in a **MemoryPool** for later retrieval.

### 2. Reading Policy Network (`policy/`)

A **cross-attention based policy network** (`ReadingPolicyNetwork`) that takes:
- A **query embedding** (from the judger's prompt)
- **Block summary vectors** (mean-pooled embeddings of each semantic block)

And outputs a selection probability for each block. The top-k blocks are selected and their KV cache (or embeddings in vLLM mode) is fed to the judger.

A lighter **MLP-based alternative** (`LightweightReadingPolicy`) is also provided for faster inference.

### 3. Heuristic Baselines (`methods/heuristic_selection.py`)

Four rule-based block selection methods for comparison with the learned policy:
- **Random** — randomly select k blocks
- **Recency** — select the most recent k blocks
- **Similarity** — select the k blocks with highest cosine similarity to the query
- **Time-Weighted Similarity** — weighted combination of recency and cosine similarity

### 4. GRPO Training (`training/`)

The reading policy is trained using **Group Relative Policy Optimization**:
- **No value network** required (unlike PPO)
- Advantages are computed group-relatively: `A_i = (r_i - mean(group)) / std(group)`
- PPO-style clipped objective + KL divergence penalty (with frozen reference policy) + entropy bonus

**Multi-component reward** `R = α·R1 + β·R2 - γ·R3`:
- **R1** (task correctness): binary reward based on answer correctness
- **R2** (evidence consistency): cosine similarity between judger output and selected blocks, minus similarity with unselected blocks — rewards selecting truly relevant blocks
- **R3** (read cost): penalizes selecting more blocks than needed (block ratio + KV token ratio)

An **AdaptiveRewardCalculator** is also provided, which gradually shifts emphasis from correctness (early training) to efficiency (late training).

### 5. RL-Enabled Method (`methods/latent_mas_rl.py`)

`LatentMASMethodRL` extends `LatentMASMethod` with:
- Memory pool management per batch
- Policy-based block selection at the judger stage
- Trajectory recording (query embedding, block summaries, selected indices, log probs) for GRPO updates
- Efficiency statistics (total blocks, selected blocks, selection ratio)
- Support for both HF and vLLM backends

### 6. New Task Support

Added **Winogrande** as a new evaluation task (answer extraction + evaluation), extending the original 9-task benchmark.

### 7. Ablation & Evaluation Scripts (`scripts/`)

Comprehensive shell scripts covering:
- Top-k ablation (k = 5, 10, 20, 30, 40, 50) across tasks and model sizes
- Architecture ablation (number of policy layers, attention heads)
- Reward component ablation (task-only, consistency-only, with/without cost)
- Model size ablation (Qwen3-1.5B, 4B, 8B, 14B)
- Parallel job submission for cluster environments

---

## Repository Structure

```
GRPO_latent/
├── run.py                        # Evaluation: baseline / text_mas / latent_mas
├── run_rl_train.py               # RL training entry point (GRPO)
├── models.py                     # ModelWrapper: HF + vLLM + latent realignment
├── methods/
│   ├── baseline.py               # [LatentMAS] Single-agent baseline
│   ├── text_mas.py               # [LatentMAS] Token-space multi-agent
│   ├── latent_mas.py             # [LatentMAS] Latent-space multi-agent
│   ├── latent_mas_rl.py          # [NEW] LatentMAS + reading policy + trajectory recording
│   └── heuristic_selection.py    # [NEW] Heuristic baselines (random/recency/similarity)
├── policy/
│   ├── reading_policy.py         # [NEW] Cross-attention policy + lightweight MLP policy
│   └── block_selector.py         # [NEW] Top-k block selection with stochastic sampling
├── training/
│   ├── rl_trainer.py             # [NEW] GRPO trainer + OnlineGRPOTrainer
│   ├── reward.py                 # [NEW] Multi-component reward (R1+R2-R3) + AdaptiveReward
│   └── trajectory.py             # [NEW] Transition / TrajectoryBuffer
├── prompts.py                    # Prompt constructors
├── data.py                       # Dataset loaders (incl. Winogrande)
├── utils.py                      # Helpers
├── scripts/                      # [NEW] Ablation, training, eval scripts
└── requirements.txt
```

---

## Setup

```bash
conda create -n grpo_latent python=3.10 -y
conda activate grpo_latent
pip install -r requirements.txt
```

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

---

## Running Experiments

### Evaluate LatentMAS (unchanged from original)

```bash
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --max_samples -1 \
  --max_new_tokens 2048
```

### Train Reading Policy with GRPO

```bash
python run_rl_train.py \
  --model_name Qwen/Qwen3-8B \
  --task gsm8k \
  --max_samples 1000 \
  --max_new_tokens 2048 \
  --top_k_blocks 5 \
  --group_size 8
```

See `scripts/run_rl_train.sh` for full argument reference.

### Run Top-k Ablation

```bash
bash scripts/submit_topk_ablation.sh
```

---

## Supported Tasks

`gsm8k`, `aime2024`, `aime2025`, `gpqa`, `arc_easy`, `arc_challenge`, `mbppplus`, `humanevalplus`, `medqa`, `winogrande`

---

## Based On

```
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```
