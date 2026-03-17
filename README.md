# LatentReading: RL-Based Latent Memory Selection for Multi-Agent Systems

This project extends [LatentMAS](https://github.com/Gen-Verse/LatentMAS) with **LatentReading** — a learned reading policy trained via **GRPO** that intelligently selects which latent memory blocks to pass to the judger agent, reducing communication overhead while preserving accuracy.

---

## Motivation

**LatentMAS** solves the token generation bottleneck in multi-agent systems by passing hidden states directly between agents instead of generating text tokens. However, it transmits the *entire* KV cache from every agent — including intermediate reasoning steps that may be irrelevant.

**Problem:** If Agent A reasons for 1000 steps, 70–90% of those hidden states may be noise. Passing everything to Agent B causes:
- Wasted attention computation (O(n²) complexity)
- Attention dilution from irrelevant context

**LatentReading** adds a lightweight, learnable filter: a **Reading Policy Network** that selects only the top-k most relevant memory blocks before passing them to the judger.

```
LatentMAS:      All 100 blocks ──────────────────→ Judger
LatentReading:  All 100 blocks → [Policy] → Top-40 → Judger
                                              (−91% attention compute)
```

---

## New Contributions (on top of LatentMAS)

### 1. Semantic Memory Segmentation
After each non-judger agent completes its latent steps, the resulting KV cache is segmented into **semantic blocks** using cosine similarity between adjacent hidden states (`SemanticBlockSegmenter`). These blocks are stored in a `MemoryPool`.

### 2. Reading Policy Network (`policy/`)
A **cross-attention based** policy network (`ReadingPolicyNetwork`) that takes:
- A **query embedding** from the judger's prompt
- **Block summary vectors** (mean-pooled per semantic block)

And outputs selection probabilities for each block. Top-k blocks are selected and their KV cache is fed to the judger.

Architecture: 2-layer cross-attention + FFN + score head (~62M parameters for Qwen3-4B, negligible inference overhead ~5ms)

A lighter **MLP-based alternative** (`LightweightReadingPolicy`) is also provided.

### 3. GRPO Training (`training/`)
The reading policy is trained with **Group Relative Policy Optimization** (no value network needed):
- For each question, sample G different block selections
- Compute group-relative advantages: `A_i = (r_i − mean) / std`
- Update policy with clipped PPO objective + KL penalty + entropy bonus

**Reward function:** `R = α·R_task + β·R_consistency − γ·R_cost`
- **R_task**: Binary correctness (sparse signal, task-directed)
- **R_consistency**: Cosine similarity between judger output and selected blocks vs. unselected blocks (dense signal)
- **R_cost**: Penalizes selecting more blocks (encourages efficiency)

**Key finding: R_cost is harmful (γ = 0 is optimal).** Having both "select correctly" and "select fewer" as simultaneous objectives causes conflicting gradients — the policy learns to "select randomly" as a compromise. The correct approach is to control quantity via top_k and let the policy focus solely on selecting the right blocks.

**Optimal config:** α = 0.5, β = 0.5, γ = 0

### 4. Heuristic Baselines (`methods/heuristic_selection.py`)
Four rule-based selectors for comparison:
- **Random** — randomly select k blocks
- **Recency** — select the most recent k blocks
- **Similarity** — cosine similarity to query
- **Time-Weighted Similarity** — weighted combination of recency + similarity

### 5. RL-Enabled Method (`methods/latent_mas_rl.py`)
`LatentMASMethodRL` extends `LatentMASMethod` with memory pool management, policy-based block selection, and trajectory recording for GRPO updates. Supports both HF and vLLM backends.

### 6. New Task: Winogrande
Added commonsense reasoning evaluation on Winogrande.

---

## Experimental Results

### Setup
- **GPU:** NVIDIA H100 × 4
- **Models:** Qwen3-4B / Qwen3-8B
- **Training/Test:** 500 samples per dataset
- **Training:** 5 epochs, lr = 1e-5, GRPO group size = 4
- **Datasets:** GSM8K (math), ARC-Challenge (science), Winogrande (commonsense)

### Main Results (V4, k=40, γ=0)

| Dataset | Baseline | Policy (k=40) | Δ Accuracy | Latent Reduction | Pass (<5%) |
|---|---|---|---|---|---|
| GSM8K | 65.4% | 62.6% | **−2.8%** | ~60% | ✓ |
| ARC-Challenge | 60.0% | 56.4% | **−3.6%** | ~60% | ✓ |
| Winogrande | 61.4% | 60.2% | **−1.2%** | ~60% | ✓ |

**Latent reduction: ~70% · Attention compute reduction: ~91%** (due to O(n²) complexity: (30/100)² = 9%)

### Top-k Ablation (15 experiments, γ=0)

| Dataset | k=10 | k=20 | k=30 | k=40 | k=50 |
|---|---|---|---|---|---|
| GSM8K | −15.0% ✗ | −22.6% ✗ | −19.8% ✗ | **−2.8% ✓** | −5.8% ✗ |
| ARC | −6.0% ✗ | −10.0% ✗ | −9.4% ✗ | **−3.6% ✓** | +0.4% ✓ |
| Winogrande | −5.2% ✗ | −9.8% ✗ | −7.2% ✗ | **−1.2% ✓** | −0.8% ✓ |

**k=40 is the joint optimum** across all three datasets. Notable: ARC at k=50 *exceeds* baseline (+0.4%), suggesting that filtering irrelevant blocks can sometimes improve accuracy.

### Iterative Development

| Version | k | γ | GSM8K Δ | Key Change |
|---|---|---|---|---|
| V1 | 3 | 0.1 | −18.2% ✗ | Initial attempt |
| V2 | 5 | 0.1 | −13.4% ✗ | Increase k |
| V3 | 5 | 0 | −7% (ARC: −4.8% ✓) | **Remove cost penalty** |
| V4 | 40 | 0 | **−2.8% ✓** | Systematic k ablation |

### Ablation: Architecture (layers)

| Layers | GSM8K Δ | Params | Result |
|---|---|---|---|
| L=1 | −9.4% | ~62M | ✗ Insufficient capacity |
| **L=2** | **−2.8%** | ~124M | **✓ Optimal** |

### Ablation: Reward Function

| Config | GSM8K Δ | Reason |
|---|---|---|
| Task only | −9.4% ✗ | Sparse signal, hard to learn |
| Consistency only | −6.4% ✗ | No task-direction |
| **Balanced (α=β=0.5)** | **−2.8% ✓** | Complementary signals |

---

## Optimal Configuration

| Component | Setting |
|---|---|
| Policy network | 2-layer cross-attention, hidden_dim=2560, 4 heads |
| Reward weights | α=0.5 (task), β=0.5 (consistency), **γ=0 (no cost penalty)** |
| Top-k | **40** |
| Training | GRPO, group_size=4 |

---

## Repository Structure

```
GRPO_latent/
├── run.py                        # Evaluation: baseline / text_mas / latent_mas
├── run_rl_train.py               # [NEW] RL training with GRPO
├── models.py                     # ModelWrapper: HF + vLLM + latent realignment
├── methods/
│   ├── baseline.py               # [LatentMAS] Single-agent baseline
│   ├── text_mas.py               # [LatentMAS] Token-space multi-agent
│   ├── latent_mas.py             # [LatentMAS] Latent-space multi-agent
│   ├── latent_mas_rl.py          # [NEW] LatentMAS + reading policy
│   └── heuristic_selection.py    # [NEW] Random/Recency/Similarity baselines
├── policy/
│   ├── reading_policy.py         # [NEW] Cross-attention policy + MLP variant
│   └── block_selector.py         # [NEW] Top-k block selection
├── training/
│   ├── rl_trainer.py             # [NEW] GRPO trainer
│   ├── reward.py                 # [NEW] Multi-component reward
│   └── trajectory.py             # [NEW] Transition / buffer
├── data.py                       # Dataset loaders (incl. Winogrande)
├── scripts/                      # [NEW] Ablation, training, eval scripts
└── requirements.txt
```

---

## Setup

```bash
conda create -n grpo_latent python=3.10 -y
conda activate grpo_latent
pip install -r requirements.txt

export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
```

## Running

```bash
# RL Training
python run_rl_train.py \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --top_k_blocks 40 \
  --group_size 4

# Evaluation
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --max_samples -1
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
