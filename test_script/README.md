# GSM8K 测试脚本目录

本目录包含用于在 JURECA 集群上训练和测试 GSM8K 数学推理任务的脚本。

## 目录结构

```
test_script/
├── data/                          # 数据目录
│   ├── train.parquet              # 训练集 (7473 样本)
│   ├── test.parquet               # 完整验证集 (1319 样本)
│   └── val_subset.parquet         # 固定验证子集 (100 样本)
├── checkpoints/                   # 检查点目录
├── outputs/                       # 训练输出目录
├── multi_agent/                   # 多智能体相关脚本
├── single_agent_gsm8k/            # 单智能体 GSM8K 脚本 (历史版本)
│
├── # === 训练脚本 ===
├── gsm8k_full_train.sh            # 完整训练脚本 (200 步)
├── gsm8k_long_train.sh            # 长时间训练脚本 (500 步, 更频繁验证)
├── minimal_train_test.sh          # 最小化训练测试
├── mini_train_test.sh             # 迷你训练测试
│
├── # === SLURM 提交脚本 ===
├── submit_gsm8k_full.slurm        # 提交完整训练任务
├── submit_long_train.slurm        # 提交长时间训练任务
├── submit_minimal_test.slurm      # 提交最小化测试任务
│
├── # === 分析工具 ===
├── analyze_training_metrics.py    # 训练指标分析脚本
├── create_fixed_val_subset.py     # 创建固定验证子集
│
├── # === 测试脚本 ===
├── quick_sanity_check.py          # 快速健全性检查 (5分钟)
├── quick_learning_test.py         # 快速学习能力测试
├── test_new_reward.py             # 测试奖励函数
├── test_prompt_passing.py         # 测试 prompt 传递
│
├── # === 核心模块 ===
├── custom_gsm8k_reward.py         # 自定义 GSM8K 奖励函数
├── gsm8k.py                       # GSM8K 数据处理
├── prepare_gsm8k_data.py          # 准备 GSM8K 数据
└── minimal_train_test.py          # 最小化训练测试模块
```

---

## 快速开始

### 1. 提交完整训练任务

```bash
cd /p/scratch/westai0052/liu52/verl-agent/test_script
sbatch submit_gsm8k_full.slurm
```

### 2. 提交长时间训练任务 (带趋势分析)

```bash
# 使用默认配置 (500 步, 每 5 步验证)
sbatch submit_long_train.slurm

# 或自定义参数
TOTAL_STEPS=1000 TEST_FREQ=10 sbatch submit_long_train.slurm
```

### 3. 查看任务状态

```bash
squeue -u $USER
```

---

## 训练脚本详解

### gsm8k_full_train.sh

标准 GRPO 训练脚本，用于 GSM8K 数学推理任务。

**配置:**
- 总步数: 200
- 验证频率: 每 10 步
- 保存频率: 每 50 步
- 学习率: 1e-6
- Batch size: 64

**使用:**
```bash
bash gsm8k_full_train.sh
```

### gsm8k_long_train.sh

长时间训练脚本，用于观察训练趋势和参数稳定性。

**特点:**
- 总步数: 500 (可配置)
- 验证频率: 每 5 步 (更密集)
- 支持使用固定验证子集加速验证
- 训练结束后自动运行分析脚本

**环境变量:**
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TOTAL_STEPS` | 500 | 总训练步数 |
| `TEST_FREQ` | 5 | 验证频率 |
| `SAVE_FREQ` | 100 | 检查点保存频率 |
| `USE_VAL_SUBSET` | true | 是否使用验证子集 |
| `LR` | 1e-6 | 学习率 |
| `KL_COEF` | 0.05 | KL 惩罚系数 |

**使用:**
```bash
# 直接运行
bash gsm8k_long_train.sh

# 自定义参数
TOTAL_STEPS=1000 TEST_FREQ=10 bash gsm8k_long_train.sh
```

---

## 分析工具

### analyze_training_metrics.py

分析训练日志，提取关键指标并可视化。

**功能:**
- 提取验证集准确率 (test_score) 和训练 reward
- 计算 reward 与 accuracy 的 Pearson 相关系数
- 分析训练稳定性 (梯度范数、熵损失)
- 生成可视化图表

**使用:**
```bash
# 激活环境
source /p/home/jusers/liu52/jureca/miniforge3/etc/profile.d/conda.sh
conda activate verl-agent

# 运行分析
python analyze_training_metrics.py <日志路径> --plot

# 示例
python analyze_training_metrics.py ../outputs/gsm8k_grpo_20260124_064518/training.log --plot
```

**输出示例:**
```
============================================================
训练指标分析摘要
============================================================

验证集准确率 (test_score):
  初始值: 0.6370
  最终值: 0.7610
  最大值: 0.7660 (step 170)
  提升幅度: 19.47%

Reward 与 Accuracy 相关性分析:
  Pearson 相关系数: 0.7416
  解读: 强正相关 - reward 上升时 accuracy 也倾向于上升
```

### create_fixed_val_subset.py

创建固定的验证子集，用于训练过程中的频繁验证。

**功能:**
- 从完整验证集随机抽取固定数量样本
- 使用固定随机种子确保可复现
- 支持分层抽样

**使用:**
```bash
# 创建 100 样本的子集 (默认)
python create_fixed_val_subset.py

# 自定义大小
python create_fixed_val_subset.py --size 200 --seed 42

# 分层抽样
python create_fixed_val_subset.py --size 100 --stratify
```

**参数:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `data/test.parquet` | 原始验证集路径 |
| `--output` | `data/val_subset.parquet` | 子集输出路径 |
| `--size` | 100 | 子集大小 |
| `--seed` | 42 | 随机种子 |
| `--stratify` | False | 是否分层抽样 |

---

## 测试脚本

### quick_sanity_check.py

快速健全性检查，5 分钟内验证训练管道是否正常。

**检查项目:**
1. 数据加载是否正确
2. 模型能否正常生成
3. Reward 函数是否能给出正确分数
4. Prompt 是否正确传递

**使用:**
```bash
python quick_sanity_check.py
```

### quick_learning_test.py

快速学习能力测试，验证训练是否能正常工作。

**功能:**
- 加载模型和数据
- 让模型生成回答
- 用奖励函数评分
- 验证是否能产生非零学习信号

**使用:**
```bash
python quick_learning_test.py
```

### test_new_reward.py

测试 GSM8K 评分函数的正确性。

**使用:**
```bash
python test_new_reward.py
```

### test_prompt_passing.py

测试 prompt 能否正确传递到模型。

**使用:**
```bash
python test_prompt_passing.py
```

---

## 核心模块

### custom_gsm8k_reward.py

自定义 GSM8K 奖励函数（渐进式奖励机制）。

**评分规则:**
| 情况 | 分数 | 说明 |
|------|------|------|
| 完全正确 + 标准格式 (#### answer) | 1.0 | 最佳情况 |
| 完全正确 + 灵活格式 | 0.8 | 正确但格式不标准 |
| 格式正确 (有 ####) 但答案错误 | 0.1 | 鼓励正确格式 |
| 有数字输出但答案错误 | 0.05 | 有输出比无输出好 |
| 没有数字输出 | 0.0 | 最差情况 |

### prepare_gsm8k_data.py

准备 GSM8K 数据，添加 ground_truth 字段。

**使用:**
```bash
python prepare_gsm8k_data.py
```

---

## 数据说明

### train.parquet
- 大小: 7473 样本
- 用途: 训练数据

### test.parquet
- 大小: 1319 样本
- 用途: 完整验证集

### val_subset.parquet
- 大小: 100 样本
- 用途: 固定验证子集，用于频繁验证
- 随机种子: 42

---

## 常用命令

```bash
# 激活环境
source /p/home/jusers/liu52/jureca/miniforge3/etc/profile.d/conda.sh
conda activate verl-agent

# 提交任务
sbatch submit_gsm8k_full.slurm
sbatch submit_long_train.slurm

# 查看任务
squeue -u $USER

# 取消任务
scancel <job_id>

# 查看日志
tail -f ../outputs/<experiment_name>/training.log

# 分析结果
python analyze_training_metrics.py ../outputs/<experiment_name>/training.log --plot

# 提取 test_score
grep -oP "step:\d+ .*?val/openai/gsm8k/test_score:\d+\.\d+" training.log | \
    sed 's/.*step:\([0-9]*\).*test_score:\([0-9.]*\).*/step:\1 test_score:\2/'
```

---

## 注意事项

1. **离线模式**: 计算节点无网络，所有脚本已配置离线模式
2. **环境变量**: 训练脚本会自动设置 `HF_OFFLINE`, `TRANSFORMERS_OFFLINE` 等
3. **Ray 清理**: 脚本会自动清理之前的 Ray session
4. **GPU 配置**: 默认使用 4 GPU，可在 SLURM 脚本中调整

---

## 最近训练结果 (参考)

从 200 步训练的结果：
- **初始准确率**: 63.7%
- **最终准确率**: 76.1%
- **最高准确率**: 76.6% (step 170)
- **提升幅度**: +19.47%
- **Reward-Accuracy 相关系数**: 0.74 (强正相关)

训练是有效的，reward 上升时 accuracy 也同步上升。
