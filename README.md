# GSM8K 实验 (基于 verl-agent)

> 本仓库基于 [verl-agent](https://github.com/langfengQ/verl-agent) 框架，针对 GSM8K 数学推理任务在 JURECA 集群上进行了扩展。

## 修改概述

1. **自定义奖励函数** - 渐进式评分机制，提供部分分数以帮助模型学习
2. **训练脚本** - 适配 JURECA 集群的离线训练脚本
3. **分析工具** - 训练指标分析和可视化
4. **固定验证子集** - 用于训练过程中的频繁验证

## 目录结构

```
test_script/
├── data/                          # 数据目录
│   ├── train.parquet              # 训练集 (7473 样本)
│   ├── test.parquet               # 完整验证集 (1319 样本)
│   └── val_subset.parquet         # 固定验证子集 (100 样本)
│
├── # === 训练脚本 ===
├── gsm8k_full_train.sh            # 完整训练脚本 (200 步)
├── gsm8k_long_train.sh            # 长时间训练脚本 (500 步, 更频繁验证)
├── minimal_train_test.sh          # 最小化训练测试
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
└── prepare_gsm8k_data.py          # 准备 GSM8K 数据
```

## 快速开始

### 1. 环境准备

```bash
source /p/home/jusers/liu52/jureca/miniforge3/etc/profile.d/conda.sh
conda activate verl-agent
```

### 2. 提交训练任务

```bash
cd /p/scratch/westai0052/liu52/verl-agent/test_script

# 完整训练 (200 步)
sbatch submit_gsm8k_full.slurm

# 长时间训练 (500 步, 更频繁验证)
sbatch submit_long_train.slurm

# 自定义参数
TOTAL_STEPS=1000 TEST_FREQ=10 sbatch submit_long_train.slurm
```

### 3. 查看任务状态

```bash
squeue -u $USER
```

## 训练脚本详解

### gsm8k_full_train.sh

标准 GRPO 训练脚本。

| 参数 | 值 |
|------|-----|
| 总步数 | 200 |
| 验证频率 | 每 10 步 |
| 保存频率 | 每 50 步 |
| 学习率 | 1e-6 |
| Batch size | 64 |

### gsm8k_long_train.sh

长时间训练脚本，用于观察训练趋势。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `TOTAL_STEPS` | 500 | 总训练步数 |
| `TEST_FREQ` | 5 | 验证频率 |
| `SAVE_FREQ` | 100 | 保存频率 |
| `USE_VAL_SUBSET` | true | 使用验证子集 |
| `LR` | 1e-6 | 学习率 |

## 分析工具

### analyze_training_metrics.py

分析训练日志，提取 reward 和 accuracy 趋势。

```bash
python analyze_training_metrics.py <日志路径> --plot
```

**功能：**
- 提取验证集准确率和训练 reward
- 计算 Pearson 相关系数
- 分析训练稳定性
- 生成可视化图表

**输出示例：**
```
============================================================
训练指标分析摘要
============================================================

验证集准确率 (test_score):
  初始值: 0.6370
  最终值: 0.7610
  提升幅度: 19.47%

Reward 与 Accuracy 相关性分析:
  Pearson 相关系数: 0.7416
  解读: 强正相关 - reward 上升时 accuracy 也倾向于上升
```

### create_fixed_val_subset.py

创建固定验证子集，加速训练过程中的验证。

```bash
python create_fixed_val_subset.py --size 100 --seed 42
```

## 自定义奖励函数

`custom_gsm8k_reward.py` 实现了渐进式奖励机制：

| 情况 | 分数 | 说明 |
|------|------|------|
| 完全正确 + 标准格式 (`#### answer`) | 1.0 | 最佳 |
| 完全正确 + 灵活格式 | 0.8 | 正确但格式不标准 |
| 格式正确但答案错误 | 0.1 | 鼓励正确格式 |
| 有数字输出但错误 | 0.05 | 有输出比没有好 |
| 没有数字输出 | 0.0 | 最差 |

## 测试脚本

| 脚本 | 用途 |
|------|------|
| `quick_sanity_check.py` | 快速验证训练管道 (5分钟) |
| `quick_learning_test.py` | 验证学习信号是否正常 |
| `test_new_reward.py` | 测试奖励函数正确性 |
| `test_prompt_passing.py` | 测试 prompt 传递 |

## 常用命令

```bash
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

## 实验结果 (参考)

200 步训练结果：

| 指标 | 值 |
|------|-----|
| 初始准确率 | 63.7% |
| 最终准确率 | 76.1% |
| 最高准确率 | 76.6% (step 170) |
| 提升幅度 | +19.47% |
| Reward-Accuracy 相关系数 | 0.74 |

## 注意事项

1. **离线模式**: JURECA 计算节点无网络，脚本已配置离线模式
2. **环境变量**: 自动设置 `HF_OFFLINE`, `TRANSFORMERS_OFFLINE` 等
3. **GPU 配置**: 默认使用 4 GPU
