# GSM8K 自定义 Reward 函数使用指南

## 📋 文件说明

- **`custom_gsm8k_reward.py`**: 自定义 reward 函数实现
- **`run_gsm8k_with_custom_reward.sh`**: 使用自定义 reward 的训练脚本示例
- **本文件**: 使用说明

## 🎯 Reward 函数评分规则

这个自定义 reward 函数使用简单的三级评分机制：

| 情况 | 描述 | Reward 分数 |
|------|------|-------------|
| ✅ 完全正确 | 答案与 ground_truth 完全匹配 | **1.0** |
| ⚠️ 格式正确但答案错误 | 能提取到 `#### answer` 格式，但答案不对 | **0.0** |
| ❌ 格式错误 | 无法提取答案（缺少 `####` 标记） | **0** |

## 🚀 快速开始

### 1. 测试 Reward 函数

首先，测试 reward 函数是否正常工作：

```bash
cd /p/scratch/westai0052/liu52/verl-agent
python test_script/custom_gsm8k_reward.py
```

你应该看到类似这样的输出：

```
==================================================
测试 GSM8K 自定义 Reward 函数
==================================================

测试1 - 完全正确:
Score: 1.0 (期望: 1.0)

测试2 - 格式正确但答案错误:
Score: 0.0 (期望: 0.0)

测试3 - 格式错误:
Score: 0 (期望: 0)
...
```

### 2. 在训练中使用自定义 Reward

#### 方法 A: 使用提供的脚本（推荐）

```bash
# 1. 修改脚本中的路径配置
vim test_script/run_gsm8k_with_custom_reward.sh

# 2. 更新这些变量:
#    - DATA_DIR: 你的 GSM8K 数据集路径
#    - MODEL_PATH: 你的模型路径

# 3. 运行训练
bash test_script/run_gsm8k_with_custom_reward.sh
```

#### 方法 B: 在现有脚本中添加配置

在你现有的训练脚本中，添加以下两行参数：

```bash
python3 -m verl.trainer.main_ppo \
    # ... 其他参数 ... \
    custom_reward_function.path="/p/scratch/westai0052/liu52/verl-agent/test_script/custom_gsm8k_reward.py" \
    custom_reward_function.name="compute_score" \
    # ... 其他参数 ...
```

#### 方法 C: 修改配置文件

如果你使用 YAML 配置文件，在 `config/ppo_trainer.yaml` 中修改：

```yaml
custom_reward_function:
  path: /p/scratch/westai0052/liu52/verl-agent/test_script/custom_gsm8k_reward.py
  name: compute_score  # 或 compute_score_flexible
```

## 🔧 两种提取模式

### 严格模式 (Strict Mode) - **推荐**

- **函数名**: `compute_score`
- **要求**: 模型输出必须包含 `#### answer` 格式
- **优点**: 强制模型学习标准格式，更好地对齐 GSM8K 数据集
- **示例**:
  ```
  Let me solve this step by step:
  7 + 13 = 20
  7/20 * 120 = 42
  #### 42  ← 必须有这个格式
  ```

### 灵活模式 (Flexible Mode)

- **函数名**: `compute_score_flexible`
- **要求**: 提取文本中最后一个数字作为答案
- **优点**: 对格式要求宽松，适用于早期训练阶段
- **示例**:
  ```
  The calculation shows the answer is 42.  ← 会提取 42
  ```

## 📝 自定义修改

如果你想调整评分规则，编辑 `custom_gsm8k_reward.py` 中的 `compute_score` 函数：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    answer = extract_answer(solution_str, method="strict")

    if answer is None:
        # 修改这里：格式错误的 reward
        return 0  # 可以改为 -0.1 给予惩罚
    else:
        if answer == ground_truth:
            # 修改这里：正确答案的 reward
            return 1.0  # 可以改为 2.0 给予更大奖励
        else:
            # 修改这里：格式正确但答案错误的 reward
            return 0.0  # 可以改为 0.1 鼓励格式
```

## 🔍 验证 Reward 是否生效

训练开始后，在日志中查找类似信息：

```
Loading custom reward function from: /path/to/custom_gsm8k_reward.py
Custom reward function loaded: compute_score
```

或者检查训练过程中的 reward 分布：

```python
# 在训练日志中应该能看到 reward 值为 0, 0.0, 或 1.0
Episode rewards: [1.0, 0.0, 0, 1.0, 0.0, ...]
```

## 📚 相关文件位置

```
verl-agent/
├── test_script/
│   ├── custom_gsm8k_reward.py              # ← 自定义 reward 函数
│   ├── run_gsm8k_with_custom_reward.sh     # ← 训练脚本示例
│   └── CUSTOM_REWARD_README.md             # ← 本文件
├── verl/utils/reward_score/
│   ├── __init__.py                          # default_compute_score
│   └── gsm8k.py                             # 原始 GSM8K reward 实现
└── verl/trainer/
    ├── main_ppo.py                          # PPO 训练主程序
    └── ppo/reward.py                        # Reward 管理器加载逻辑
```

## ❓ 常见问题

### Q1: 为什么我的自定义 reward 没有生效？

**A**: 检查以下几点：
1. 确认 `custom_reward_function.path` 路径正确
2. 确认函数签名正确：`def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
3. 查看训练日志中是否有加载自定义 reward 的提示

### Q2: 可以使用相对路径吗？

**A**: 可以，但推荐使用绝对路径避免路径问题：

```bash
# 相对路径（相对于执行目录）
custom_reward_function.path="test_script/custom_gsm8k_reward.py"

# 绝对路径（推荐）
custom_reward_function.path="/p/scratch/westai0052/liu52/verl-agent/test_script/custom_gsm8k_reward.py"
```

### Q3: 如何添加中间步骤奖励？

**A**: 修改 `compute_score` 函数，例如：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    answer = extract_answer(solution_str, method="strict")

    # 基础分数
    if answer is None:
        base_score = 0
    elif answer == ground_truth:
        base_score = 1.0
    else:
        base_score = 0.0

    # 额外奖励：如果包含推理步骤
    if "step by step" in solution_str.lower():
        base_score += 0.1

    return base_score
```

### Q4: 如何切换回默认 reward 函数？

**A**: 删除或注释掉 `custom_reward_function.path` 参数：

```bash
# 使用默认 reward
python3 -m verl.trainer.main_ppo \
    # custom_reward_function.path="..."  # 注释掉这一行
    # ... 其他参数 ...
```

## 📊 预期效果

使用这个简单的三级评分 reward 函数，你应该看到：

1. **训练初期**:
   - 大量 reward = 0 (格式错误)
   - 少量 reward = 0.0 (格式正确但答案错误)

2. **训练中期**:
   - Reward = 0 逐渐减少
   - Reward = 0.0 增加（模型学会了格式）

3. **训练后期**:
   - Reward = 1.0 逐渐增加（模型开始答对题目）

## 🔗 参考资料

- [VeRL 官方文档 - Reward 函数](../docs/preparation/reward_function.rst)
- [GSM8K 论文](https://arxiv.org/pdf/2110.14168)
- [VeRL GitHub](https://github.com/volcengine/verl)
