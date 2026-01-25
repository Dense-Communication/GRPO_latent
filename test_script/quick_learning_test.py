#!/usr/bin/env python3
"""
快速学习能力测试 - 5分钟内验证训练是否能正常工作

这个脚本不运行完整的 GRPO 训练，而是：
1. 加载模型和数据
2. 让模型生成几个回答
3. 用新的奖励函数评分
4. 验证是否能产生非零学习信号
"""

import os
import sys

# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
sys.path.insert(0, '/p/scratch/westai0052/liu52/verl-agent')
sys.path.insert(0, '/p/scratch/westai0052/liu52/verl-agent/test_script')

from custom_gsm8k_reward import compute_score, extract_answer

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    print_header("快速学习能力测试")
    print("目标: 验证新的奖励函数能否为模型提供学习信号")

    model_path = "/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct"
    data_path = "/p/scratch/westai0052/liu52/verl-agent/test_script/data/train.parquet"

    # 加载数据
    print_header("1. 加载数据")
    df = pd.read_parquet(data_path)
    print(f"加载了 {len(df)} 条数据")

    # 加载模型
    print_header("2. 加载模型")
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("加载模型... (约30秒)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("模型加载完成!")

    # 测试多个样本
    print_header("3. 生成回答并评分")

    num_samples = 5  # 测试5个样本
    scores = []
    score_distribution = {"1.0": 0, "0.8": 0, "0.1": 0, "0.05": 0, "0.0": 0}

    for i in range(num_samples):
        sample = df.iloc[i]
        raw_prompt = sample['prompt']
        ground_truth = sample['reward_model']['ground_truth']

        # 处理 prompt
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()

        # 应用 chat template
        chat_text = tokenizer.apply_chat_template(
            raw_prompt,
            add_generation_prompt=True,
            tokenize=False
        )

        # 生成回答
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # 评分
        score = compute_score("openai/gsm8k", response, ground_truth)
        scores.append(score)

        # 统计分数分布
        score_key = str(score) if score in [1.0, 0.8, 0.1, 0.05, 0.0] else "other"
        if score_key in score_distribution:
            score_distribution[score_key] += 1

        # 打印详情
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {raw_prompt[0]['content'][:80]}...")
        print(f"正确答案: {ground_truth}")
        print(f"模型回答: {response[:150]}...")

        # 提取答案
        strict_ans = extract_answer(response, "strict")
        flex_ans = extract_answer(response, "flexible")
        print(f"严格提取: {strict_ans}, 灵活提取: {flex_ans}")
        print(f"得分: {score}")

    # 总结
    print_header("4. 测试结果总结")

    print(f"\n测试样本数: {num_samples}")
    print(f"平均分数: {np.mean(scores):.3f}")
    print(f"最高分数: {max(scores)}")
    print(f"最低分数: {min(scores)}")

    print(f"\n分数分布:")
    for score_val, count in score_distribution.items():
        bar = "█" * (count * 4)
        print(f"  {score_val}: {count} {bar}")

    # 判断是否有学习信号
    print_header("5. 学习信号诊断")

    non_zero_count = sum(1 for s in scores if s > 0)
    has_learning_signal = non_zero_count > 0

    if has_learning_signal:
        print(f"✅ 检测到学习信号！{non_zero_count}/{num_samples} 个样本获得非零分数")
        print("\n新的奖励函数正在工作，训练应该能够正常进行。")

        if max(scores) >= 0.8:
            print("✅ 模型已经能产生正确答案，训练效果应该不错")
        elif max(scores) >= 0.1:
            print("⚠️ 模型还没有正确答案，但有使用正确格式的尝试")
        else:
            print("⚠️ 模型只获得了微弱的学习信号，训练可能需要更多时间")
    else:
        print("❌ 没有检测到学习信号！所有样本得分都是 0")
        print("\n可能的原因:")
        print("  1. 模型输出完全没有数字")
        print("  2. Prompt 传递仍有问题")
        print("  3. 需要检查 rollout_loop.py 的修复是否生效")

    print_header("测试完成")

    return has_learning_signal

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
