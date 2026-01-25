#!/usr/bin/env python3
"""
快速健全性检查脚本 - 5分钟内验证训练管道是否正常

检查项目：
1. 数据加载是否正确
2. 模型能否正常生成
3. Reward 函数是否能给出正确分数
4. Prompt 是否正确传递
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

def print_result(name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {name}")
    if details:
        print(f"       {details}")

def main():
    print_header("GSM8K GRPO 训练管道健全性检查")

    model_path = "/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct"
    data_path = "/p/scratch/westai0052/liu52/verl-agent/test_script/data/train.parquet"

    all_passed = True

    # ============ 检查 1: 数据加载 ============
    print_header("1. 数据加载检查")
    try:
        df = pd.read_parquet(data_path)
        print_result("读取 parquet 文件", True, f"共 {len(df)} 条数据")

        # 检查必需的列
        required_cols = ['data_source', 'prompt', 'reward_model']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print_result("必需列存在", False, f"缺少: {missing}")
            all_passed = False
        else:
            print_result("必需列存在", True)

        # 检查 prompt 格式
        sample_prompt = df.iloc[0]['prompt']
        is_numpy = isinstance(sample_prompt, np.ndarray)
        has_content = len(sample_prompt) > 0 and 'content' in sample_prompt[0]
        print_result("Prompt 格式正确", has_content,
                    f"类型: {type(sample_prompt).__name__}, 包含 content: {has_content}")

        # 检查 ground_truth
        gt = df.iloc[0]['reward_model']['ground_truth']
        print_result("Ground truth 存在", gt is not None, f"示例: '{gt}'")

    except Exception as e:
        print_result("数据加载", False, str(e))
        all_passed = False

    # ============ 检查 2: Tokenizer 和 Chat Template ============
    print_header("2. Tokenizer 检查")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print_result("加载 tokenizer", True)

        # 测试 chat template
        sample_prompt = df.iloc[0]['prompt']
        if isinstance(sample_prompt, np.ndarray):
            sample_prompt = sample_prompt.tolist()

        chat_text = tokenizer.apply_chat_template(
            sample_prompt,
            add_generation_prompt=True,
            tokenize=False
        )

        # 验证问题内容在 chat 中
        question_snippet = sample_prompt[0]['content'][:50]
        content_in_chat = question_snippet in chat_text
        print_result("Chat template 包含问题内容", content_in_chat,
                    f"问题片段: '{question_snippet}...'")

        if not content_in_chat:
            all_passed = False

        tokens = tokenizer.encode(chat_text, add_special_tokens=False)
        print_result("Tokenization", True, f"生成 {len(tokens)} 个 tokens")

    except Exception as e:
        print_result("Tokenizer 检查", False, str(e))
        all_passed = False

    # ============ 检查 3: 模型生成 ============
    print_header("3. 模型生成检查")
    try:
        print("加载模型中... (这可能需要 30 秒)")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print_result("加载模型", True)

        # 生成回复
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
        print_result("生成回复", len(response) > 10, f"生成 {len(response)} 字符")

        print("\n--- 模型生成示例 ---")
        print(f"问题: {sample_prompt[0]['content'][:100]}...")
        print(f"回复: {response[:300]}...")

        # 检查回复是否是乱码
        # 简单检查：乱码通常包含大量非 ASCII 混合字符
        ascii_ratio = sum(1 for c in response if ord(c) < 128) / max(len(response), 1)
        is_coherent = ascii_ratio > 0.5 or response.count('####') > 0
        print_result("回复内容有意义", is_coherent, f"ASCII比例: {ascii_ratio:.2%}")

        if not is_coherent:
            all_passed = False

    except Exception as e:
        print_result("模型生成", False, str(e))
        all_passed = False

    # ============ 检查 4: Reward 函数 ============
    print_header("4. Reward 函数检查")
    try:
        # 测试正确答案
        test_response_correct = "Let me calculate step by step.\n48/2 = 24\n48 + 24 = 72\n#### 72"
        gt = "72"
        score_correct = compute_score("openai/gsm8k", test_response_correct, gt)
        print_result("正确答案得分", score_correct == 1.0, f"得分: {score_correct}")

        # 测试错误答案
        test_response_wrong = "The answer is #### 50"
        score_wrong = compute_score("openai/gsm8k", test_response_wrong, gt)
        print_result("错误答案得分", score_wrong == 0.0, f"得分: {score_wrong}")

        # 测试无格式答案
        test_response_no_format = "The answer is 72."
        score_no_format = compute_score("openai/gsm8k", test_response_no_format, gt)
        print_result("无格式答案得分", score_no_format == 0, f"得分: {score_no_format}")

        # 测试模型实际生成的回复
        if 'response' in dir():
            answer = extract_answer(response, method="strict")
            actual_gt = df.iloc[0]['reward_model']['ground_truth']
            actual_score = compute_score("openai/gsm8k", response, actual_gt)
            print_result("模型回复评分", True,
                        f"提取答案: '{answer}', 正确答案: '{actual_gt}', 得分: {actual_score}")

    except Exception as e:
        print_result("Reward 函数", False, str(e))
        all_passed = False

    # ============ 检查 5: rollout_loop.py 修复 ============
    print_header("5. rollout_loop.py 修复检查")
    try:
        with open('/p/scratch/westai0052/liu52/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py', 'r') as f:
            content = f.read()

        fixed = 'isinstance(raw_prompt, (list, np.ndarray))' in content
        print_result("numpy array 类型检查已修复", fixed)

        if not fixed:
            all_passed = False
            print("       ⚠️  需要修复 rollout_loop.py 第 98 行")
            print("       将 isinstance(raw_prompt, list) 改为")
            print("       isinstance(raw_prompt, (list, np.ndarray))")

    except Exception as e:
        print_result("rollout_loop.py 检查", False, str(e))

    # ============ 最终结果 ============
    print_header("最终结果")
    if all_passed:
        print("🎉 所有检查通过！训练管道应该可以正常工作。")
        print("\n如果训练仍然失败，可能的原因是：")
        print("  1. 学习率太高导致策略崩溃")
        print("  2. 没有 KL 惩罚导致模型偏离太远")
        print("  3. Batch size 太小，采样不到正确答案")
    else:
        print("❌ 部分检查失败，请先修复上述问题再训练。")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
