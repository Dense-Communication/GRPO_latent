#!/usr/bin/env python3
"""
测试 prompt 是否能正确传递到模型

这个脚本模拟 rollout_loop.py 中的 preprocess_single_sample 逻辑，
验证修复后 numpy array 类型的 raw_prompt 能否正确处理。
"""

import numpy as np
from transformers import AutoTokenizer

def test_prompt_passing():
    print("=" * 60)
    print("测试 Prompt 传递")
    print("=" * 60)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/p/scratch/westai0052/liu52/models/Qwen2.5-1.5B-Instruct'
    )

    # 模拟从 parquet 加载的 raw_prompt (numpy array 类型)
    raw_prompt = np.array([{
        'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".',
        'role': 'user'
    }], dtype=object)

    print(f"\n1. raw_prompt 类型: {type(raw_prompt)}")
    print(f"   raw_prompt 内容: {raw_prompt}")

    # 模拟环境观测为空的情况 (GSM8K 无环境模式)
    obs_content = ''

    # ========== 修复前的逻辑 (会失败) ==========
    print("\n2. 测试修复前的逻辑 (isinstance(raw_prompt, list)):")
    if obs_content == '' and isinstance(raw_prompt, list) and len(raw_prompt) > 0:
        chat_old = np.array(raw_prompt)
        print("   ✓ 条件满足，使用 raw_prompt")
    else:
        chat_old = np.array([{"content": obs_content, "role": "user"}])
        print("   ✗ 条件不满足，使用空的 obs_content")

    prompt_old = tokenizer.apply_chat_template(chat_old, add_generation_prompt=True, tokenize=False)
    print(f"   生成的 prompt:\n{prompt_old}")

    # ========== 修复后的逻辑 (应该成功) ==========
    print("\n3. 测试修复后的逻辑 (isinstance(raw_prompt, (list, np.ndarray))):")
    if obs_content == '' and isinstance(raw_prompt, (list, np.ndarray)) and len(raw_prompt) > 0:
        chat_new = np.array(raw_prompt)
        print("   ✓ 条件满足，使用 raw_prompt")
    else:
        chat_new = np.array([{"content": obs_content, "role": "user"}])
        print("   ✗ 条件不满足，使用空的 obs_content")

    prompt_new = tokenizer.apply_chat_template(chat_new, add_generation_prompt=True, tokenize=False)
    print(f"   生成的 prompt:\n{prompt_new}")

    # ========== 验证结果 ==========
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    # 检查 user 消息是否包含实际内容
    has_question_old = "Natalia" in prompt_old
    has_question_new = "Natalia" in prompt_new

    print(f"\n修复前: prompt 包含问题内容? {has_question_old}")
    print(f"修复后: prompt 包含问题内容? {has_question_new}")

    # 检查 token 长度
    tokens_old = tokenizer.encode(prompt_old, add_special_tokens=False)
    tokens_new = tokenizer.encode(prompt_new, add_special_tokens=False)

    print(f"\n修复前: prompt token 数量 = {len(tokens_old)}")
    print(f"修复后: prompt token 数量 = {len(tokens_new)}")

    if has_question_new and len(tokens_new) > 50:
        print("\n" + "=" * 60)
        print("✓ 测试通过！Prompt 现在可以正确传递了！")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("✗ 测试失败！Prompt 仍然无法正确传递")
        print("=" * 60)
        return False


def test_actual_rollout_code():
    """
    直接导入并测试实际的 rollout_loop.py 代码
    """
    print("\n\n" + "=" * 60)
    print("测试实际的 rollout_loop.py 代码")
    print("=" * 60)

    try:
        # 读取实际的代码文件检查修复是否已应用
        with open('/p/scratch/westai0052/liu52/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py', 'r') as f:
            content = f.read()

        # 检查修复是否已应用
        if 'isinstance(raw_prompt, (list, np.ndarray))' in content:
            print("✓ rollout_loop.py 中的修复已应用！")
            print("  找到: isinstance(raw_prompt, (list, np.ndarray))")
            return True
        elif 'isinstance(raw_prompt, list)' in content:
            print("✗ rollout_loop.py 中的修复尚未应用！")
            print("  仍然是: isinstance(raw_prompt, list)")
            return False
        else:
            print("? 无法确定修复状态")
            return None

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


if __name__ == "__main__":
    # 运行测试
    test1_passed = test_prompt_passing()
    test2_passed = test_actual_rollout_code()

    print("\n\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    if test1_passed and test2_passed:
        print("✓ 所有测试通过！可以重新运行训练了。")
        print("\n运行命令:")
        print("  cd /p/scratch/westai0052/liu52/verl-agent")
        print("  bash test_script/single_agent_gsm8k/run_gsm8k_4gpu.sh")
    else:
        print("✗ 部分测试失败，请检查修复是否正确应用。")
