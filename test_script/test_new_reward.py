#!/usr/bin/env python3
"""快速测试新的 GSM8K 评分函数"""

import sys
sys.path.insert(0, '/p/scratch/westai0052/liu52/verl-agent')

from verl.utils.reward_score.gsm8k import compute_score, extract_solution

print("=" * 60)
print("测试新的 GSM8K 评分函数")
print("=" * 60)

test_cases = [
    # (response, ground_truth, expected_score, description)
    ("#### 42", "42", 1.0, "标准格式 #### 正确"),
    ("\\boxed{105}", "105", 1.0, "LaTeX \\boxed 格式正确"),
    ("The answer is 42.", "42", 1.0, "自然语言格式正确"),
    ("Therefore, the final answer is:\n\\[\n\\boxed{162000}\n\\]", "162000", 1.0, "多行 boxed 格式"),
    ("Step by step... The total is 30 apples.", "30", 0.8, "灵活格式正确"),
    ("#### 50", "42", 0.1, "#### 格式但答案错误"),
    ("\\boxed{999}", "42", 0.1, "boxed 格式但答案错误"),
    ("I think it's about 50 or 60.", "42", 0.05, "有数字但答案错误"),
    ("I'm not sure what you're asking.", "42", 0.0, "没有数字输出"),
    ("The answer is forty-two.", "42", 0.0, "文字数字不计分"),
]

passed = 0
failed = 0

for response, gt, expected, desc in test_cases:
    score = compute_score(response, gt)
    status = "✓" if abs(score - expected) < 0.001 else "✗"
    if status == "✓":
        passed += 1
    else:
        failed += 1
    print(f"\n{status} {desc}")
    print(f"  Response: {response[:50]}...")
    print(f"  Ground Truth: {gt}")
    print(f"  Score: {score} (期望: {expected})")

print("\n" + "=" * 60)
print(f"结果: {passed}/{passed+failed} 通过")
print("=" * 60)

# 测试实际日志中的例子
print("\n\n测试训练日志中的实际例子:")
print("-" * 60)

real_examples = [
    ("To determine how much money James has in cents, let's break it down step by step:\n\n1. **Quarter**: A quarter is worth 25 cents.\n   \\[\n   \\text{Value of a quarter} = 25 \\text{ cents}\n   \\]\n\n2. **Nickels**: A nickel is worth 5 cents.\n   \\[\n   \\text{Value of two nickels} = 2 \\times 5 = 10 \\text{ cents}\n   \\]\n\n3. **Dimes**: A dime is worth 10 cents.\n   \\[\n   \\text{Value of 7 dimes} = 7 \\times 10 = 70 \\text{ cents}\n   \\]\n\nNow, let's add up all the values:\n\\[\n\\text{Total value} = 25 + 10 + 70 = 105 \\text{ cents}\n\\]\n\nTherefore, the final answer is:\n\\[\n\\boxed{105}\n\\]", "105", "James coins 问题"),

    ("Therefore, each graduate would receive **6 tickets**.", "5", "Tickets 问题 (答案错误但有结构)"),
]

for response, gt, desc in real_examples:
    strict = extract_solution(response, "strict")
    flexible = extract_solution(response, "flexible")
    score = compute_score(response, gt)

    print(f"\n{desc}:")
    print(f"  Ground Truth: {gt}")
    print(f"  Strict提取: {strict}")
    print(f"  Flexible提取: {flexible}")
    print(f"  得分: {score}")

print("\n" + "=" * 60)
print("如果 James coins 问题得 1.0 分，说明修复成功！")
print("=" * 60)
