# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
自定义 GSM8K Reward 函数 (宽松版本)

使用渐进式奖励机制，提供部分分数以帮助模型学习：
- 完全正确 + 标准格式 (#### answer) → 1.0
- 完全正确 + 灵活格式 (最后一个数字匹配) → 0.8
- 格式正确 (有 ####) 但答案错误 → 0.1
- 有数字输出但都不正确 → 0.05
- 没有数字输出 → 0.0

这种设计的目的：
1. 鼓励模型使用正确格式 (1.0 > 0.8)
2. 即使答案错误，正确格式也有小奖励 (0.1)
3. 至少输出数字比完全不输出好 (0.05 > 0.0)
4. 提供学习信号，防止策略崩溃
"""

import re


def extract_answer(solution_str, method="strict"):
    """
    从模型生成的文本中提取答案

    Args:
        solution_str: 模型生成的完整回答文本
        method: 提取方法，"strict" 要求严格格式 "#### answer"，"flexible" 更宽松

    Returns:
        提取到的答案字符串，如果未找到则返回 None
    """
    assert method in ["strict", "flexible"], f"method must be 'strict' or 'flexible', got {method}"

    if method == "strict":
        # 严格模式：要求答案格式为 "#### 数字"
        # 匹配模式：#### 后跟可选的负号和数字（可能包含小数点和逗号）
        match = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
        if match is None:
            return None
        else:
            # 提取答案并清理格式（移除逗号和美元符号）
            answer = match.group(1).replace(",", "").replace("$", "")
            return answer

    elif method == "flexible":
        # 灵活模式：提取文本中最后一个数字作为答案
        # 匹配整数或小数 (如 42, -3.14, 1,000)
        numbers = re.findall(r"(\-?[\d,]+\.?\d*)", solution_str)
        if len(numbers) == 0:
            return None

        # 从后往前找第一个有效数字
        invalid_str = ["", ".", "-"]
        for num in reversed(numbers):
            cleaned = num.replace(",", "").replace("$", "").rstrip(".")
            if cleaned not in invalid_str and cleaned:
                # 如果是整数形式，移除末尾的 .0
                if cleaned.endswith(".0"):
                    cleaned = cleaned[:-2]
                return cleaned

        return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    GSM8K 自定义 Reward 函数 (宽松版本)

    使用渐进式奖励机制：
    1. 完全正确 + 标准格式 (#### answer) → 1.0
    2. 完全正确 + 灵活格式 (最后一个数字匹配) → 0.8
    3. 格式正确 (有 ####) 但答案错误 → 0.1
    4. 有数字输出但都不正确 → 0.05
    5. 没有数字输出 → 0.0

    Args:
        data_source: 数据集来源，如 "openai/gsm8k"
        solution_str: 模型生成的完整回答文本
        ground_truth: 正确答案（字符串格式）
        extra_info: 额外信息（可选，本函数未使用）
        **kwargs: 其他可选参数

    Returns:
        float: reward 分数
    """
    # 尝试严格模式提取 (#### answer 格式)
    strict_answer = extract_answer(solution_str, method="strict")

    # 尝试灵活模式提取 (最后一个数字)
    flexible_answer = extract_answer(solution_str, method="flexible")

    # 情况1: 严格格式 + 答案正确 → 1.0 (最高奖励)
    if strict_answer is not None and strict_answer == ground_truth:
        return 1.0

    # 情况2: 灵活格式 + 答案正确 → 0.8 (次高奖励)
    if flexible_answer is not None and flexible_answer == ground_truth:
        return 0.8

    # 情况3: 有 #### 格式但答案错误 → 0.1 (格式奖励)
    if strict_answer is not None:
        return 0.1

    # 情况4: 有数字输出但答案错误 → 0.05 (尝试奖励)
    if flexible_answer is not None:
        return 0.05

    # 情况5: 完全没有数字输出 → 0.0
    return 0.0


# 如果你想要更灵活的提取方式，可以使用下面这个函数
def compute_score_flexible(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    GSM8K Reward 函数（灵活模式）

    使用更宽松的答案提取策略，适用于模型可能不按标准格式输出的情况

    Args:
        data_source: 数据集来源
        solution_str: 模型生成的完整回答文本
        ground_truth: 正确答案
        extra_info: 额外信息（可选）
        **kwargs: 其他可选参数

    Returns:
        float: reward 分数 (0, 0.0, 或 1.0)
    """
    # 使用灵活模式提取答案（提取文本中最后一个数字）
    answer = extract_answer(solution_str, method="flexible")

    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0


if __name__ == "__main__":
    # 测试用例
    print("=" * 60)
    print("测试 GSM8K 自定义 Reward 函数 (宽松版本)")
    print("=" * 60)
    print("\n评分标准:")
    print("  1.0  - 严格格式 (####) + 答案正确")
    print("  0.8  - 灵活格式 (最后数字) + 答案正确")
    print("  0.1  - 有 #### 格式但答案错误")
    print("  0.05 - 有数字输出但答案错误")
    print("  0.0  - 没有数字输出")

    # 测试案例1：完全正确 (严格格式)
    test_solution_1 = """
    Let's solve this step by step:
    First, we add 7 and 13: 7 + 13 = 20
    Then we calculate the fraction: 7/20 * 120 = 42
    #### 42
    """
    ground_truth_1 = "42"
    score_1 = compute_score("openai/gsm8k", test_solution_1, ground_truth_1)
    print(f"\n测试1 - 严格格式 + 答案正确:")
    print(f"  Ground Truth: {ground_truth_1}")
    print(f"  Score: {score_1} (期望: 1.0) {'✓' if score_1 == 1.0 else '✗'}")

    # 测试案例2：灵活格式正确
    test_solution_2 = "After calculation, the final answer is 42."
    score_2 = compute_score("openai/gsm8k", test_solution_2, ground_truth_1)
    print(f"\n测试2 - 灵活格式 + 答案正确:")
    print(f"  Ground Truth: {ground_truth_1}")
    print(f"  Solution: {test_solution_2}")
    print(f"  Score: {score_2} (期望: 0.8) {'✓' if score_2 == 0.8 else '✗'}")

    # 测试案例3：格式正确但答案错误
    test_solution_3 = """
    Let me calculate:
    7 + 13 = 20
    The answer is wrong calculation
    #### 50
    """
    score_3 = compute_score("openai/gsm8k", test_solution_3, ground_truth_1)
    print(f"\n测试3 - 严格格式 + 答案错误:")
    print(f"  Ground Truth: {ground_truth_1}")
    print(f"  Solution: ...#### 50")
    print(f"  Score: {score_3} (期望: 0.1) {'✓' if score_3 == 0.1 else '✗'}")

    # 测试案例4：有数字但不匹配
    test_solution_4 = "I think the answer might be 50 or 60."
    score_4 = compute_score("openai/gsm8k", test_solution_4, ground_truth_1)
    print(f"\n测试4 - 有数字但答案错误:")
    print(f"  Ground Truth: {ground_truth_1}")
    print(f"  Solution: {test_solution_4}")
    print(f"  Score: {score_4} (期望: 0.05) {'✓' if score_4 == 0.05 else '✗'}")

    # 测试案例5：没有数字
    test_solution_5 = "Let me think about this problem. The answer is forty-two."
    score_5 = compute_score("openai/gsm8k", test_solution_5, ground_truth_1)
    print(f"\n测试5 - 没有数字输出:")
    print(f"  Ground Truth: {ground_truth_1}")
    print(f"  Solution: {test_solution_5}")
    print(f"  Score: {score_5} (期望: 0.0) {'✓' if score_5 == 0.0 else '✗'}")

    # 测试案例6：乱码/无意义输出
    test_solution_6 = "I'm sorry, but I'm not sure what you're asking."
    score_6 = compute_score("openai/gsm8k", test_solution_6, ground_truth_1)
    print(f"\n测试6 - 无意义输出 (模拟策略崩溃):")
    print(f"  Ground Truth: {ground_truth_1}")
    print(f"  Solution: {test_solution_6}")
    print(f"  Score: {score_6} (期望: 0.0) {'✓' if score_6 == 0.0 else '✗'}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n这个宽松版本的优势：")
    print("  - 即使模型一开始不会用 #### 格式，也能获得 0.8 分")
    print("  - 尝试使用格式但答案错误，也有 0.1 的小奖励")
    print("  - 提供持续的学习信号，防止策略崩溃")
