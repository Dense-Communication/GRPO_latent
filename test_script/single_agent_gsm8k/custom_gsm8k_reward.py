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
自定义 GSM8K Reward 函数
使用简单的三级评分机制：
- 完全正确（答案匹配） → 返回 1.0
- 格式正确但答案错误 → 返回 0.0
- 格式错误（无法提取答案） → 返回 0
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
        numbers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        if len(numbers) == 0:
            return None

        # 从后往前找第一个有效数字（排除单独的 "." 等无效字符）
        invalid_str = ["", "."]
        for num in reversed(numbers):
            if num not in invalid_str:
                return num.replace(",", "").replace("$", "")

        return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    GSM8K 自定义 Reward 函数

    这是一个简单的三级评分函数：
    1. 如果答案完全正确 → reward = 1.0
    2. 如果格式正确但答案错误 → reward = 0.0
    3. 如果格式错误（无法提取答案） → reward = 0

    Args:
        data_source: 数据集来源，如 "openai/gsm8k"
        solution_str: 模型生成的完整回答文本
        ground_truth: 正确答案（字符串格式）
        extra_info: 额外信息（可选，本函数未使用）
        **kwargs: 其他可选参数

    Returns:
        float: reward 分数 (0, 0.0, 或 1.0)
    """
    # 提取模型回答中的答案（使用严格模式）
    answer = extract_answer(solution_str, method="strict")

    if answer is None:
        # 情况3：格式错误，无法提取答案
        return 0
    else:
        # 将答案与 ground_truth 比较
        if answer == ground_truth:
            # 情况1：完全正确
            return 1.0
        else:
            # 情况2：格式正确但答案错误
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
    print("=" * 50)
    print("测试 GSM8K 自定义 Reward 函数")
    print("=" * 50)

    # 测试案例1：完全正确
    test_solution_1 = """
    Let's solve this step by step:
    First, we add 7 and 13: 7 + 13 = 20
    Then we calculate the fraction: 7/20 * 120 = 42
    #### 42
    """
    ground_truth_1 = "42"
    score_1 = compute_score("openai/gsm8k", test_solution_1, ground_truth_1)
    print(f"\n测试1 - 完全正确:")
    print(f"Ground Truth: {ground_truth_1}")
    print(f"Solution: {test_solution_1.strip()}")
    print(f"Score: {score_1} (期望: 1.0)")

    # 测试案例2：格式正确但答案错误
    test_solution_2 = """
    Let me calculate:
    7 + 13 = 20
    The answer is wrong calculation
    #### 50
    """
    ground_truth_2 = "42"
    score_2 = compute_score("openai/gsm8k", test_solution_2, ground_truth_2)
    print(f"\n测试2 - 格式正确但答案错误:")
    print(f"Ground Truth: {ground_truth_2}")
    print(f"Solution: {test_solution_2.strip()}")
    print(f"Score: {score_2} (期望: 0.0)")

    # 测试案例3：格式错误
    test_solution_3 = """
    Let me think about this problem.
    The calculation shows the answer is forty-two.
    """
    ground_truth_3 = "42"
    score_3 = compute_score("openai/gsm8k", test_solution_3, ground_truth_3)
    print(f"\n测试3 - 格式错误 (没有 #### ):")
    print(f"Ground Truth: {ground_truth_3}")
    print(f"Solution: {test_solution_3.strip()}")
    print(f"Score: {score_3} (期望: 0)")

    # 测试案例4：灵活模式
    score_4 = compute_score_flexible("openai/gsm8k", test_solution_3, ground_truth_3)
    print(f"\n测试4 - 灵活模式 (同样的错误格式):")
    print(f"Score (flexible): {score_4} (期望: 0，因为没有数字)")

    # 测试案例5：灵活模式可以提取的情况
    test_solution_5 = "The final answer is 42."
    score_5 = compute_score_flexible("openai/gsm8k", test_solution_5, ground_truth_3)
    print(f"\n测试5 - 灵活模式可以提取:")
    print(f"Solution: {test_solution_5}")
    print(f"Score (flexible): {score_5} (期望: 1.0)")

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)
