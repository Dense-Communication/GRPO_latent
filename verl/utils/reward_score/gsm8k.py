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

import re


def extract_solution(solution_str, method="strict"):
    """
    从模型输出中提取答案

    支持的格式:
    - #### number (GSM8K 标准格式)
    - \\boxed{number} (LaTeX 格式，Qwen 常用)
    - The answer is number (自然语言格式)
    """
    assert method in ["strict", "flexible"]

    if method == "strict":
        # 首先尝试 #### 格式
        solution = re.search(r"####\s*(\-?[0-9\.\,]+)", solution_str)
        if solution is not None:
            final_answer = solution.group(1).replace(",", "").replace("$", "")
            return final_answer

        # 然后尝试 \boxed{} 格式 (Qwen 模型常用)
        boxed = re.search(r"\\boxed\{(\-?[0-9\.\,]+)\}", solution_str)
        if boxed is not None:
            final_answer = boxed.group(1).replace(",", "").replace("$", "")
            return final_answer

        # 尝试 **answer** 或 answer is X 格式
        answer_is = re.search(r"(?:answer|result|total)\s*(?:is|=|:)\s*\**(\-?[0-9\.\,]+)\**", solution_str, re.IGNORECASE)
        if answer_is is not None:
            final_answer = answer_is.group(1).replace(",", "").replace("$", "")
            return final_answer

        return None

    elif method == "flexible":
        answer = re.findall(r"(\-?[\d,]+\.?\d*)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", ".", "-"]
            # find the last number that is not '.'
            for num in reversed(answer):
                cleaned = num.replace(",", "").rstrip(".")
                if cleaned not in invalid_str and cleaned:
                    final_answer = cleaned
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k (宽松版本).

    使用渐进式奖励:
    - 1.0: 答案完全正确
    - 0.1: 有结构化输出但答案错误
    - 0.05: 有数字但答案错误 (flexible mode)
    - 0.0: 没有数字输出

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format (default 0.0, but we use 0.1 for partial credit)
        score: the score for the correct answer
    """
    # 尝试严格模式提取
    strict_answer = extract_solution(solution_str=solution_str, method="strict")

    # 尝试灵活模式提取
    flexible_answer = extract_solution(solution_str=solution_str, method="flexible")

    # 情况1: 严格格式 + 答案正确 → 满分
    if strict_answer is not None and strict_answer == ground_truth:
        return score  # 1.0

    # 情况2: 灵活格式 + 答案正确 → 0.8 分
    if flexible_answer is not None and flexible_answer == ground_truth:
        return score * 0.8  # 0.8

    # 情况3: 有结构化格式但答案错误 → 0.1 分
    if strict_answer is not None:
        return 0.1

    # 情况4: 有数字但答案错误 → 0.05 分
    if flexible_answer is not None:
        return 0.05

    # 情况5: 完全没有数字 → 0 分
    return 0
