import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_choice_answer(text: str) -> Optional[str]:
    """
    Extract multiple choice answer (A/B/C/D) from text.
    Used for GPQA, ARC, and other multiple choice datasets.
    """
    # Look for patterns like "Answer: A", "The answer is B", "(C)", etc.
    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\s*(?:is correct|is the answer)",
        r"\\boxed\{([A-Da-d])\}",
        r"\(([A-Da-d])\)\s*$",
        r"^([A-Da-d])[\.\):\s]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).lower()

    # Last resort: find any standalone A/B/C/D at end of text
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):
        line = line.strip()
        if re.match(r'^[A-Da-d][\.\):\s]*$', line):
            return line[0].lower()
        match = re.search(r'\b([A-Da-d])\b', line)
        if match and len(line) < 50:  # Short line with a letter
            return match.group(1).lower()

    return None


def extract_winogrande_answer(text: str) -> Optional[str]:
    """
    Extract Winogrande answer (1 or 2) from text.
    """
    # Look for patterns like "Answer: 1", "The answer is 2", "option 1", etc.
    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)?\s*([12])",
        r"\b([12])\s*(?:is correct|is the answer)",
        r"\\boxed\{([12])\}",
        r"^([12])[\.\):\s]*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)

    # Last resort: find any standalone 1 or 2 at end of text
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):
        line = line.strip()
        if re.match(r'^[12][\.\):\s]*$', line):
            return line[0]
        # Look for "1" or "2" in short lines
        match = re.search(r'\b([12])\b', line)
        if match and len(line) < 30:
            return match.group(1)

    return None


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)

