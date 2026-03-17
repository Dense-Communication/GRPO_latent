#!/bin/bash
# LatentMAS 数据集下载脚本
# 在登录节点运行此脚本下载所有测试数据集
#
# 使用方法 (登录节点):
#   hf-online && bash scripts/download_datasets.sh
#

set -e

# 使用 bashrc 中已配置的 scratch 缓存目录
CACHE_DIR="${HF_HOME:-/p/scratch/westai0052/liu52/.cache/huggingface}"

echo "=============================================="
echo "LatentMAS 数据集下载脚本"
echo "=============================================="
echo "缓存目录: $CACHE_DIR"
echo "数据集目录: ${HF_DATASETS_CACHE:-$CACHE_DIR/datasets}"
echo ""

# 检查是否在线模式
if [ "${HF_HUB_OFFLINE:-1}" = "1" ]; then
    echo "警告: 当前处于离线模式!"
    echo "请先运行: hf-online"
    echo ""
    read -p "是否继续? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    # 临时开启在线模式
    export TRANSFORMERS_OFFLINE=0
    export HF_HUB_OFFLINE=0
    export HF_DATASETS_OFFLINE=0
fi

# Python 下载脚本
python3 << 'EOF'
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("错误: 请先安装 datasets 库")
    print("运行: pip install datasets")
    sys.exit(1)

cache_dir = os.environ.get('HF_DATASETS_CACHE', os.path.join(os.environ.get('HF_HOME', ''), 'datasets'))
print(f"数据集缓存目录: {cache_dir}\n")

datasets_to_download = [
    # (名称, 数据集路径, 子集, split)
    ("GSM8K", "gsm8k", "main", "test"),
    ("AIME 2025", "yentinglin/aime_2025", None, "train"),
    ("AIME 2024", "HuggingFaceH4/aime_2024", None, "train"),
    ("GPQA Diamond", "fingertap/GPQA-Diamond", None, "test"),
    ("ARC-Easy", "allenai/ai2_arc", "ARC-Easy", "test"),
    ("ARC-Challenge", "allenai/ai2_arc", "ARC-Challenge", "test"),
    ("Winogrande", "allenai/winogrande", "winogrande_debiased", "validation"),
    ("MBPP+", "evalplus/mbppplus", None, "test"),
    ("HumanEval+", "evalplus/humanevalplus", None, "test"),
]

success_count = 0
failed = []

for name, path, subset, split in datasets_to_download:
    print(f"[{success_count + 1}/{len(datasets_to_download)}] 下载: {name} ({path})")
    try:
        if subset:
            ds = load_dataset(path, subset, split=split, cache_dir=cache_dir)
        else:
            ds = load_dataset(path, split=split, cache_dir=cache_dir)
        print(f"    ✓ 成功! 样本数: {len(ds)}")
        success_count += 1
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        failed.append((name, path, str(e)))

print("\n" + "=" * 50)
print(f"下载完成: {success_count}/{len(datasets_to_download)} 个数据集成功")

if failed:
    print("\n失败的数据集:")
    for name, path, error in failed:
        print(f"  - {name} ({path}): {error}")

print("\nMedQA 已包含在仓库中 (./data/medqa.json)")
print("=" * 50)
EOF

echo ""
echo "完成! 工作节点会自动使用 scratch 缓存 (通过 bashrc 配置)"
