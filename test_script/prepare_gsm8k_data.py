#!/usr/bin/env python3
"""准备 GSM8K 数据，添加 ground_truth"""
import pandas as pd
import re

data_dir = "/p/scratch/westai0052/liu52/verl-agent/test_script/data"

train_df = pd.read_parquet(f"{data_dir}/train.parquet")
test_df = pd.read_parquet(f"{data_dir}/test.parquet")

print("原始列:", train_df.columns.tolist())

# GSM8K 标准格式处理
for df in [train_df, test_df]:
    if 'answer' in df.columns:
        # 提取数字答案 (GSM8K 答案格式: "#### 数字")
        df['ground_truth'] = df['answer'].apply(
            lambda x: re.findall(r'\d+', str(x))[-1] if re.findall(r'\d+', str(x)) else "0"
        )
    elif 'ground_truth' not in df.columns:
        print("✗ 没有 answer 或 ground_truth 列")
        print(f"  可用列: {df.columns.tolist()}")
        exit(1)

train_df.to_parquet(f"{data_dir}/train.parquet", index=False)
test_df.to_parquet(f"{data_dir}/test.parquet", index=False)

print(f"✓ 已保存数据，包含 ground_truth")
print(f"  ground_truth 样本: {train_df['ground_truth'].head().tolist()}")
