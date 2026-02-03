#!/usr/bin/env python3
"""
创建固定的验证子集
- 从完整验证集中随机抽取固定数量的样本
- 使用固定随机种子确保可复现
- 生成子集用于训练过程中的频繁验证
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def create_fixed_subset(
    input_path: str,
    output_path: str,
    subset_size: int = 100,
    seed: int = 42,
    stratify: bool = False
):
    """
    创建固定的验证子集

    Args:
        input_path: 原始验证集路径
        output_path: 子集输出路径
        subset_size: 子集大小
        seed: 随机种子
        stratify: 是否按 data_source 分层抽样
    """
    # 读取原始数据
    df = pd.read_parquet(input_path)
    print(f"原始验证集大小: {len(df)}")

    if subset_size >= len(df):
        print(f"子集大小 ({subset_size}) >= 原始数据大小 ({len(df)}), 使用全部数据")
        subset_df = df
    else:
        np.random.seed(seed)

        if stratify and 'data_source' in df.columns:
            # 分层抽样
            groups = df.groupby('data_source')
            sampled_dfs = []

            for name, group in groups:
                n_samples = max(1, int(len(group) / len(df) * subset_size))
                if n_samples > len(group):
                    n_samples = len(group)
                sampled = group.sample(n=n_samples, random_state=seed)
                sampled_dfs.append(sampled)
                print(f"  {name}: 从 {len(group)} 中抽取 {n_samples}")

            subset_df = pd.concat(sampled_dfs, ignore_index=True)
        else:
            # 简单随机抽样
            indices = np.random.choice(len(df), size=subset_size, replace=False)
            subset_df = df.iloc[indices].reset_index(drop=True)

    # 保存子集
    subset_df.to_parquet(output_path, index=False)

    print(f"\n固定验证子集已创建:")
    print(f"  大小: {len(subset_df)}")
    print(f"  随机种子: {seed}")
    print(f"  保存路径: {output_path}")

    # 打印数据源分布
    if 'data_source' in subset_df.columns:
        print(f"\n数据源分布:")
        for source, count in subset_df['data_source'].value_counts().items():
            print(f"  {source}: {count}")

    return subset_df


def main():
    parser = argparse.ArgumentParser(description='创建固定验证子集')
    parser.add_argument('--input', type=str,
                        default='/p/scratch/westai0052/liu52/verl-agent/test_script/data/test.parquet',
                        help='原始验证集路径')
    parser.add_argument('--output', type=str,
                        default='/p/scratch/westai0052/liu52/verl-agent/test_script/data/val_subset.parquet',
                        help='子集输出路径')
    parser.add_argument('--size', type=int, default=100,
                        help='子集大小 (默认: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--stratify', action='store_true',
                        help='是否分层抽样')

    args = parser.parse_args()

    create_fixed_subset(
        input_path=args.input,
        output_path=args.output,
        subset_size=args.size,
        seed=args.seed,
        stratify=args.stratify
    )


if __name__ == '__main__':
    main()
