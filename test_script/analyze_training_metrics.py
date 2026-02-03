#!/usr/bin/env python3
"""
训练指标分析脚本
分析训练过程中 reward 和 accuracy (test_score) 的变化趋势
"""

import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_log_file(log_path):
    """解析训练日志文件，提取关键指标"""

    metrics = defaultdict(list)

    with open(log_path, 'r') as f:
        content = f.read()

    # 提取每个step的指标
    # 格式: step:N - metric1:value1 - metric2:value2 ...
    step_pattern = r'step:(\d+)\s+-\s+(.+?)(?=step:\d+|$)'

    lines = content.split('\n')
    for line in lines:
        # 匹配包含 step: 的行
        if 'step:' not in line:
            continue

        # 提取 step 号
        step_match = re.search(r'step:(\d+)', line)
        if not step_match:
            continue
        step = int(step_match.group(1))

        # 提取各个指标
        # test_score (验证集准确率)
        test_score_match = re.search(r'val/openai/gsm8k/test_score:([0-9.]+)', line)
        if test_score_match:
            metrics['test_score'].append((step, float(test_score_match.group(1))))

        # 训练 reward (critic/score/mean)
        train_reward_match = re.search(r'critic/score/mean:([0-9.]+)', line)
        if train_reward_match:
            metrics['train_reward'].append((step, float(train_reward_match.group(1))))

        # KL penalty
        kl_match = re.search(r'actor/reward_kl_penalty:([0-9.]+)', line)
        if kl_match:
            metrics['kl_penalty'].append((step, float(kl_match.group(1))))

        # Entropy loss
        entropy_match = re.search(r'actor/entropy_loss:([0-9.]+)', line)
        if entropy_match:
            metrics['entropy_loss'].append((step, float(entropy_match.group(1))))

        # Grad norm
        grad_match = re.search(r'actor/grad_norm:([0-9.]+)', line)
        if grad_match:
            metrics['grad_norm'].append((step, float(grad_match.group(1))))

        # Response length
        resp_len_match = re.search(r'response_length/mean:([0-9.]+)', line)
        if resp_len_match:
            metrics['response_length'].append((step, float(resp_len_match.group(1))))

    return metrics


def compute_moving_average(data, window=5):
    """计算滑动平均"""
    if len(data) < window:
        return data

    steps = [d[0] for d in data]
    values = [d[1] for d in data]

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        avg = np.mean(values[start:end])
        smoothed.append((steps[i], avg))

    return smoothed


def analyze_correlation(metrics):
    """分析 reward 和 accuracy 的相关性"""

    # 获取在相同 step 的 test_score
    test_score_dict = {step: val for step, val in metrics['test_score']}

    # 计算每个 test_freq 区间内的平均 train_reward
    test_steps = sorted(test_score_dict.keys())

    if len(test_steps) < 2:
        return None

    interval_rewards = []
    interval_test_scores = []

    for i, test_step in enumerate(test_steps[1:], 1):
        prev_step = test_steps[i-1]

        # 获取这个区间内的所有 train_reward
        rewards_in_interval = [
            val for step, val in metrics['train_reward']
            if prev_step < step <= test_step
        ]

        if rewards_in_interval:
            avg_reward = np.mean(rewards_in_interval)
            interval_rewards.append(avg_reward)
            interval_test_scores.append(test_score_dict[test_step])

    if len(interval_rewards) >= 3:
        correlation = np.corrcoef(interval_rewards, interval_test_scores)[0, 1]
        return {
            'correlation': correlation,
            'interval_rewards': interval_rewards,
            'interval_test_scores': interval_test_scores
        }

    return None


def print_summary(metrics, correlation_result):
    """打印分析摘要"""

    print("=" * 60)
    print("训练指标分析摘要")
    print("=" * 60)

    # Test Score 趋势
    if metrics['test_score']:
        test_scores = [v for _, v in metrics['test_score']]
        print(f"\n验证集准确率 (test_score):")
        print(f"  初始值: {test_scores[0]:.4f}")
        print(f"  最终值: {test_scores[-1]:.4f}")
        print(f"  最大值: {max(test_scores):.4f} (step {metrics['test_score'][test_scores.index(max(test_scores))][0]})")
        print(f"  最小值: {min(test_scores):.4f}")
        print(f"  提升幅度: {(test_scores[-1] - test_scores[0]) / test_scores[0] * 100:.2f}%")

    # Train Reward 趋势
    if metrics['train_reward']:
        train_rewards = [v for _, v in metrics['train_reward']]
        # 计算滑动平均来看趋势
        window = min(20, len(train_rewards) // 4)
        if window >= 3:
            first_avg = np.mean(train_rewards[:window])
            last_avg = np.mean(train_rewards[-window:])
            print(f"\n训练 Reward (critic/score/mean):")
            print(f"  初期平均 (前{window}步): {first_avg:.4f}")
            print(f"  末期平均 (后{window}步): {last_avg:.4f}")
            print(f"  整体平均: {np.mean(train_rewards):.4f}")
            print(f"  标准差: {np.std(train_rewards):.4f}")

    # Correlation
    if correlation_result:
        print(f"\nReward 与 Accuracy 相关性分析:")
        print(f"  Pearson 相关系数: {correlation_result['correlation']:.4f}")
        if correlation_result['correlation'] > 0.5:
            print("  解读: 强正相关 - reward 上升时 accuracy 也倾向于上升")
        elif correlation_result['correlation'] > 0.2:
            print("  解读: 弱正相关 - reward 和 accuracy 有一定正向关系")
        elif correlation_result['correlation'] > -0.2:
            print("  解读: 无明显相关性 - reward 和 accuracy 变化相对独立")
        elif correlation_result['correlation'] > -0.5:
            print("  解读: 弱负相关 - 可能存在 reward hacking")
        else:
            print("  解读: 强负相关 - 严重 reward hacking，需要调整奖励函数")

    # 参数稳定性分析
    if metrics['grad_norm']:
        grad_norms = [v for _, v in metrics['grad_norm']]
        print(f"\n梯度范数稳定性:")
        print(f"  平均值: {np.mean(grad_norms):.4f}")
        print(f"  标准差: {np.std(grad_norms):.4f}")
        print(f"  最大值: {max(grad_norms):.4f}")
        if max(grad_norms) > 10:
            print("  警告: 梯度范数较大，可能需要梯度裁剪")

    if metrics['entropy_loss']:
        entropies = [v for _, v in metrics['entropy_loss']]
        print(f"\n熵损失趋势:")
        print(f"  初始: {entropies[0]:.4f}")
        print(f"  最终: {entropies[-1]:.4f}")
        change = (entropies[-1] - entropies[0]) / entropies[0] * 100
        print(f"  变化: {change:+.2f}%")
        if change < -30:
            print("  警告: 熵下降过快，模型可能过早收敛")

    print("\n" + "=" * 60)


def plot_metrics(metrics, output_path=None):
    """绘制指标变化图（如果matplotlib可用）"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Test Score 趋势
        ax1 = axes[0, 0]
        if metrics['test_score']:
            steps = [s for s, _ in metrics['test_score']]
            values = [v for _, v in metrics['test_score']]
            ax1.plot(steps, values, 'b-o', label='Test Score', markersize=4)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Test Score (Accuracy)')
            ax1.set_title('验证集准确率趋势')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

        # 2. Train Reward 趋势（含滑动平均）
        ax2 = axes[0, 1]
        if metrics['train_reward']:
            steps = [s for s, _ in metrics['train_reward']]
            values = [v for _, v in metrics['train_reward']]
            ax2.plot(steps, values, 'g-', alpha=0.3, label='原始值')

            # 添加滑动平均
            smoothed = compute_moving_average(metrics['train_reward'], window=10)
            smooth_steps = [s for s, _ in smoothed]
            smooth_values = [v for _, v in smoothed]
            ax2.plot(smooth_steps, smooth_values, 'g-', linewidth=2, label='滑动平均(10步)')

            ax2.set_xlabel('Step')
            ax2.set_ylabel('Train Reward')
            ax2.set_title('训练 Reward 趋势')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        # 3. Test Score vs Train Reward 对比
        ax3 = axes[1, 0]
        if metrics['test_score'] and metrics['train_reward']:
            # 在相同图上绘制两个指标
            steps1 = [s for s, _ in metrics['test_score']]
            values1 = [v for _, v in metrics['test_score']]
            ax3.plot(steps1, values1, 'b-o', label='Test Score', markersize=4)

            ax3_twin = ax3.twinx()
            smoothed = compute_moving_average(metrics['train_reward'], window=10)
            steps2 = [s for s, _ in smoothed]
            values2 = [v for _, v in smoothed]
            ax3_twin.plot(steps2, values2, 'r-', label='Train Reward (MA)', linewidth=2)

            ax3.set_xlabel('Step')
            ax3.set_ylabel('Test Score', color='b')
            ax3_twin.set_ylabel('Train Reward', color='r')
            ax3.set_title('Test Score vs Train Reward')
            ax3.grid(True, alpha=0.3)

            # 合并图例
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

        # 4. 其他稳定性指标
        ax4 = axes[1, 1]
        if metrics['grad_norm']:
            steps = [s for s, _ in metrics['grad_norm']]
            values = [v for _, v in metrics['grad_norm']]
            ax4.plot(steps, values, 'purple', alpha=0.5, label='Grad Norm')

        if metrics['entropy_loss']:
            steps = [s for s, _ in metrics['entropy_loss']]
            values = [v for _, v in metrics['entropy_loss']]
            ax4_twin = ax4.twinx()
            ax4_twin.plot(steps, values, 'orange', label='Entropy Loss')
            ax4_twin.set_ylabel('Entropy', color='orange')

        ax4.set_xlabel('Step')
        ax4.set_ylabel('Grad Norm', color='purple')
        ax4.set_title('训练稳定性指标')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n图表已保存到: {output_path}")
        else:
            # 保存到日志同目录
            plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
            print("\n图表已保存到: training_metrics.png")

        plt.close()

    except ImportError:
        print("\n注意: matplotlib 未安装，跳过图表生成")


def main():
    parser = argparse.ArgumentParser(description='分析 verl-agent 训练日志')
    parser.add_argument('log_path', type=str, help='训练日志文件路径')
    parser.add_argument('--plot', action='store_true', help='生成可视化图表')
    parser.add_argument('--output', type=str, default=None, help='图表输出路径')

    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"错误: 日志文件不存在: {log_path}")
        return

    print(f"正在分析日志: {log_path}")

    # 解析日志
    metrics = parse_log_file(log_path)

    # 分析相关性
    correlation_result = analyze_correlation(metrics)

    # 打印摘要
    print_summary(metrics, correlation_result)

    # 生成图表
    if args.plot:
        output_path = args.output or str(log_path.parent / 'training_metrics.png')
        plot_metrics(metrics, output_path)

    # 输出详细数据供进一步分析
    print("\n详细数据 (test_score):")
    for step, score in metrics['test_score']:
        print(f"  Step {step:4d}: {score:.4f}")


if __name__ == '__main__':
    main()
