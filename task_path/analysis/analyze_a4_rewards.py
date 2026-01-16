"""A4实验组分析：奖励函数对比

分析不同奖励函数设计对训练效果的影响
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.data_loader import scan_experiment_results, group_experiments_by
from analysis.metrics_calculator import calculate_all_metrics, extract_eval_metrics
from analysis.visualization import (
    plot_learning_curves,
    configure_plot_style,
)


def parse_reward_name(exp_name: str) -> str:
    """从实验名称中提取奖励函数类型（移除时间戳）"""
    # 从exp_name中提取reward类型
    # 例如：sac_exp_a4_reward_baseline_20260108_124435 -> Baseline
    
    parts = exp_name.split('_')
    for i, part in enumerate(parts):
        if part == 'reward' and i+1 < len(parts):
            reward_type = parts[i+1]
            # 转换为首字母大写
            if reward_type == 'baseline':
                return 'Baseline'
            elif reward_type == 'sparse':
                return 'Sparse'
            elif reward_type == 'dense':
                return 'Dense'
            elif reward_type == 'nostep':
                return 'NoStep'
            elif reward_type == 'energy':
                return 'Energy'
    return 'Unknown'


def main():
    """主函数：A4奖励函数对比分析"""
    
    print("="*80)
    print("A4 实验组分析：奖励函数对比")
    print("="*80)
    
    # 配置路径
    results_dir = Path(__file__).parent.parent / 'results'
    a4_dir = results_dir / 'a4'
    output_dir = a4_dir / 'analysis'
    figures_dir = output_dir / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 加载实验数据...")
    experiments = scan_experiment_results(str(results_dir), experiment_group='a4')
    
    if not experiments:
        print("❌ 未找到A4实验数据！")
        return
    
    print(f"\n✅ 成功加载 {len(experiments)} 个实验")
    
    # 为每个实验设置简短的显示名称
    for exp in experiments:
        exp.display_name = parse_reward_name(exp.exp_name)
    
    # 计算指标
    print(f"\n📊 计算性能指标...")
    exps_with_eval = []
    for exp in experiments:
        metrics = calculate_all_metrics(exp.timesteps, exp.results, exp.ep_lengths)
        exp.metrics = metrics
        
        if exp.eval_data:
            exps_with_eval.append(exp)
            eval_metrics = extract_eval_metrics(exp.eval_data)
            print(f"  {exp.display_name}: eval_success={eval_metrics['eval_success_rate']*100:.1f}%")
    
    print(f"\n  ✅ {len(exps_with_eval)}/{len(experiments)} 个实验包含eval数据")
    
    # 按奖励函数分组
    reward_types = {}
    for exp in experiments:
        reward_name = exp.display_name
        if reward_name not in reward_types:
            reward_types[reward_name] = []
        reward_types[reward_name].append(exp)
    
    print(f"\n按奖励函数分组:")
    for reward, exps in sorted(reward_types.items()):
        print(f"  {reward}: {len(exps)} 个实验")
    
    configure_plot_style()
    
    print(f"\n🎨 生成可视化图表...")
    
    # 图1: 学习曲线对比
    print(f"\n  [1/4] 学习曲线对比...")
    plot_learning_curves(
        experiments,
        group_by='display_name',
        title='Learning Curves: Reward Functions',
        ylabel='Average Return',
        save_path=str(figures_dir / 'a4_learning_curves.png'),
        figsize=(10, 6)
    )
    
    # 图2: Eval性能对比（4个子图）
    if exps_with_eval:
        print(f"  [2/4] Eval性能对比...")
        
        metrics_to_plot = {
            'Success Rate': 'eval_success_rate',
            'Collision Rate': 'eval_collision_rate',
            'Path Length (m)': 'eval_mean_path_length',
            'Energy Consumption': 'eval_mean_energy'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
            ax = axes[idx]
            
            reward_names = sorted(reward_types.keys())
            values = []
            errors = []
            
            for reward_name in reward_names:
                reward_exps = [e for e in reward_types[reward_name] if e.eval_data is not None]
                if reward_exps:
                    metric_values = [extract_eval_metrics(e.eval_data)[metric_key] for e in reward_exps]
                    values.append(np.mean(metric_values))
                    errors.append(np.std(metric_values) if len(metric_values) > 1 else 0)
                else:
                    values.append(0)
                    errors.append(0)
            
            # 绘制柱状图
            bars = ax.bar(range(len(reward_names)), values, yerr=errors, 
                          capsize=5, alpha=0.7, edgecolor='black')
            
            # 设置颜色
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            for bar, color in zip(bars, colors[:len(bars)]):
                bar.set_color(color)
            
            ax.set_xticks(range(len(reward_names)))
            ax.set_xticklabels(reward_names, rotation=45, ha='right')
            ax.set_ylabel(metric_name, fontweight='bold')
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 在柱子上显示数值
            for i, (v, e) in enumerate(zip(values, errors)):
                if metric_name == 'Success Rate' or metric_name == 'Collision Rate':
                    ax.text(i, v + e + 0.02, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9)
                else:
                    ax.text(i, v + e + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Reward Functions: Evaluation Performance Comparison', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(str(figures_dir / 'a4_eval_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ 保存: a4_eval_performance.png")
    
    # 图3: 多维度雷达图
    if exps_with_eval:
        print(f"  [3/4] 多维度雷达图...")
        
        # 准备雷达图数据
        categories = ['Success\nRate', 'Safety\n(1-Collision)', 'Path\nEfficiency', 
                     'Smoothness', 'Energy\nEfficiency']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (reward_name, reward_exps) in enumerate(sorted(reward_types.items())):
            eval_exps = [e for e in reward_exps if e.eval_data is not None]
            if not eval_exps:
                continue
            
            # 计算各维度指标
            metrics_list = [extract_eval_metrics(e.eval_data) for e in eval_exps]
            
            success_rate = np.mean([m['eval_success_rate'] for m in metrics_list])
            safety = 1 - np.mean([m['eval_collision_rate'] for m in metrics_list])
            path_eff = np.mean([15.0 / max(m['eval_mean_path_length'], 15.0) for m in metrics_list])
            smoothness = np.mean([1.0 - m['eval_mean_smoothness'] / 10.0 for m in metrics_list])  # 反转：越小越好变为越大越好
            energy_eff = np.mean([1.0 / (1.0 + m['eval_mean_energy'] / 100.0) for m in metrics_list])
            
            values = [success_rate, safety, path_eff, smoothness, energy_eff]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=reward_name, 
                   color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.title('Reward Functions: Multi-dimensional Performance', 
                 size=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(str(figures_dir / 'a4_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ 保存: a4_radar.png")
    
    # 图4: 样本效率对比
    print(f"  [4/4] 样本效率对比...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    reward_names = sorted(reward_types.keys())
    efficiency_values = []
    
    for reward_name in reward_names:
        reward_exps = reward_types[reward_name]
        eff_list = [e.metrics['sample_efficiency'] for e in reward_exps 
                   if e.metrics['sample_efficiency'] is not None]
        if eff_list:
            efficiency_values.append(np.mean(eff_list) / 1000)  # 转换为K
        else:
            efficiency_values.append(None)
    
    # 绘制柱状图
    x_pos = range(len(reward_names))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(x_pos, [v if v else 0 for v in efficiency_values], 
                  alpha=0.7, edgecolor='black')
    
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)
    
    # 在柱子上标注数值
    for i, v in enumerate(efficiency_values):
        if v:
            ax.text(i, v + 5, f'{v:.0f}K', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(reward_names, fontsize=11)
    ax.set_ylabel('Training Steps to 80% of Max Return (K)', fontsize=12, fontweight='bold')
    ax.set_title('Sample Efficiency: Reward Functions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(str(figures_dir / 'a4_sample_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 保存: a4_sample_efficiency.png")
    plt.close()
    print(f"    ✅ 保存: a4_sample_efficiency.png")
    
    # 生成汇总表
    print(f"\n📄 生成汇总表格...")
    summary_data = []
    for exp in experiments:
        eval_metrics = {}
        if exp.eval_data:
            eval_metrics = extract_eval_metrics(exp.eval_data)
        
        summary_data.append({
            'Experiment': exp.display_name,
            'Algorithm': exp.algorithm,
            'Final_Return': exp.metrics['final_mean'],
            'Peak_Value': exp.metrics['peak_value'],
            'Sample_Efficiency': exp.metrics['sample_efficiency'],
            'Training_Stability': exp.metrics['training_stability'],
            'Eval_Success_Rate': eval_metrics.get('eval_success_rate', np.nan),
            'Eval_Collision_Rate': eval_metrics.get('eval_collision_rate', np.nan),
            'Eval_Path_Length': eval_metrics.get('eval_mean_path_length', np.nan),
            'Eval_Energy': eval_metrics.get('eval_mean_energy', np.nan),
        })
    
    df = pd.DataFrame(summary_data)
    summary_csv = output_dir / 'a4_detailed_summary.csv'
    df.to_csv(summary_csv, index=False, float_format='%.4f')
    print(f"  汇总表: {summary_csv}")
    
    # 生成报告
    print(f"\n📝 生成分析报告...")
    report_path = output_dir / 'a4_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# A4实验组分析报告：奖励函数对比\n\n")
        f.write("**实验目标**: 分析不同奖励函数设计对训练效果的影响\n\n")
        f.write("---\n\n")
        
        f.write("## 1. 实验概览\n\n")
        f.write(f"- **总实验数**: {len(experiments)}\n")
        f.write(f"- **奖励函数变体**: {', '.join(sorted(reward_types.keys()))}\n")
        f.write(f"- **包含Eval数据**: {len(exps_with_eval)}/{len(experiments)}\n")
        f.write(f"- **算法**: SAC (固定)\n\n")
        
        f.write("### 奖励函数设计\n\n")
        f.write("- **Baseline**: 平衡的奖励设计，包含所有组件\n")
        f.write("- **Sparse**: 仅依赖终止奖励（成功/碰撞），无中间引导\n")
        f.write("- **Dense**: 强化进度奖励和时间惩罚，加快学习\n")
        f.write("- **NoStep**: 移除时间压力，让智能体有更多时间探索\n")
        f.write("- **Energy**: 强化能量效率，鼓励平滑控制\n\n")
        
        f.write("---\n\n")
        
        f.write("## 2. 可视化分析\n\n")
        
        f.write("### 2.1 学习曲线对比\n\n")
        f.write("![Learning Curves](figures/a4_learning_curves.png)\n\n")
        f.write("展示不同奖励函数下的训练过程和收敛速度。\n\n")
        
        f.write("### 2.2 Eval性能对比\n\n")
        f.write("![Eval Performance](figures/a4_eval_performance.png)\n\n")
        f.write("基于100个独立评估episodes的性能对比（成功率、碰撞率、路径长度、能量消耗）。\n\n")
        
        f.write("### 2.3 多维度雷达图\n\n")
        f.write("![Radar Chart](figures/a4_radar.png)\n\n")
        f.write("综合评估各奖励函数在多个维度的表现：\n")
        f.write("- Success Rate: 任务成功率\n")
        f.write("- Safety: 安全性 (1 - 碰撞率)\n")
        f.write("- Path Efficiency: 路径效率\n")
        f.write("- Smoothness: 路径平滑度\n")
        f.write("- Energy Efficiency: 能量效率\n\n")
        
        f.write("### 2.4 样本效率对比\n\n")
        f.write("![Sample Efficiency](figures/a4_sample_efficiency.png)\n\n")
        f.write("达到最大回报80%所需的训练步数（越少越好）。\n\n")
        
        f.write("---\n\n")
        
        f.write("## 3. 主要发现\n\n")
        f.write("### 3.1 性能排序\n\n")
        f.write("*(请根据生成的图表填写关键发现)*\n\n")
        f.write("**成功率**: ...\n\n")
        f.write("**学习速度**: ...\n\n")
        f.write("**训练稳定性**: ...\n\n")
        
        f.write("### 3.2 奖励函数影响分析\n\n")
        f.write("- **Sparse奖励**: ...\n")
        f.write("- **Dense奖励**: ...\n")
        f.write("- **NoStep奖励**: ...\n")
        f.write("- **Energy奖励**: ...\n\n")
        
        f.write("### 3.3 推荐配置\n\n")
        f.write("根据实验结果，推荐使用 **[...]** 奖励函数，因为...\n\n")
        
        f.write("---\n\n")
        
        f.write("## 4. 数据表格\n\n")
        f.write(f"完整数据请查看: `a4_detailed_summary.csv`\n\n")
        f.write("```\n")
        f.write(df.to_string(index=False))
        f.write("\n```\n\n")
        
        f.write("---\n\n")
        f.write(f"*报告生成时间: {pd.Timestamp.now()}*\n")
    
    print(f"  分析报告: {report_path}")
    
    print("\n" + "="*80)
    print("✅ A4实验组分析完成！")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
