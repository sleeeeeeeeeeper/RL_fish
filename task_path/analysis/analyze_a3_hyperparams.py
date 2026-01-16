"""A3实验组分析：超参数敏感性

分析不同超参数对各算法性能的影响
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.data_loader import scan_experiment_results, group_experiments_by
from analysis.metrics_calculator import calculate_all_metrics, extract_eval_metrics
from analysis.visualization import (
    plot_learning_curves,
    configure_plot_style,
)


def parse_hyperparam_from_name(exp_name: str) -> tuple:
    """从实验名称解析超参数类型和值
    
    Returns:
        (param_type, param_value, display_name)
    """
    # 移除时间戳部分
    name_parts = exp_name.split('_')
    
    # 找到超参数部分（不含时间戳）
    hyperparam_part = None
    for part in name_parts:
        if any(x in part for x in ['lr', 'batch', 'clip', 'ep', 'buf', 'ent', 'delay', 'noise']):
            hyperparam_part = part
            break
    
    if not hyperparam_part:
        return ('unknown', 'unknown', exp_name)
    
    # 解析不同类型的超参数
    if hyperparam_part.startswith('lr'):
        # lr1e4, lr3e4, lr1e3 -> Learning Rate
        value = hyperparam_part[2:]
        if 'e' in value:
            # 转换科学计数法
            numeric_value = float(value.replace('e', 'e-'))
            display_name = f'LR={numeric_value:.0e}'
        else:
            display_name = f'LR={value}'
        return ('Learning Rate', value, display_name)
    
    elif hyperparam_part.startswith('batch'):
        # batch32, batch64 -> Batch Size
        value = hyperparam_part[5:]
        return ('Batch Size', value, f'Batch={value}')
    
    elif hyperparam_part.startswith('clip'):
        # clip01, clip02, clip03 -> Clip Range (需要转换为0.1, 0.2, 0.3)
        value = hyperparam_part[4:]
        actual_value = float(value) / 10  # 01->0.1, 02->0.2
        return ('Clip Range', value, f'Clip={actual_value:.1f}')
    
    elif hyperparam_part.startswith('ep'):
        # ep5, ep10, ep15, ep20 -> Epochs
        value = hyperparam_part[2:]
        return ('Epochs', value, f'Epochs={value}')
    
    elif hyperparam_part.startswith('buf'):
        # buf100k, buf500k, buf1m, buf2m -> Buffer Size
        value = hyperparam_part[3:]
        if 'k' in value:
            display_value = value.upper()
        elif 'm' in value:
            display_value = value.upper()
        else:
            display_value = value
        return ('Buffer Size', value, f'Buffer={display_value}')
    
    elif hyperparam_part.startswith('ent'):
        # ent01, ent03, ent05, ent_auto -> Entropy Coef
        value = hyperparam_part[3:]
        if 'auto' in value or value == '' or value.startswith('_'):
            display_name = 'Ent=auto'
            actual_value = 'auto'
        else:
            actual_value = float(value) / 10
            display_name = f'Ent={actual_value:.1f}'
        return ('Entropy Coef', value if value else 'auto', display_name)
    
    elif hyperparam_part.startswith('delay'):
        # delay1, delay2, delay3 -> Policy Delay
        value = hyperparam_part[5:]
        return ('Policy Delay', value, f'Delay={value}')
    
    elif hyperparam_part.startswith('noise'):
        # noise01, noise02, noise03 -> Target Noise
        value = hyperparam_part[5:]
        actual_value = float(value) / 10
        return ('Target Noise', value, f'Noise={actual_value:.1f}')
    
    return ('unknown', hyperparam_part, hyperparam_part)


def get_hyperparam_sort_key(param_type: str, param_value: str) -> float:
    """获取超参数的排序键值（用于横坐标排序）"""
    try:
        # Learning Rate: 1e4, 3e4, 1e3
        if param_type == 'Learning Rate':
            return float(param_value.replace('e', 'e-'))
        
        # Clip Range: 01, 02, 03 -> 0.1, 0.2, 0.3
        elif param_type == 'Clip Range':
            return float(param_value) / 10
        
        # Entropy: 01, 03, 05, auto
        elif param_type == 'Entropy Coef':
            if 'auto' in str(param_value) or param_value == '':
                return -1  # auto放在最前面
            try:
                return float(param_value) / 10
            except:
                return -1
        
        # Target Noise: 01, 02, 03
        elif param_type == 'Target Noise':
            return float(param_value) / 10
        
        # Buffer Size: 100k, 500k, 1m, 1500k, 2m
        elif param_type == 'Buffer Size':
            if 'k' in param_value:
                return float(param_value.replace('k', '')) * 1000
            elif 'm' in param_value:
                return float(param_value.replace('m', '')) * 1000000
            return float(param_value)
        
        # 其他：直接转数字
        else:
            return float(param_value)
    except:
        return 0


def plot_hyperparam_sensitivity_line(experiments, param_type, algo, save_path, figsize=(10, 6)):
    """绘制超参数敏感性折线图（双纵轴：成功率+碰撞率）"""
    
    # 提取数据
    data_points = []
    for exp in experiments:
        ptype, pvalue, display_name = parse_hyperparam_from_name(exp.exp_name)
        if ptype == param_type and exp.eval_data:
            eval_metrics = extract_eval_metrics(exp.eval_data)
            sort_key = get_hyperparam_sort_key(param_type, pvalue)
            data_points.append({
                'sort_key': sort_key,
                'display_name': display_name,
                'success_rate': eval_metrics['eval_success_rate'] * 100,  # 转换为百分比
                'collision_rate': eval_metrics['eval_collision_rate'] * 100,
            })
    
    if not data_points:
        print(f"    ⚠️ {param_type}: 无有效数据")
        return
    
    # 按sort_key排序
    data_points.sort(key=lambda x: x['sort_key'])
    
    # 提取数据
    x_labels = [d['display_name'] for d in data_points]
    success_rates = [d['success_rate'] for d in data_points]
    collision_rates = [d['collision_rate'] for d in data_points]
    
    # 绘图
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 左侧Y轴：成功率（绿色）
    color1 = '#2ecc71'
    ax1.set_xlabel(param_type, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(x_labels, success_rates, color=color1, marker='o', 
                      linewidth=2, markersize=8, label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 100])
    
    # 右侧Y轴：碰撞率（红色）
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel('Collision Rate (%)', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(x_labels, collision_rates, color=color2, marker='s', 
                      linewidth=2, markersize=8, label='Collision Rate')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 100])
    
    # 标题
    plt.title(f'{algo.upper()} - {param_type} Sensitivity', 
              fontsize=14, fontweight='bold', pad=20)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ {param_type}: {save_path}")


def plot_hyperparam_learning_curves(experiments, param_type, algo, save_path, figsize=(12, 8)):
    """绘制不同超参数值下的学习曲线对比（子图形式）"""
    
    # 按超参数值分组
    param_groups = defaultdict(list)
    for exp in experiments:
        ptype, pvalue, display_name = parse_hyperparam_from_name(exp.exp_name)
        if ptype == param_type:
            param_groups[param_type].append(exp)
            exp.display_name = display_name  # 设置显示名称
            # 调试：输出display_name
            if param_type == 'Buffer Size':
                print(f"      - {exp.exp_name.split('_2026')[0]}: display_name='{display_name}'")
    
    if not param_groups[param_type]:
        print(f"    ⚠️ {param_type}: 无学习曲线数据")
        return
    
    # 使用visualization模块的函数绘制学习曲线
    plot_learning_curves(
        param_groups[param_type],
        group_by='display_name',
        title=f'{algo.upper()} - {param_type} Learning Curves',
        ylabel='Average Return',
        save_path=save_path,
        figsize=figsize
    )
    print(f"    ✅ {param_type}学习曲线: {save_path}")


def main():
    """主函数：A3超参数敏感性分析"""
    
    print("="*80)
    print("A3 实验组分析：超参数敏感性分析")
    print("="*80)
    
    # 配置路径
    results_dir = Path(__file__).parent.parent / 'results'
    a3_dir = results_dir / 'a3'
    output_dir = a3_dir / 'analysis'
    figures_dir = output_dir / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 加载实验数据...")
    experiments = scan_experiment_results(str(results_dir), experiment_group='a3')
    
    if not experiments:
        print("❌ 未找到A3实验数据！")
        return
    
    print(f"\n✅ 成功加载 {len(experiments)} 个实验")
    
    # 计算指标
    print(f"\n📊 计算性能指标...")
    exps_with_eval = []
    for exp in experiments:
        metrics = calculate_all_metrics(exp.timesteps, exp.results, exp.ep_lengths)
        exp.metrics = metrics
        
        if exp.eval_data:
            exps_with_eval.append(exp)
            eval_metrics = extract_eval_metrics(exp.eval_data)
            print(f"  {exp.exp_name.split('_2026')[0]}: eval_success={eval_metrics['eval_success_rate']*100:.1f}%")
    
    print(f"\n  ✅ {len(exps_with_eval)}/{len(experiments)} 个实验包含eval数据")
    
    # 按算法分组
    exps_by_algo = group_experiments_by(experiments, by='algorithm')
    
    print(f"\n按算法分组:")
    for algo, exps in sorted(exps_by_algo.items()):
        print(f"  {algo}: {len(exps)} 个实验")
    
    configure_plot_style()
    
    # 定义每个算法需要分析的超参数
    algo_params = {
        'PPO': ['Learning Rate', 'Batch Size', 'Clip Range', 'Epochs'],
        'SAC': ['Learning Rate', 'Batch Size', 'Buffer Size', 'Entropy Coef'],
        'TD3': ['Learning Rate', 'Policy Delay', 'Target Noise']
    }
    
    print(f"\n🎨 生成可视化图表...")
    
    # 为每个算法生成图表
    for algo_idx, (algo, algo_exps) in enumerate(sorted(exps_by_algo.items())):
        print(f"\n{'='*60}")
        print(f"📊 [{algo_idx+1}/{len(exps_by_algo)}] {algo}超参数分析")
        print(f"{'='*60}")
        
        if algo.upper() not in algo_params:
            print(f"  ⚠️ 未定义{algo}的超参数列表")
            continue
        
        params_to_analyze = algo_params[algo.upper()]
        
        for param_idx, param_type in enumerate(params_to_analyze):
            print(f"\n  [{param_idx+1}/{len(params_to_analyze)}] {param_type}:")
            
            # 过滤出该超参数类型的实验
            param_exps = []
            for exp in algo_exps:
                ptype, _, _ = parse_hyperparam_from_name(exp.exp_name)
                if ptype == param_type:
                    param_exps.append(exp)
            
            if not param_exps:
                print(f"    ⚠️ 无{param_type}实验")
                continue
            
            print(f"    找到 {len(param_exps)} 个实验")
            
            # 1. 敏感性折线图（成功率+碰撞率）
            safe_param_name = param_type.lower().replace(' ', '_')
            plot_hyperparam_sensitivity_line(
                param_exps,
                param_type,
                algo,
                save_path=str(figures_dir / f'a3_{algo.lower()}_{safe_param_name}_sensitivity.png'),
                figsize=(10, 6)
            )
            
            # 2. 学习曲线对比
            plot_hyperparam_learning_curves(
                param_exps,
                param_type,
                algo,
                save_path=str(figures_dir / f'a3_{algo.lower()}_{safe_param_name}_learning.png'),
                figsize=(10, 6)
            )
    
    # 生成汇总表
    print(f"\n📄 生成汇总表格...")
    summary_data = []
    for exp in experiments:
        ptype, pvalue, display_name = parse_hyperparam_from_name(exp.exp_name)
        
        eval_metrics = {}
        if exp.eval_data:
            eval_metrics = extract_eval_metrics(exp.eval_data)
        
        summary_data.append({
            'Experiment': exp.exp_name.split('_2026')[0],  # 移除时间戳
            'Algorithm': exp.algorithm,
            'Hyperparam_Type': ptype,
            'Hyperparam_Display': display_name,
            'Final_Return': exp.metrics['final_mean'],
            'Sample_Efficiency': exp.metrics['sample_efficiency'],
            'Eval_Success_Rate': eval_metrics.get('eval_success_rate', np.nan),
            'Eval_Collision_Rate': eval_metrics.get('eval_collision_rate', np.nan),
            'Eval_Path_Length': eval_metrics.get('eval_mean_path_length', np.nan),
            'Eval_Energy': eval_metrics.get('eval_mean_energy', np.nan),
        })
    
    df = pd.DataFrame(summary_data)
    summary_csv = output_dir / 'a3_detailed_summary.csv'
    df.to_csv(summary_csv, index=False, float_format='%.4f')
    print(f"  汇总表: {summary_csv}")
    
    # 生成报告
    print(f"\n📝 生成分析报告...")
    report_path = output_dir / 'a3_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# A3实验组分析报告：超参数敏感性\n\n")
        f.write("**实验目标**: 分析超参数对算法性能的影响\n\n")
        f.write("---\n\n")
        
        f.write("## 1. 实验概览\n\n")
        f.write(f"- **总实验数**: {len(experiments)}\n")
        f.write(f"- **算法**: {', '.join(sorted(exps_by_algo.keys()))}\n")
        f.write(f"- **包含Eval数据**: {len(exps_with_eval)}/{len(experiments)}\n\n")
        
        # 按算法统计实验数
        f.write("### 各算法实验分布\n\n")
        for algo, exps in sorted(exps_by_algo.items()):
            f.write(f"- **{algo}**: {len(exps)}个实验\n")
            
            # 统计该算法测试的超参数类型
            param_types = set()
            for exp in exps:
                ptype, _, _ = parse_hyperparam_from_name(exp.exp_name)
                if ptype != 'unknown':
                    param_types.add(ptype)
            f.write(f"  - 测试超参数: {', '.join(sorted(param_types))}\n")
        
        f.write("\n---\n\n")
        
        f.write("## 2. 超参数敏感性分析\n\n")
        f.write("### 2.1 图表说明\n\n")
        f.write("每个算法生成两类图表：\n\n")
        f.write("1. **敏感性折线图** (`*_sensitivity.png`)\n")
        f.write("   - 横坐标：超参数值\n")
        f.write("   - 左侧纵坐标（绿色）：成功率 (%)\n")
        f.write("   - 右侧纵坐标（红色）：碰撞率 (%)\n")
        f.write("   - 用于观察超参数变化对性能和安全性的影响\n\n")
        
        f.write("2. **学习曲线对比** (`*_learning.png`)\n")
        f.write("   - 展示不同超参数值下的训练过程\n")
        f.write("   - 用于观察学习速度和收敛行为的差异\n\n")
        
        f.write("### 2.2 图表列表\n\n")
        f.write("详细图表请查看 `figures/` 目录：\n\n")
        
        for algo in sorted(exps_by_algo.keys()):
            f.write(f"#### {algo}\n\n")
            if algo.upper() in algo_params:
                for param in algo_params[algo.upper()]:
                    safe_name = param.lower().replace(' ', '_')
                    f.write(f"- {param}:\n")
                    f.write(f"  - 敏感性: `a3_{algo.lower()}_{safe_name}_sensitivity.png`\n")
                    f.write(f"  - 学习曲线: `a3_{algo.lower()}_{safe_name}_learning.png`\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        f.write("## 3. 主要发现\n\n")
        f.write("### 3.1 超参数敏感性排序\n\n")
        f.write("*(请根据生成的图表填写关键发现)*\n\n")
        f.write("- **PPO**: ...\n")
        f.write("- **SAC**: ...\n")
        f.write("- **TD3**: ...\n\n")
        
        f.write("### 3.2 最优超参数配置\n\n")
        f.write("基于实验结果，推荐的超参数配置：\n\n")
        f.write("*(请根据敏感性图表填写最优配置)*\n\n")
        
        f.write("---\n\n")
        
        f.write("## 4. 数据表格\n\n")
        f.write(f"完整数据请查看: `a3_detailed_summary.csv`\n\n")
        
        # 展示前10行数据（不使用to_markdown以避免依赖tabulate）
        f.write("### 示例数据（前10行）\n\n")
        f.write("```\n")
        f.write(df.head(10).to_string(index=False))
        f.write("\n```\n\n")
        
        f.write("---\n\n")
        f.write(f"*报告生成时间: {pd.Timestamp.now()}*\n")
    
    print(f"  分析报告: {report_path}")
    
    print("\n" + "="*80)
    print("✅ A3实验组分析完成！")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
