"""A2实验组分析：环境影响

分析不同环境维度（洋流、障碍物、距离）对性能的影响
每个维度单独分析，不混合比较
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.data_loader import scan_experiment_results, group_experiments_by
from analysis.metrics_calculator import calculate_all_metrics, extract_eval_metrics
from analysis.visualization import (
    plot_learning_curves,
    plot_grouped_bars,
    plot_radar_chart,
    plot_success_vs_collision_scatter,
    configure_plot_style,
    ALGORITHM_COLORS,
)


def analyze_current_dimension(experiments, figures_dir, output_dir):
    """分析洋流类型影响 (NC/UC/VC)"""
    
    if not experiments:
        print("\n⚠️ 洋流维度：无实验数据")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 A2.1: 洋流类型影响分析 ({len(experiments)}个实验)")
    print(f"{'='*60}")
    
    exps_with_eval = [e for e in experiments if e.eval_data is not None]
    
    # 识别洋流类型变体（从实验名称提取）
    current_types = {}
    for exp in experiments:
        if '_nc_' in exp.exp_name.lower():
            current_type = 'NC (No Current)'
        elif '_uc_' in exp.exp_name.lower():
            current_type = 'UC (Uniform)'
        elif '_vc_' in exp.exp_name.lower():
            current_type = 'VC (Vortex)'
        else:
            current_type = 'Unknown'
        
        if current_type not in current_types:
            current_types[current_type] = []
        current_types[current_type].append(exp)
    
    print(f"  洋流类型: {list(current_types.keys())}")
    
    # 为实验添加简短显示名称
    for exp in experiments:
        if '_nc_' in exp.exp_name.lower():
            exp.display_name = 'NC (No Current)'
        elif '_uc_' in exp.exp_name.lower():
            exp.display_name = 'UC (Uniform)'
        elif '_vc_' in exp.exp_name.lower():
            exp.display_name = 'VC (Vortex)'
        else:
            exp.display_name = exp.exp_name
    
    # 图1: 学习曲线对比
    print(f"  [1/4] 学习曲线...")
    plot_learning_curves(
        experiments,
        group_by='display_name',
        title='Learning Curves: Ocean Current Types',
        ylabel='Average Return',
        save_path=str(figures_dir / 'a2_current_learning_curves.png'),
        figsize=(10, 6)
    )
    
    # 图2: Eval性能对比柱状图
    if exps_with_eval:
        print(f"  [2/4] Eval性能对比...")
        from matplotlib import pyplot as plt
        
        metrics_to_plot = {
            'Success Rate': 'eval_success_rate',
            'Collision Rate': 'eval_collision_rate',
            'Path Length (m)': 'eval_mean_path_length',
            'Energy': 'eval_mean_energy'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
            ax = axes[idx]
            
            type_names = sorted(current_types.keys())
            values = []
            errors = []
            
            for current_type in type_names:
                type_exps = [e for e in current_types[current_type] if e.eval_data is not None]
                if type_exps:
                    vals = [e.metrics.get(metric_key, 0) for e in type_exps]
                    values.append(np.mean(vals))
                    errors.append(np.std(vals) if len(vals) > 1 else 0)
                else:
                    values.append(0)
                    errors.append(0)
            
            x = np.arange(len(type_names))
            ax.bar(x, values, yerr=errors, capsize=5, alpha=0.7, color='steelblue')
            ax.set_xticks(x)
            ax.set_xticklabels(type_names, rotation=15, ha='right')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Ocean Current Impact on Performance', fontsize=14)
        plt.tight_layout()
        plt.savefig(str(figures_dir / 'a2_current_eval_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图3: 雷达图
        print(f"  [3/4] 多维度雷达图...")
        radar_data = {}
        for current_type, type_exps in current_types.items():
            type_exps_eval = [e for e in type_exps if e.eval_data is not None]
            if type_exps_eval:
                radar_data[current_type] = {
                    'Success': np.mean([e.metrics.get('eval_success_rate', 0) for e in type_exps_eval]),
                    'Safety': 1.0 - np.mean([e.metrics.get('eval_collision_rate', 0) for e in type_exps_eval]),
                    'Path_Eff': 1.0 / (np.mean([e.metrics.get('eval_mean_path_length', 1) for e in type_exps_eval]) + 1e-6),
                    'Smoothness': 1.0 - np.mean([e.metrics.get('eval_mean_smoothness', 0) for e in type_exps_eval]),  # 反转：越小越好变为越大越好
                    'Energy_Eff': 1.0 / (np.mean([e.metrics.get('eval_mean_energy', 1) for e in type_exps_eval]) + 1e-6),
                }
        
        # 归一化
        categories = ['Success', 'Safety', 'Path_Eff', 'Smoothness', 'Energy_Eff']
        for cat in categories:
            max_val = max(radar_data[t][cat] for t in radar_data.keys())
            if max_val > 0:
                for t in radar_data.keys():
                    radar_data[t][cat] = radar_data[t][cat] / max_val
        
        plot_radar_chart(
            radar_data,
            categories=categories,
            title='Multi-Dimensional Comparison: Ocean Current Types',
            save_path=str(figures_dir / 'a2_current_radar.png'),
            figsize=(8, 8)
        )
        
        # 图4: 样本效率对比
        print(f"  [4/4] 样本效率...")
        eff_data = {}
        for current_type, type_exps in current_types.items():
            eff_data[current_type] = {}
            eff_data[current_type]['SAC'] = np.mean([e.metrics['sample_efficiency'] for e in type_exps]) / 1000
        
        plot_grouped_bars(
            eff_data,
            title='Sample Efficiency: Ocean Current Types',
            xlabel='Current Type',
            ylabel='Steps to 80% (K)',
            save_path=str(figures_dir / 'a2_current_sample_efficiency.png'),
            figsize=(8, 6)
        )
    
    print(f"  ✅ 洋流类型分析完成")


def analyze_obstacle_dimension(experiments, figures_dir, output_dir):
    """分析障碍物密度影响 (SP/MD/DN/MZ)"""
    
    if not experiments:
        print("\n⚠️ 障碍物维度：无实验数据")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 A2.2: 障碍物密度影响分析 ({len(experiments)}个实验)")
    print(f"{'='*60}")
    
    exps_with_eval = [e for e in experiments if e.eval_data is not None]
    
    # 识别障碍物密度类型
    obs_types = {}
    for exp in experiments:
        if '_sp_' in exp.exp_name.lower():
            obs_type = 'SP (Sparse)'
        elif '_md_' in exp.exp_name.lower():
            obs_type = 'MD (Medium)'
        elif '_dn_' in exp.exp_name.lower():
            obs_type = 'DN (Dense)'
        elif '_mz_' in exp.exp_name.lower():
            obs_type = 'MZ (Maze)'
        else:
            obs_type = 'Unknown'
        
        if obs_type not in obs_types:
            obs_types[obs_type] = []
        obs_types[obs_type].append(exp)
    
    print(f"  障碍物类型: {list(obs_types.keys())}")
    
    # 为实验添加简短显示名称
    for exp in experiments:
        if '_sp_' in exp.exp_name.lower():
            exp.display_name = 'SP (Sparse)'
        elif '_md_' in exp.exp_name.lower():
            exp.display_name = 'MD (Medium)'
        elif '_dn_' in exp.exp_name.lower():
            exp.display_name = 'DN (Dense)'
        elif '_mz_' in exp.exp_name.lower():
            exp.display_name = 'MZ (Maze)'
        else:
            exp.display_name = exp.exp_name
    
    # 图1: 学习曲线
    print(f"  [1/4] 学习曲线...")
    plot_learning_curves(
        experiments,
        group_by='display_name',
        title='Learning Curves: Obstacle Densities',
        ylabel='Average Return',
        save_path=str(figures_dir / 'a2_obstacle_learning_curves.png'),
        figsize=(10, 6)
    )
    
    # 图2-4: 类似洋流分析
    if exps_with_eval:
        print(f"  [2/4] Eval性能对比...")
        from matplotlib import pyplot as plt
        
        metrics_to_plot = {
            'Success Rate': 'eval_success_rate',
            'Collision Rate': 'eval_collision_rate',
            'Path Length (m)': 'eval_mean_path_length',
            'Energy': 'eval_mean_energy'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
            ax = axes[idx]
            
            type_names = sorted(obs_types.keys())
            values = []
            errors = []
            
            for obs_type in type_names:
                type_exps = [e for e in obs_types[obs_type] if e.eval_data is not None]
                if type_exps:
                    vals = [e.metrics.get(metric_key, 0) for e in type_exps]
                    values.append(np.mean(vals))
                    errors.append(np.std(vals) if len(vals) > 1 else 0)
                else:
                    values.append(0)
                    errors.append(0)
            
            x = np.arange(len(type_names))
            ax.bar(x, values, yerr=errors, capsize=5, alpha=0.7, color='coral')
            ax.set_xticks(x)
            ax.set_xticklabels(type_names, rotation=15, ha='right')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Obstacle Density Impact on Performance', fontsize=14)
        plt.tight_layout()
        plt.savefig(str(figures_dir / 'a2_obstacle_eval_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 雷达图
        print(f"  [3/4] 多维度雷达图...")
        radar_data = {}
        for obs_type, type_exps in obs_types.items():
            type_exps_eval = [e for e in type_exps if e.eval_data is not None]
            if type_exps_eval:
                radar_data[obs_type] = {
                    'Success': np.mean([e.metrics.get('eval_success_rate', 0) for e in type_exps_eval]),
                    'Safety': 1.0 - np.mean([e.metrics.get('eval_collision_rate', 0) for e in type_exps_eval]),
                    'Path_Eff': 1.0 / (np.mean([e.metrics.get('eval_mean_path_length', 1) for e in type_exps_eval]) + 1e-6),
                    'Smoothness': 1.0 - np.mean([e.metrics.get('eval_mean_smoothness', 0) for e in type_exps_eval]),  # 反转：越小越好变为越大越好
                    'Energy_Eff': 1.0 / (np.mean([e.metrics.get('eval_mean_energy', 1) for e in type_exps_eval]) + 1e-6),
                }
        
        categories = ['Success', 'Safety', 'Path_Eff', 'Smoothness', 'Energy_Eff']
        for cat in categories:
            max_val = max(radar_data[t][cat] for t in radar_data.keys())
            if max_val > 0:
                for t in radar_data.keys():
                    radar_data[t][cat] = radar_data[t][cat] / max_val
        
        plot_radar_chart(
            radar_data,
            categories=categories,
            title='Multi-Dimensional Comparison: Obstacle Densities',
            save_path=str(figures_dir / 'a2_obstacle_radar.png'),
            figsize=(8, 8)
        )
        
        # 样本效率
        print(f"  [4/4] 样本效率...")
        eff_data = {}
        for obs_type, type_exps in obs_types.items():
            eff_data[obs_type] = {}
            eff_data[obs_type]['SAC'] = np.mean([e.metrics['sample_efficiency'] for e in type_exps]) / 1000
        
        plot_grouped_bars(
            eff_data,
            title='Sample Efficiency: Obstacle Densities',
            xlabel='Obstacle Density',
            ylabel='Steps to 80% (K)',
            save_path=str(figures_dir / 'a2_obstacle_sample_efficiency.png'),
            figsize=(8, 6)
        )
    
    print(f"  ✅ 障碍物密度分析完成")


def analyze_distance_dimension(experiments, figures_dir, output_dir):
    """分析目标距离影响 (SD/MD/LD)"""
    
    if not experiments:
        print("\n⚠️ 距离维度：无实验数据")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 A2.3: 目标距离影响分析 ({len(experiments)}个实验)")
    print(f"{'='*60}")
    
    exps_with_eval = [e for e in experiments if e.eval_data is not None]
    
    # 识别距离类型
    dist_types = {}
    for exp in experiments:
        if '_sd_' in exp.exp_name.lower():
            dist_type = 'SD (Short)'
        elif '_md_' in exp.exp_name.lower():
            dist_type = 'MD (Medium)'
        elif '_ld_' in exp.exp_name.lower():
            dist_type = 'LD (Long)'
        else:
            dist_type = 'Unknown'
        
        if dist_type not in dist_types:
            dist_types[dist_type] = []
        dist_types[dist_type].append(exp)
    
    print(f"  距离类型: {list(dist_types.keys())}")
    
    # 为实验添加简短显示名称
    for exp in experiments:
        if '_sd_' in exp.exp_name.lower():
            exp.display_name = 'SD (Short)'
        elif '_md_' in exp.exp_name.lower():
            exp.display_name = 'MD (Medium)'
        elif '_ld_' in exp.exp_name.lower():
            exp.display_name = 'LD (Long)'
        else:
            exp.display_name = exp.exp_name
    
    # 图1: 学习曲线
    print(f"  [1/4] 学习曲线...")
    plot_learning_curves(
        experiments,
        group_by='display_name',
        title='Learning Curves: Goal Distances',
        ylabel='Average Return',
        save_path=str(figures_dir / 'a2_distance_learning_curves.png'),
        figsize=(10, 6)
    )
    
    # 图2-4: 类似前面分析
    if exps_with_eval:
        print(f"  [2/4] Eval性能对比...")
        from matplotlib import pyplot as plt
        
        metrics_to_plot = {
            'Success Rate': 'eval_success_rate',
            'Collision Rate': 'eval_collision_rate',
            'Path Length (m)': 'eval_mean_path_length',
            'Energy': 'eval_mean_energy'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
            ax = axes[idx]
            
            type_names = sorted(dist_types.keys())
            values = []
            errors = []
            
            for dist_type in type_names:
                type_exps = [e for e in dist_types[dist_type] if e.eval_data is not None]
                if type_exps:
                    vals = [e.metrics.get(metric_key, 0) for e in type_exps]
                    values.append(np.mean(vals))
                    errors.append(np.std(vals) if len(vals) > 1 else 0)
                else:
                    values.append(0)
                    errors.append(0)
            
            x = np.arange(len(type_names))
            ax.bar(x, values, yerr=errors, capsize=5, alpha=0.7, color='seagreen')
            ax.set_xticks(x)
            ax.set_xticklabels(type_names, rotation=15, ha='right')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Goal Distance Impact on Performance', fontsize=14)
        plt.tight_layout()
        plt.savefig(str(figures_dir / 'a2_distance_eval_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 雷达图
        print(f"  [3/4] 多维度雷达图...")
        radar_data = {}
        for dist_type, type_exps in dist_types.items():
            type_exps_eval = [e for e in type_exps if e.eval_data is not None]
            if type_exps_eval:
                radar_data[dist_type] = {
                    'Success': np.mean([e.metrics.get('eval_success_rate', 0) for e in type_exps_eval]),
                    'Safety': 1.0 - np.mean([e.metrics.get('eval_collision_rate', 0) for e in type_exps_eval]),
                    'Path_Eff': 1.0 / (np.mean([e.metrics.get('eval_mean_path_length', 1) for e in type_exps_eval]) + 1e-6),
                    'Smoothness': 1.0 - np.mean([e.metrics.get('eval_mean_smoothness', 0) for e in type_exps_eval]),  # 反转：越小越好变为越大越好
                    'Energy_Eff': 1.0 / (np.mean([e.metrics.get('eval_mean_energy', 1) for e in type_exps_eval]) + 1e-6),
                }
        
        categories = ['Success', 'Safety', 'Path_Eff', 'Smoothness', 'Energy_Eff']
        for cat in categories:
            max_val = max(radar_data[t][cat] for t in radar_data.keys())
            if max_val > 0:
                for t in radar_data.keys():
                    radar_data[t][cat] = radar_data[t][cat] / max_val
        
        plot_radar_chart(
            radar_data,
            categories=categories,
            title='Multi-Dimensional Comparison: Goal Distances',
            save_path=str(figures_dir / 'a2_distance_radar.png'),
            figsize=(8, 8)
        )
        
        # 样本效率
        print(f"  [4/4] 样本效率...")
        eff_data = {}
        for dist_type, type_exps in dist_types.items():
            eff_data[dist_type] = {}
            eff_data[dist_type]['SAC'] = np.mean([e.metrics['sample_efficiency'] for e in type_exps]) / 1000
        
        plot_grouped_bars(
            eff_data,
            title='Sample Efficiency: Goal Distances',
            xlabel='Goal Distance',
            ylabel='Steps to 80% (K)',
            save_path=str(figures_dir / 'a2_distance_sample_efficiency.png'),
            figsize=(8, 6)
        )
    
    print(f"  ✅ 目标距离分析完成")


def generate_comprehensive_report(experiments, exps_by_dimension, output_dir, figures_dir):
    """生成综合分析报告"""
    
    print(f"\n📝 生成综合分析报告...")
    
    report_path = output_dir / 'a2_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# A2实验组分析报告：环境影响\n\n")
        f.write("**实验目标**: 分析不同环境维度（洋流、障碍物、距离）对性能的独立影响\n\n")
        f.write("**算法**: SAC（固定算法，控制变量）\n\n")
        f.write("---\n\n")
        
        f.write("## 1. 实验概览\n\n")
        f.write(f"- **总实验数**: {len(experiments)}\n")
        f.write(f"- **研究维度**: 3个（洋流类型、障碍物密度、目标距离）\n")
        f.write(f"- **实验方法**: 单变量控制，逐一分析各维度影响\n\n")
        
        for dim, exps in sorted(exps_by_dimension.items()):
            f.write(f"- **{dim}维度**: {len(exps)}个实验\n")
        
        f.write("\n---\n\n")
        
        # A2.1: 洋流类型影响
        f.write("## 2. A2.1: 洋流类型影响分析\n\n")
        f.write("**研究问题**: 不同洋流环境（无洋流/均匀洋流/涡旋洋流）如何影响性能？\n\n")
        
        f.write("### 2.1.1 学习曲线\n\n")
        f.write("![Current Learning Curves](figures/a2_current_learning_curves.png)\n\n")
        
        f.write("### 2.1.2 性能对比\n\n")
        f.write("![Current Performance](figures/a2_current_eval_performance.png)\n\n")
        f.write("*对比：成功率、碰撞率、路径长度、能量消耗*\n\n")
        
        f.write("### 2.1.3 多维度对比\n\n")
        f.write("![Current Radar](figures/a2_current_radar.png)\n\n")
        
        f.write("### 2.1.4 样本效率\n\n")
        f.write("![Current Efficiency](figures/a2_current_sample_efficiency.png)\n\n")
        
        f.write("---\n\n")
        
        # A2.2: 障碍物密度影响
        f.write("## 3. A2.2: 障碍物密度影响分析\n\n")
        f.write("**研究问题**: 不同障碍物密度（稀疏/中等/密集/迷宫）如何影响性能？\n\n")
        
        f.write("### 3.1 学习曲线\n\n")
        f.write("![Obstacle Learning Curves](figures/a2_obstacle_learning_curves.png)\n\n")
        
        f.write("### 3.2 性能对比\n\n")
        f.write("![Obstacle Performance](figures/a2_obstacle_eval_performance.png)\n\n")
        
        f.write("### 3.3 多维度对比\n\n")
        f.write("![Obstacle Radar](figures/a2_obstacle_radar.png)\n\n")
        
        f.write("### 3.4 样本效率\n\n")
        f.write("![Obstacle Efficiency](figures/a2_obstacle_sample_efficiency.png)\n\n")
        
        f.write("---\n\n")
        
        # A2.3: 目标距离影响
        f.write("## 4. A2.3: 目标距离影响分析\n\n")
        f.write("**研究问题**: 不同目标距离（短/中/长）如何影响性能？\n\n")
        
        f.write("### 4.1 学习曲线\n\n")
        f.write("![Distance Learning Curves](figures/a2_distance_learning_curves.png)\n\n")
        
        f.write("### 4.2 性能对比\n\n")
        f.write("![Distance Performance](figures/a2_distance_eval_performance.png)\n\n")
        
        f.write("### 4.3 多维度对比\n\n")
        f.write("![Distance Radar](figures/a2_distance_radar.png)\n\n")
        
        f.write("### 4.4 样本效率\n\n")
        f.write("![Distance Efficiency](figures/a2_distance_sample_efficiency.png)\n\n")
        
        f.write("---\n\n")
        
        # 综合对比
        exps_with_eval = [e for e in experiments if e.eval_data is not None]
        if exps_with_eval:
            f.write("## 5. 综合对比\n\n")
            f.write("### 5.1 成功率vs安全性权衡\n\n")
            f.write("![All Dimensions Scatter](figures/a2_all_dimensions_scatter.png)\n\n")
            f.write("*展示三个维度所有实验在成功率与碰撞率空间的分布*\n\n")
        
        f.write("---\n\n")
        
        f.write("## 6. 结论与建议\n\n")
        f.write("1. **洋流影响**: 待分析结果填充\n")
        f.write("2. **障碍物影响**: 待分析结果填充\n")
        f.write("3. **距离影响**: 待分析结果填充\n")
        f.write("4. **综合建议**: 基于单变量分析结果，提供环境设计建议\n\n")
        
        f.write("---\n\n")
        f.write(f"*报告生成时间: {pd.Timestamp.now()}*\n")
    
    print(f"  分析报告: {report_path}")
    
    # 生成汇总表
    summary_data = []
    for exp in experiments:
        row = {
            'Experiment': exp.exp_name,
            'Dimension': exp.env_dimension,
            'Final_Mean': exp.metrics['final_mean'],
            'Sample_Efficiency': exp.metrics['sample_efficiency'],
        }
        if exp.eval_data:
            row.update({
                'Eval_Success': exp.metrics.get('eval_success_rate', np.nan),
                'Eval_Collision': exp.metrics.get('eval_collision_rate', np.nan),
                'Eval_Path_Length': exp.metrics.get('eval_mean_path_length', np.nan),
            })
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    summary_csv = output_dir / 'a2_detailed_summary.csv'
    df.to_csv(summary_csv, index=False, float_format='%.4f')
    print(f"  汇总表: {summary_csv}")


def main():
    """主函数：A2环境影响分析"""
    
    print("="*80)
    print("A2 实验组分析：环境影响分析（分维度独立分析）")
    print("="*80)
    
    # 配置路径
    results_dir = Path(__file__).parent.parent / 'results'
    a2_dir = results_dir / 'a2'
    output_dir = a2_dir / 'analysis'
    figures_dir = output_dir / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 加载实验数据...")
    experiments = scan_experiment_results(str(results_dir), experiment_group='a2')
    
    if not experiments:
        print("❌ 未找到A2实验数据！")
        return
    
    print(f"\n✅ 成功加载 {len(experiments)} 个实验")
    
    # 计算指标（包括eval数据）
    print(f"\n📊 计算性能指标...")
    for exp in experiments:
        metrics = calculate_all_metrics(exp.timesteps, exp.results, exp.ep_lengths)
        exp.metrics = metrics
        
        # 如果有eval数据，提取并添加
        if exp.eval_data is not None:
            eval_metrics = extract_eval_metrics(exp.eval_data)
            exp.metrics.update(eval_metrics)
            print(f"  {exp.exp_name}: eval_success={eval_metrics.get('eval_success_rate', 0):.2%}")
        else:
            print(f"  {exp.exp_name}: [no eval data]")
    
    exps_with_eval = [e for e in experiments if e.eval_data is not None]
    print(f"\n  ✅ {len(exps_with_eval)}/{len(experiments)} 个实验包含eval数据")
    
    # 按环境维度分组
    exps_by_dimension = group_experiments_by(experiments, by='env_dimension')
    
    print(f"\n按环境维度分组:")
    for dim, exps in sorted(exps_by_dimension.items()):
        print(f"  {dim}: {len(exps)} 个实验")
    
    configure_plot_style()
    
    # ========== 分析各维度 ==========
    
    # A2.1: 洋流类型影响
    analyze_current_dimension(
        exps_by_dimension.get('current', []),
        figures_dir,
        output_dir
    )
    
    # A2.2: 障碍物密度影响
    analyze_obstacle_dimension(
        exps_by_dimension.get('obstacle', []),
        figures_dir,
        output_dir
    )
    
    # A2.3: 目标距离影响
    analyze_distance_dimension(
        exps_by_dimension.get('distance', []),
        figures_dir,
        output_dir
    )
    
    # 综合分析（仅散点图）
    if exps_with_eval:
        print(f"\n🎨 生成综合对比图...")
        plot_success_vs_collision_scatter(
            exps_with_eval,
            group_by='env_dimension',
            title='Success Rate vs Collision Rate (All Dimensions)',
            save_path=str(figures_dir / 'a2_all_dimensions_scatter.png'),
            figsize=(10, 8)
        )
    
    # 生成综合报告
    generate_comprehensive_report(
        experiments,
        exps_by_dimension,
        output_dir,
        figures_dir
    )
    
    print("\n" + "="*80)
    print("✅ A2实验组分析完成！")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
