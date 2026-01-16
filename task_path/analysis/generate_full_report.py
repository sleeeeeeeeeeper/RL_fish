"""总报告生成脚本

整合所有实验组的分析结果，生成完整的实验报告
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_analysis_script(script_name: str) -> bool:
    """运行分析脚本
    
    Args:
        script_name: 脚本名称
    
    Returns:
        是否成功
    """
    script_path = Path(__file__).parent / script_name
    print(f"\n{'='*80}")
    print(f"运行: {script_name}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} 运行失败: {e}")
        return False
    except Exception as e:
        print(f"❌ {script_name} 运行异常: {e}")
        return False


def main():
    """主函数：生成完整实验报告"""
    
    print("\n" + "="*80)
    print("生成完整实验报告")
    print("="*80)
    
    results_dir = Path(__file__).parent.parent / 'results'
    output_dir = results_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 运行所有分析脚本
    print("\n[步骤 1/3] 运行所有实验组分析脚本...\n")
    
    analysis_scripts = [
        'analyze_a1_algorithms.py',
        'analyze_a2_environment.py',
        'analyze_a3_hyperparams.py',
        'analyze_a4_rewards.py',
    ]
    
    results = {}
    for script in analysis_scripts:
        success = run_analysis_script(script)
        results[script] = success
    
    # 2. 汇总所有结果
    print("\n" + "="*80)
    print("[步骤 2/3] 汇总所有实验结果...")
    print("="*80)
    
    all_summaries = []
    
    for exp_group in ['a1', 'a2', 'a3', 'a4']:
        summary_file = results_dir / exp_group / 'analysis' / f'{exp_group}_summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            df['Experiment_Group'] = exp_group.upper()
            all_summaries.append(df)
            print(f"  ✓ 加载 {exp_group.upper()} 汇总表: {len(df)} 个实验")
        else:
            # 尝试详细汇总表
            detailed_file = results_dir / exp_group / 'analysis' / f'{exp_group}_detailed_summary.csv'
            if detailed_file.exists():
                df = pd.read_csv(detailed_file)
                df['Experiment_Group'] = exp_group.upper()
                all_summaries.append(df)
                print(f"  ✓ 加载 {exp_group.upper()} 详细汇总表: {len(df)} 个实验")
    
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        combined_csv = output_dir / 'all_experiments_summary.csv'
        combined_df.to_csv(combined_csv, index=False, float_format='%.4f')
        print(f"\n  📊 总计: {len(combined_df)} 个实验")
        print(f"  💾 保存到: {combined_csv}")
    else:
        print("\n  ⚠️  没有找到汇总表")
        combined_df = None
    
    # 3. 生成最终报告
    print("\n" + "="*80)
    print("[步骤 3/3] 生成最终实验报告...")
    print("="*80)
    
    report_path = output_dir / 'full_experiment_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 基于强化学习的仿生鱼路径规划实验报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 实验概要\n\n")
        
        if combined_df is not None:
            total_exps = len(combined_df)
            exp_groups = combined_df['Experiment_Group'].unique()
            
            f.write(f"- **总实验数**: {total_exps}\n")
            f.write(f"- **实验组**: {', '.join(sorted(exp_groups))}\n")
            
            if 'Algorithm' in combined_df.columns:
                algorithms = combined_df['Algorithm'].dropna().unique()
                f.write(f"- **测试算法**: {', '.join(sorted(algorithms))}\n")
            
            f.write("\n")
        else:
            f.write("*数据汇总未完成*\n\n")
        
        f.write("---\n\n")
        
        f.write("## 实验组详细报告\n\n")
        
        # A1: 算法对比
        f.write("### A1: 算法对比分析\n\n")
        a1_report = results_dir / 'a1' / 'analysis' / 'a1_report.md'
        if a1_report.exists():
            f.write(f"📄 [详细报告](../a1/analysis/a1_report.md)\n\n")
            f.write(f"**关键发现**:\n")
            f.write(f"- 在5个难度级别下对比PPO、SAC、TD3算法\n")
            f.write(f"- 生成了学习曲线、性能对比、雷达图等5个可视化图表\n\n")
        else:
            f.write("*报告生成中...*\n\n")
        
        # A2: 环境影响
        f.write("### A2: 环境影响分析\n\n")
        a2_report = results_dir / 'a2' / 'analysis' / 'a2_report.md'
        if a2_report.exists():
            f.write(f"📄 [详细报告](../a2/analysis/a2_report.md)\n\n")
            f.write(f"**关键发现**:\n")
            f.write(f"- 分析了洋流、障碍物、距离等环境维度的影响\n\n")
        else:
            f.write("*报告生成中...*\n\n")
        
        # A3: 超参数
        f.write("### A3: 超参数敏感性分析\n\n")
        a3_report = results_dir / 'a3' / 'analysis' / 'a3_report.md'
        if a3_report.exists():
            f.write(f"📄 [详细报告](../a3/analysis/a3_report.md)\n\n")
            f.write(f"**关键发现**:\n")
            f.write(f"- 测试了学习率、批量大小等关键超参数\n\n")
        else:
            f.write("*报告生成中...*\n\n")
        
        # A4: 奖励函数
        f.write("### A4: 奖励函数对比分析\n\n")
        a4_report = results_dir / 'a4' / 'analysis' / 'a4_report.md'
        if a4_report.exists():
            f.write(f"📄 [详细报告](../a4/analysis/a4_report.md)\n\n")
            f.write(f"**关键发现**:\n")
            f.write(f"- 对比了不同奖励函数设计的效果\n\n")
        else:
            f.write("*报告生成中...*\n\n")
        
        f.write("---\n\n")
        
        f.write("## 研究问题回答\n\n")
        
        f.write("### RQ1: 在相同环境下，哪种算法性能最优？\n\n")
        f.write("**答案**: 详见 A1 算法对比分析报告\n\n")
        
        f.write("### RQ2: 相同算法在不同难度环境下的性能衰减规律？\n\n")
        f.write("**答案**: 详见 A1 和 A2 分析报告\n\n")
        
        f.write("### RQ3: 超参数如何影响算法性能？\n\n")
        f.write("**答案**: 详见 A3 超参数敏感性分析报告\n\n")
        
        f.write("### RQ4: 奖励函数设计对训练效果的影响？\n\n")
        f.write("**答案**: 详见 A4 奖励函数对比分析报告\n\n")
        
        f.write("---\n\n")
        
        f.write("## 文件组织\n\n")
        f.write("```\n")
        f.write("results/\n")
        f.write("├── a1/analysis/          # A1 算法对比\n")
        f.write("├── a2/analysis/          # A2 环境影响\n")
        f.write("├── a3/analysis/          # A3 超参数\n")
        f.write("├── a4/analysis/          # A4 奖励函数\n")
        f.write("└── analysis/             # 总体分析\n")
        f.write("    ├── all_experiments_summary.csv\n")
        f.write("    └── full_experiment_report.md (本文件)\n")
        f.write("```\n\n")
        
        f.write("---\n\n")
        f.write("*报告结束*\n")
    
    print(f"\n  💾 最终报告: {report_path}")
    
    # 显示运行结果摘要
    print("\n" + "="*80)
    print("运行结果摘要")
    print("="*80)
    
    for script, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {script:30s} {status}")
    
    print("\n" + "="*80)
    print("✅ 完整实验报告生成完成！")
    print("="*80)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"📄 主报告: {output_dir}/full_experiment_report.md")
    print()


if __name__ == '__main__':
    main()
