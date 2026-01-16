"""可视化模块

提供各种绘图函数用于数据分析和报告生成
"""

import os
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib defaults")


# 配置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def configure_plot_style(style: str = 'seaborn-v0_8-darkgrid'):
    """配置绘图样式
    
    Args:
        style: matplotlib样式名称
    """
    try:
        plt.style.use(style)
    except:
        # Fallback to ggplot style if seaborn styles not available
        try:
            plt.style.use('ggplot')
        except:
            plt.style.use('default')
    
    # 设置seaborn调色板（如果可用）
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # 设置默认字体大小
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
    })


# 算法颜色映射
ALGORITHM_COLORS = {
    'PPO': '#1f77b4',  # 蓝色
    'SAC': '#ff7f0e',  # 橙色
    'TD3': '#2ca02c',  # 绿色
}

# 难度颜色映射
DIFFICULTY_COLORS = {
    'L1': '#d4eac7',  # 浅绿
    'L2': '#a6d96a',  # 绿
    'L3': '#ffcc99',  # 浅橙
    'L4': '#ff9966',  # 橙
    'L5': '#cc5544',  # 红
}


def plot_learning_curves(experiments: List[Any],
                         group_by: str = 'algorithm',
                         metric: str = 'mean',
                         title: str = 'Learning Curves',
                         xlabel: str = 'Training Steps',
                         ylabel: str = 'Average Return',
                         save_path: Optional[str] = None,
                         show_std: bool = True,
                         figsize: Tuple[int, int] = (10, 6)):
    """绘制学习曲线
    
    Args:
        experiments: ExperimentData对象列表
        group_by: 分组依据 ('algorithm', 'difficulty', etc.)
        metric: 绘制的指标 ('mean', 'max', 'min')
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径
        show_std: 是否显示标准差区域
        figsize: 图表大小
    """
    configure_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # 按指定字段分组
    groups = {}
    for exp in experiments:
        key = getattr(exp, group_by, 'unknown')
        if key is None:
            key = 'unknown'
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)
    
    # 为每组绘制曲线
    for group_name, group_exps in sorted(groups.items()):
        if not group_exps:
            continue
        
        # 获取颜色
        if group_by == 'algorithm':
            color = ALGORITHM_COLORS.get(group_name, None)
        elif group_by == 'env_difficulty':
            color = DIFFICULTY_COLORS.get(group_name, None)
        else:
            color = None
        
        # 平均所有实验的学习曲线
        all_timesteps = []
        all_means = []
        all_stds = []
        
        for exp in group_exps:
            all_timesteps.append(exp.timesteps)
            all_means.append(exp.get_mean_results())
            all_stds.append(exp.get_std_results())
        
        # 找到最短长度（对齐）
        min_len = min(len(ts) for ts in all_timesteps)
        
        # 裁剪到相同长度
        timesteps = all_timesteps[0][:min_len]
        means = np.array([m[:min_len] for m in all_means])
        stds = np.array([s[:min_len] for s in all_stds])
        
        # 计算平均和标准差
        mean_curve = np.mean(means, axis=0)
        std_curve = np.std(means, axis=0)  # 跨实验的标准差
        
        # 绘制主曲线
        ax.plot(timesteps, mean_curve, label=group_name, color=color, linewidth=2)
        
        # 绘制标准差区域
        if show_std and len(group_exps) > 1:
            ax.fill_between(timesteps,
                           mean_curve - std_curve,
                           mean_curve + std_curve,
                           alpha=0.2,
                           color=color)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_bar_comparison(data: Dict[str, float],
                       title: str = 'Performance Comparison',
                       xlabel: str = 'Algorithms',
                       ylabel: str = 'Performance',
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (8, 6),
                       color_map: Optional[Dict[str, str]] = None):
    """绘制柱状图对比
    
    Args:
        data: {名称: 值} 字典
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径
        figsize: 图表大小
        color_map: 颜色映射
    """
    configure_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(data.keys())
    values = list(data.values())
    
    # 确定颜色
    if color_map:
        colors = [color_map.get(name, '#1f77b4') for name in names]
    else:
        if HAS_SEABORN:
            colors = sns.color_palette("husl", len(names))
        else:
            # Use matplotlib's default color cycle
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color'][:len(names)]
    
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # 在柱子上添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}',
               ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_grouped_bars(data: Dict[str, Dict[str, float]],
                     title: str = 'Grouped Performance Comparison',
                     xlabel: str = 'Difficulty',
                     ylabel: str = 'Success Rate',
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 6)):
    """绘制分组柱状图
    
    Args:
        data: {组名: {子项名: 值}} 嵌套字典
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径
        figsize: 图表大小
    """
    configure_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = list(data.keys())
    subgroups = list(next(iter(data.values())).keys())
    
    x = np.arange(len(groups))
    width = 0.8 / len(subgroups)
    
    for i, subgroup in enumerate(subgroups):
        values = [data[g].get(subgroup, 0) for g in groups]
        offset = (i - len(subgroups)/2 + 0.5) * width
        color = ALGORITHM_COLORS.get(subgroup, None)
        ax.bar(x + offset, values, width, label=subgroup, color=color, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_heatmap(data: np.ndarray,
                xticklabels: List[str],
                yticklabels: List[str],
                title: str = 'Heatmap',
                xlabel: str = 'X',
                ylabel: str = 'Y',
                cmap: str = 'YlOrRd',
                save_path: Optional[str] = None,
                figsize: Tuple[int, int] = (10, 8),
                annot: bool = True,
                fmt: str = '.2f'):
    """绘制热力图
    
    Args:
        data: 2D数组 shape: (n_y, n_x)
        xticklabels: X轴标签
        yticklabels: Y轴标签
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        cmap: 颜色映射
        save_path: 保存路径
        figsize: 图表大小
        annot: 是否显示数值
        fmt: 数值格式
    """
    configure_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if HAS_SEABORN:
        # Use seaborn for prettier heatmaps
        sns.heatmap(data, 
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    annot=annot,
                    fmt=fmt,
                    cmap=cmap,
                    cbar_kws={'label': 'Value'},
                    ax=ax,
                    linewidths=0.5,
                    linecolor='gray')
    else:
        # Use matplotlib imshow
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value')
        
        # Add annotations if requested
        if annot:
            for i in range(len(yticklabels)):
                for j in range(len(xticklabels)):
                    text = ax.text(j, i, format(data[i, j], fmt),
                                 ha="center", va="center", color="black")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_radar_chart(data: Dict[str, Dict[str, float]],
                    categories: List[str],
                    title: str = 'Radar Chart',
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (8, 8)):
    """绘制雷达图
    
    Args:
        data: {名称: {类别: 值}} 字典
        categories: 类别列表（按顺序）
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    configure_plot_style()
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    for name, values_dict in data.items():
        values = [values_dict.get(cat, 0) for cat in categories]
        values += values[:1]  # 闭合
        
        color = ALGORITHM_COLORS.get(name, None)
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_box_plot(data: Dict[str, List[float]],
                 title: str = 'Box Plot',
                 xlabel: str = 'Category',
                 ylabel: str = 'Value',
                 save_path: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6)):
    """绘制箱线图
    
    Args:
        data: {名称: [值列表]} 字典
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径
        figsize: 图表大小
    """
    configure_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(data.keys())
    values_list = [data[name] for name in names]
    
    bp = ax.boxplot(values_list, labels=names, patch_artist=True)
    
    # 设置颜色
    for patch, name in zip(bp['boxes'], names):
        color = ALGORITHM_COLORS.get(name, '#1f77b4')
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_multi_panel_learning_curves(experiments_by_difficulty: Dict[str, List[Any]],
                                     algorithms: List[str],
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (15, 10)):
    """绘制多面板学习曲线（算法×难度）
    
    Args:
        experiments_by_difficulty: {难度: [实验列表]} 字典
        algorithms: 算法列表
        save_path: 保存路径
        figsize: 图表大小
    """
    configure_plot_style()
    
    difficulties = sorted(experiments_by_difficulty.keys())
    n_rows = (len(difficulties) + 2) // 3  # 3列布局
    n_cols = min(3, len(difficulties))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for idx, difficulty in enumerate(difficulties):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        exps = experiments_by_difficulty[difficulty]
        
        # 按算法分组
        for algo in algorithms:
            algo_exps = [e for e in exps if e.algorithm == algo]
            
            if not algo_exps:
                continue
            
            # 平均学习曲线
            all_timesteps = [e.timesteps for e in algo_exps]
            all_means = [e.get_mean_results() for e in algo_exps]
            
            min_len = min(len(ts) for ts in all_timesteps)
            timesteps = all_timesteps[0][:min_len]
            means = np.array([m[:min_len] for m in all_means])
            
            mean_curve = np.mean(means, axis=0)
            std_curve = np.std(means, axis=0)
            
            color = ALGORITHM_COLORS.get(algo, None)
            ax.plot(timesteps, mean_curve, label=algo, color=color, linewidth=2)
            
            if len(algo_exps) > 1:
                ax.fill_between(timesteps,
                               mean_curve - std_curve,
                               mean_curve + std_curve,
                               alpha=0.2,
                               color=color)
        
        ax.set_title(f'Difficulty: {difficulty}')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Average Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(len(difficulties), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Learning Curves by Difficulty Level', fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_eval_metrics_comparison(experiments: List[Any],
                                 group_by: str = 'algorithm',
                                 metrics: List[str] = None,
                                 title: str = 'Evaluation Metrics Comparison',
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (14, 10)):
    """绘制eval.py生成的评估指标对比图
    
    Args:
        experiments: ExperimentData对象列表
        group_by: 分组依据 ('algorithm', 'difficulty')
        metrics: 要绘制的指标列表，None表示全部
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    # 默认指标
    if metrics is None:
        metrics = [
            'eval_success_rate',
            'eval_collision_rate',
            'eval_mean_path_length',
            'eval_mean_energy',
            'eval_mean_smoothness',
            'eval_mean_min_obstacle_dist'
        ]
    
    # 过滤出有eval数据的实验
    exps_with_eval = [e for e in experiments if e.eval_data is not None]
    
    if not exps_with_eval:
        print("Warning: No experiments with eval data found")
        return
    
    configure_plot_style()
    
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # 按group_by分组
    groups = {}
    for exp in exps_with_eval:
        key = getattr(exp, group_by, 'unknown')
        if key is None:
            key = 'unknown'
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)
    
    # 为每个指标绘制柱状图
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        group_names = sorted(groups.keys())
        values = []
        errors = []
        
        for group_name in group_names:
            group_exps = groups[group_name]
            metric_values = [exp.metrics.get(metric, 0.0) for exp in group_exps if metric in exp.metrics]
            
            if metric_values:
                values.append(np.mean(metric_values))
                errors.append(np.std(metric_values) if len(metric_values) > 1 else 0.0)
            else:
                values.append(0.0)
                errors.append(0.0)
        
        # 绘制柱状图
        x_pos = np.arange(len(group_names))
        colors = [ALGORITHM_COLORS.get(g, '#' + ''.join([f'{i*30:02x}' for i in range(3)])) 
                 for g in group_names]
        
        ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7, color=colors)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_names, rotation=45, ha='right')
        
        # 设置标题（移除eval_前缀并美化）
        metric_title = metric.replace('eval_', '').replace('_', ' ').title()
        ax.set_title(metric_title)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_success_vs_collision_scatter(experiments: List[Any],
                                      group_by: str = 'algorithm',
                                      title: str = 'Success Rate vs Collision Rate',
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (10, 8)):
    """绘制成功率vs碰撞率散点图
    
    Args:
        experiments: ExperimentData对象列表
        group_by: 分组依据
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    exps_with_eval = [e for e in experiments if e.eval_data is not None]
    
    if not exps_with_eval:
        print("Warning: No experiments with eval data found")
        return
    
    configure_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # 按group_by分组
    groups = {}
    for exp in exps_with_eval:
        key = getattr(exp, group_by, 'unknown')
        if key is None:
            key = 'unknown'
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)
    
    # 为每个组绘制散点
    for group_name in sorted(groups.keys()):
        group_exps = groups[group_name]
        
        success_rates = [exp.metrics.get('eval_success_rate', 0.0) for exp in group_exps]
        collision_rates = [exp.metrics.get('eval_collision_rate', 0.0) for exp in group_exps]
        
        color = ALGORITHM_COLORS.get(group_name, None)
        ax.scatter(collision_rates, success_rates, label=group_name, 
                  s=100, alpha=0.6, color=color, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Collision Rate', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # 添加理想区域标注（高成功率，低碰撞率）
    ax.axhspan(0.8, 1.0, xmin=0, xmax=0.2, alpha=0.1, color='green', label='Ideal Region')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
