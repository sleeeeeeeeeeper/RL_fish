"""实验数据分析模块

提供数据加载、指标计算、可视化和报告生成功能
"""

from .data_loader import (
    load_evaluations,
    load_experiment_config,
    ExperimentData,
    scan_experiment_results,
)

from .metrics_calculator import (
    calculate_sample_efficiency,
    calculate_convergence_speed,
    calculate_training_stability,
    calculate_peak_performance,
    calculate_final_performance,
)

from .visualization import (
    plot_learning_curves,
    plot_bar_comparison,
    plot_heatmap,
    plot_radar_chart,
    configure_plot_style,
)

__all__ = [
    # Data loading
    'load_evaluations',
    'load_experiment_config',
    'ExperimentData',
    'scan_experiment_results',
    # Metrics
    'calculate_sample_efficiency',
    'calculate_convergence_speed',
    'calculate_training_stability',
    'calculate_peak_performance',
    'calculate_final_performance',
    # Visualization
    'plot_learning_curves',
    'plot_bar_comparison',
    'plot_heatmap',
    'plot_radar_chart',
    'configure_plot_style',
]
