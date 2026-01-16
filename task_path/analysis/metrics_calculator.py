"""指标计算模块

计算各种训练和评估指标
"""

import numpy as np
from typing import Dict, Optional, Tuple


def calculate_sample_efficiency(timesteps: np.ndarray, 
                                mean_results: np.ndarray,
                                threshold: float = 0.8,
                                metric: str = 'relative') -> Optional[int]:
    """计算样本效率：首次达到阈值所需的训练步数
    
    Args:
        timesteps: 评估时的训练步数 shape: (n_evals,)
        mean_results: 平均回报或成功率 shape: (n_evals,)
        threshold: 阈值（相对或绝对值）
        metric: 'relative' (相对于最大值) 或 'absolute' (绝对值)
    
    Returns:
        首次达到阈值的步数，如果未达到则返回None
    """
    if metric == 'relative':
        # 相对于最大值的百分比
        max_val = np.max(mean_results)
        if max_val <= 0:
            return None
        threshold_val = threshold * max_val
    else:
        # 绝对阈值
        threshold_val = threshold
    
    # 找到首次超过阈值的索引
    indices = np.where(mean_results >= threshold_val)[0]
    
    if len(indices) > 0:
        return int(timesteps[indices[0]])
    else:
        return None


def calculate_convergence_speed(timesteps: np.ndarray,
                                mean_results: np.ndarray,
                                stability_window: int = 5,
                                stability_threshold: float = 0.05) -> Optional[int]:
    """计算收敛速度：性能稳定所需的步数
    
    定义：连续N次评估的方差小于阈值时认为已收敛
    
    Args:
        timesteps: 评估时的训练步数 shape: (n_evals,)
        mean_results: 平均回报 shape: (n_evals,)
        stability_window: 稳定性窗口大小
        stability_threshold: 相对标准差阈值
    
    Returns:
        收敛所需步数，如果未收敛则返回None
    """
    if len(mean_results) < stability_window:
        return None
    
    # 计算滑动窗口的相对标准差
    for i in range(len(mean_results) - stability_window + 1):
        window = mean_results[i:i+stability_window]
        mean_val = np.mean(window)
        
        if mean_val != 0:
            rel_std = np.std(window) / abs(mean_val)
            
            if rel_std < stability_threshold:
                return int(timesteps[i + stability_window - 1])
    
    return None


def calculate_training_stability(timesteps: np.ndarray,
                                 mean_results: np.ndarray,
                                 last_n_percent: float = 0.2) -> float:
    """计算训练稳定性：后期训练的相对标准差倒数
    
    稳定性越高，值越大
    
    Args:
        timesteps: 评估时的训练步数
        mean_results: 平均回报
        last_n_percent: 计算最后N%的数据
    
    Returns:
        稳定性指标 (1 / coefficient_of_variation)
    """
    n_points = len(mean_results)
    start_idx = max(0, int(n_points * (1 - last_n_percent)))
    
    last_results = mean_results[start_idx:]
    
    if len(last_results) < 2:
        return 0.0
    
    mean_val = np.mean(last_results)
    std_val = np.std(last_results)
    
    if std_val == 0 or mean_val == 0:
        return 100.0  # 完全稳定
    
    # 变异系数的倒数
    cv = std_val / abs(mean_val)
    stability = 1.0 / cv
    
    return stability


def calculate_peak_performance(mean_results: np.ndarray,
                               timesteps: Optional[np.ndarray] = None) -> Dict[str, float]:
    """计算峰值性能
    
    Args:
        mean_results: 平均回报
        timesteps: 训练步数（可选）
    
    Returns:
        包含 'peak_value' 和 'peak_step'（如果提供timesteps）的字典
    """
    peak_idx = np.argmax(mean_results)
    result = {
        'peak_value': float(mean_results[peak_idx]),
        'peak_index': int(peak_idx),
    }
    
    if timesteps is not None:
        result['peak_step'] = int(timesteps[peak_idx])
    
    return result


def calculate_final_performance(mean_results: np.ndarray,
                                last_n: int = 3) -> float:
    """计算最终性能：最后N次评估的平均值
    
    Args:
        mean_results: 平均回报
        last_n: 最后N次评估
    
    Returns:
        最终性能值
    """
    if len(mean_results) < last_n:
        last_n = len(mean_results)
    
    return float(np.mean(mean_results[-last_n:]))


def calculate_learning_curve_auc(timesteps: np.ndarray,
                                 mean_results: np.ndarray,
                                 normalize: bool = True) -> float:
    """计算学习曲线下面积 (AUC)
    
    Args:
        timesteps: 训练步数
        mean_results: 平均回报
        normalize: 是否归一化到[0,1]
    
    Returns:
        AUC值
    """
    # 使用梯形法则计算面积
    auc = np.trapz(mean_results, timesteps)
    
    if normalize:
        # 归一化到 [0, 1]
        max_possible_auc = np.max(mean_results) * (timesteps[-1] - timesteps[0])
        if max_possible_auc > 0:
            auc = auc / max_possible_auc
    
    return float(auc)


def calculate_improvement_rate(mean_results: np.ndarray,
                               timesteps: np.ndarray,
                               start_percent: float = 0.1,
                               end_percent: float = 0.9) -> float:
    """计算改进速率：从10%到90%性能的斜率
    
    Args:
        mean_results: 平均回报
        timesteps: 训练步数
        start_percent: 起始百分位
        end_percent: 结束百分位
    
    Returns:
        改进速率 (性能提升 / 训练步数)
    """
    n = len(mean_results)
    start_idx = int(n * start_percent)
    end_idx = int(n * end_percent)
    
    if start_idx >= end_idx:
        return 0.0
    
    perf_improvement = mean_results[end_idx] - mean_results[start_idx]
    steps_used = timesteps[end_idx] - timesteps[start_idx]
    
    if steps_used == 0:
        return 0.0
    
    return float(perf_improvement / steps_used)


def calculate_all_metrics(timesteps: np.ndarray,
                          results: np.ndarray,
                          ep_lengths: Optional[np.ndarray] = None) -> Dict[str, float]:
    """计算所有相关指标
    
    Args:
        timesteps: 评估时的训练步数 shape: (n_evals,)
        results: 评估回报 shape: (n_evals, n_eval_episodes)
        ep_lengths: episode长度 shape: (n_evals, n_eval_episodes)
    
    Returns:
        包含所有指标的字典
    """
    mean_results = np.mean(results, axis=1)
    std_results = np.std(results, axis=1)
    
    metrics = {}
    
    # 基本统计
    metrics['final_mean'] = calculate_final_performance(mean_results, last_n=3)
    metrics['final_std'] = float(np.mean(std_results[-3:]))
    
    peak_info = calculate_peak_performance(mean_results, timesteps)
    metrics['peak_value'] = peak_info['peak_value']
    metrics['peak_step'] = peak_info.get('peak_step', 0)
    
    # 学习效率
    sample_eff = calculate_sample_efficiency(timesteps, mean_results, threshold=0.8)
    metrics['sample_efficiency'] = sample_eff if sample_eff is not None else float(timesteps[-1])
    
    conv_speed = calculate_convergence_speed(timesteps, mean_results)
    metrics['convergence_steps'] = conv_speed if conv_speed is not None else float(timesteps[-1])
    
    # 稳定性
    metrics['training_stability'] = calculate_training_stability(timesteps, mean_results)
    
    # 学习曲线特征
    metrics['auc'] = calculate_learning_curve_auc(timesteps, mean_results, normalize=True)
    metrics['improvement_rate'] = calculate_improvement_rate(mean_results, timesteps)
    
    # Episode长度统计（如果提供）
    if ep_lengths is not None:
        mean_lengths = np.mean(ep_lengths, axis=1)
        metrics['final_episode_length'] = float(np.mean(mean_lengths[-3:]))
        metrics['min_episode_length'] = float(np.min(mean_lengths))
    
    return metrics


def compare_algorithms(experiments_by_algo: Dict[str, list],
                      metric_name: str = 'final_mean') -> Dict[str, float]:
    """比较不同算法在某个指标上的性能
    
    Args:
        experiments_by_algo: 按算法分组的实验列表
        metric_name: 要比较的指标名称
    
    Returns:
        每个算法的平均指标值
    """
    results = {}
    
    for algo, exps in experiments_by_algo.items():
        values = [exp.metrics.get(metric_name, 0.0) for exp in exps if metric_name in exp.metrics]
        
        if values:
            results[algo] = float(np.mean(values))
        else:
            results[algo] = 0.0
    
    return results


def calculate_relative_performance(baseline_value: float,
                                   test_value: float) -> float:
    """计算相对性能提升百分比
    
    Args:
        baseline_value: 基准值
        test_value: 测试值
    
    Returns:
        相对提升百分比
    """
    if baseline_value == 0:
        return 0.0
    
    return ((test_value - baseline_value) / abs(baseline_value)) * 100.0


def extract_eval_metrics(eval_data: dict) -> Dict[str, float]:
    """从eval.py生成的评估数据中提取关键指标
    
    Args:
        eval_data: eval_results.json的内容
    
    Returns:
        提取的指标字典
    """
    metrics = {}
    
    # 基本性能指标
    metrics['eval_mean_return'] = eval_data.get('mean_return', 0.0)
    metrics['eval_std_return'] = eval_data.get('std_return', 0.0)
    metrics['eval_success_rate'] = eval_data.get('success_rate', 0.0)
    metrics['eval_collision_rate'] = eval_data.get('collision_rate', 0.0)
    
    # Episode统计
    metrics['eval_mean_length'] = eval_data.get('mean_length', 0.0)
    metrics['eval_std_length'] = eval_data.get('std_length', 0.0)
    
    # 路径质量指标
    metrics['eval_mean_path_length'] = eval_data.get('mean_path_length', 0.0)
    metrics['eval_std_path_length'] = eval_data.get('std_path_length', 0.0)
    metrics['eval_mean_smoothness'] = eval_data.get('mean_smoothness', 0.0)
    metrics['eval_std_smoothness'] = eval_data.get('std_smoothness', 0.0)
    
    # 安全性指标
    metrics['eval_mean_min_obstacle_dist'] = eval_data.get('mean_min_obstacle_dist', 0.0)
    metrics['eval_std_min_obstacle_dist'] = eval_data.get('std_min_obstacle_dist', 0.0)
    
    # 能量消耗
    metrics['eval_mean_energy'] = eval_data.get('mean_energy', 0.0)
    metrics['eval_std_energy'] = eval_data.get('std_energy', 0.0)
    
    # 计算时间
    metrics['eval_mean_computation_time'] = eval_data.get('mean_computation_time', 0.0)
    
    # 总评估信息
    metrics['eval_total_episodes'] = eval_data.get('total_episodes', 0)
    
    return metrics
