"""路径规划性能评估指标

路径规划任务的专用指标，比较强化学习方法与传统方法
如A*和Dijkstra算法的性能
"""
import numpy as np

from typing import List, Dict, Any, Optional


class PathPlanningMetrics:
    """路径规划任务的指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.episode_returns = []
        self.episode_lengths = []
        self.success_count = 0
        self.collision_count = 0
        self.total_episodes = 0
        
        # 路径质量指标
        self.path_lengths = []  # 实际行驶的路径长度
        self.path_smoothness = []  # 路径平滑度度量
        self.min_obstacle_distances = []  # 与障碍物的最近距离
        
        # 效率指标
        self.computation_times = []  # 完成回合的时间
        self.energy_consumption = []  # 总能量消耗（动作幅度之和）
    
    def update(self, episode_info: Dict[str, Any]):
        """用回合结果更新指标
        
        Args:
            episode_info: 包含以下内容的字典:
                - episode_return: 总奖励
                - episode_length: 步数
                - success: 是否到达目标
                - collision: 是否发生碰撞
                - path_length: 行驶路径长度 (可选)
                - smoothness: 路径平滑度度量 (可选)
                - min_obstacle_dist: 与障碍物的最近距离 (可选)
                - computation_time: 计算时间 (可选)
                - energy: 总能量消耗 (可选)
        """
        self.episode_returns.append(episode_info['episode_return'])
        self.episode_lengths.append(episode_info['episode_length'])
        self.total_episodes += 1
        
        if episode_info.get('success', False):
            self.success_count += 1
        
        if episode_info.get('collision', False):
            self.collision_count += 1
        
        if 'path_length' in episode_info:
            self.path_lengths.append(episode_info['path_length'])
        
        if 'smoothness' in episode_info:
            self.path_smoothness.append(episode_info['smoothness'])
        
        if 'min_obstacle_dist' in episode_info:
            self.min_obstacle_distances.append(episode_info['min_obstacle_dist'])
        
        if 'computation_time' in episode_info:
            self.computation_times.append(episode_info['computation_time'])
        
        if 'energy' in episode_info:
            self.energy_consumption.append(episode_info['energy'])
    
    def get_summary(self) -> Dict[str, float]:
        """获取汇总统计
        
        Returns:
            包含指标统计的字典
        """
        if self.total_episodes == 0:
            return {}
        
        summary = {
            # 基本指标
            'mean_return': np.mean(self.episode_returns),
            'std_return': np.std(self.episode_returns),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            
            # 成功指标
            'success_rate': self.success_count / self.total_episodes,
            'collision_rate': self.collision_count / self.total_episodes,
            'total_episodes': self.total_episodes,
        }
        
        # 路径质量指标
        if self.path_lengths:
            summary['mean_path_length'] = np.mean(self.path_lengths)
            summary['std_path_length'] = np.std(self.path_lengths)
        
        if self.path_smoothness:
            summary['mean_smoothness'] = np.mean(self.path_smoothness)
            summary['std_smoothness'] = np.std(self.path_smoothness)
        
        if self.min_obstacle_distances:
            summary['mean_min_obstacle_dist'] = np.mean(self.min_obstacle_distances)
            summary['std_min_obstacle_dist'] = np.std(self.min_obstacle_distances)
        
        # 效率指标
        if self.computation_times:
            summary['mean_computation_time'] = np.mean(self.computation_times)
        
        if self.energy_consumption:
            summary['mean_energy'] = np.mean(self.energy_consumption)
            summary['std_energy'] = np.std(self.energy_consumption)
        
        return summary
    
    def print_summary(self):
        """打印格式化的汇总统计"""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("路径规划性能总结")
        print("="*50)
        
        print(f"\n基本指标:")
        print(f"  回合数: {summary.get('total_episodes', 0)}")
        print(f"  平均回报: {summary.get('mean_return', 0):.2f} ± {summary.get('std_return', 0):.2f}")
        print(f"  平均长度: {summary.get('mean_length', 0):.1f} ± {summary.get('std_length', 0):.1f} 步")
        
        print(f"\n成功指标:")
        print(f"  成功率: {summary.get('success_rate', 0)*100:.1f}%")
        print(f"  碰撞率: {summary.get('collision_rate', 0)*100:.1f}%")
        
        if 'mean_path_length' in summary:
            print(f"\n路径质量:")
            print(f"  路径长度: {summary['mean_path_length']:.2f} ± {summary['std_path_length']:.2f} m")
        
        if 'mean_smoothness' in summary:
            print(f"  平滑度: {summary['mean_smoothness']:.3f} ± {summary['std_smoothness']:.3f}")
        
        if 'mean_min_obstacle_dist' in summary:
            print(f"  最小障碍物距离: {summary['mean_min_obstacle_dist']:.3f} ± {summary['std_min_obstacle_dist']:.3f} m")
        
        if 'mean_energy' in summary:
            print(f"\n效率:")
            print(f"  能量消耗: {summary['mean_energy']:.2f} ± {summary['std_energy']:.2f}")
        
        if 'mean_computation_time' in summary:
            print(f"  计算时间: {summary['mean_computation_time']*1000:.1f} ms")
        
        print("="*50 + "\n")


def compute_path_length(trajectory: np.ndarray) -> float:
    """从轨迹计算总路径长度
    
    Args:
        trajectory: 形状为(N, 2)的位置数组
        
    Returns:
        总路径长度
    """
    if len(trajectory) < 2:
        return 0.0
    
    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)


def compute_path_smoothness(trajectory: np.ndarray) -> float:
    """计算路径平滑度（越小越平滑）
    
    通过累加绝对方向变化来测量总曲率
    
    Args:
        trajectory: 形状为(N, 2)的位置数组
        
    Returns:
        平滑度指标（0 = 直线，越高 = 转弯越多）
    """
    if len(trajectory) < 3:
        return 0.0
    
    # 计算方向向量
    diffs = np.diff(trajectory, axis=0)
    
    # 避免除零
    norms = np.linalg.norm(diffs, axis=1)
    norms[norms < 1e-6] = 1e-6
    
    # 归一化为单位向量
    directions = diffs / norms[:, np.newaxis]
    
    # 计算连续段之间的角度变化
    angle_changes = []
    for i in range(len(directions) - 1):
        dot_product = np.clip(np.dot(directions[i], directions[i+1]), -1.0, 1.0)
        angle_change = np.arccos(dot_product)
        angle_changes.append(angle_change)
    
    # 绝对角度变化之和（按段数归一化）
    if angle_changes:
        return np.sum(angle_changes) / len(angle_changes)
    return 0.0


def compute_min_obstacle_distance(trajectory: np.ndarray, 
                                  obstacles: List[Dict]) -> float:
    """计算沿轨迹到任何障碍物的最小距离
    
    Args:
        trajectory: 形状为(N, 2)的位置数组
        obstacles: 障碍物字典列表
        
    Returns:
        到障碍物的最小距离
    """
    if len(trajectory) == 0 or len(obstacles) == 0:
        return float('inf')
    
    min_dist = float('inf')
    
    for point in trajectory:
        x, y = point
        
        for obs in obstacles:
            # 转换为字典格式（如果是对象）
            if hasattr(obs, 'to_dict'):
                obs_dict = obs.to_dict()
            else:
                obs_dict = obs
            
            if obs_dict['type'] == 'circle':
                center_x, center_y = obs_dict['center']
                radius = obs_dict['radius']
                dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                dist = dist_to_center - radius
            
            elif obs_dict['type'] == 'rectangle':
                x_min, x_max, y_min, y_max = obs_dict['bounds']
                dx = max(x_min - x, 0, x - x_max)
                dy = max(y_min - y, 0, y - y_max)
                dist = np.sqrt(dx**2 + dy**2)
            
            min_dist = min(min_dist, dist)
    
    return min_dist


def compute_energy_consumption(actions: np.ndarray) -> float:
    """从动作序列计算总能量消耗
    
    能量消耗定义为动作幅度的平方和，反映控制力度
    
    Args:
        actions: 形状为(N, action_dim)的动作数组
        
    Returns:
        总能量消耗
    """
    if len(actions) == 0:
        return 0.0
    
    # 计算每个动作的L2范数的平方
    energy_per_step = np.sum(actions**2, axis=1)
    return np.sum(energy_per_step)


def compute_episode_metrics(trajectory: np.ndarray,
                           actions: np.ndarray,
                           obstacles: List[Dict],
                           success: bool,
                           collision: bool,
                           episode_return: float,
                           episode_length: int,
                           computation_time: Optional[float] = None) -> Dict[str, Any]:
    """计算一个完整回合的所有指标
    
    这是一个便捷函数，计算所有可用的指标
    
    Args:
        trajectory: 位置轨迹 (N, 2)
        actions: 动作序列 (N, action_dim)
        obstacles: 障碍物列表
        success: 是否成功到达目标
        collision: 是否发生碰撞
        episode_return: 总奖励
        episode_length: 回合步数
        computation_time: 计算时间（可选）
        
    Returns:
        包含所有指标的字典
    """
    metrics = {
        'episode_return': episode_return,
        'episode_length': episode_length,
        'success': success,
        'collision': collision,
    }
    
    # 路径长度
    if len(trajectory) > 0:
        metrics['path_length'] = compute_path_length(trajectory)
    
    # 路径平滑度
    if len(trajectory) > 0:
        metrics['smoothness'] = compute_path_smoothness(trajectory)
    
    # 最小障碍物距离
    if len(trajectory) > 0 and len(obstacles) > 0:
        metrics['min_obstacle_dist'] = compute_min_obstacle_distance(trajectory, obstacles)
    
    # 能量消耗
    if len(actions) > 0:
        metrics['energy'] = compute_energy_consumption(actions)
    
    # 计算时间
    if computation_time is not None:
        metrics['computation_time'] = computation_time
    
    return metrics

