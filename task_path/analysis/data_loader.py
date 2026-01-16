"""数据加载模块

从实验结果目录加载训练数据、评估数据和配置信息
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ExperimentData:
    """实验数据的统一结构"""
    
    # 基本信息
    exp_name: str
    exp_dir: str
    algorithm: str
    
    # 配置信息
    config: Dict[str, Any]
    
    # 评估数据 (来自 evaluations.npz)
    timesteps: np.ndarray  # 评估时的训练步数 shape: (n_evals,)
    results: np.ndarray    # 每次评估的回报 shape: (n_evals, n_eval_episodes)
    ep_lengths: np.ndarray # 每次评估的episode长度 shape: (n_evals, n_eval_episodes)
    
    # Eval数据 (来自 eval_results.json，可选)
    eval_data: Optional[Dict[str, Any]] = None  # eval.py生成的评估数据
    
    # 派生信息
    env_difficulty: Optional[str] = None  # L1-L5
    hyperparam_variant: Optional[str] = None  # 超参数变体
    reward_variant: Optional[str] = None  # 奖励函数变体
    env_dimension: Optional[str] = None  # 环境维度 (current/obstacle/distance)
    
    # 计算的指标 (延迟计算)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """从实验名称解析元信息"""
        self._parse_experiment_name()
    
    def _parse_experiment_name(self):
        """从实验名称解析算法、难度等信息"""
        # 实验命名格式: {algo}_{exp_id}_{timestamp}
        # 例如: sac_exp_a1_l1_sac_20260108_001306
        
        # 提取算法
        if 'ppo' in self.exp_name.lower():
            self.algorithm = 'PPO'
        elif 'sac' in self.exp_name.lower():
            self.algorithm = 'SAC'
        elif 'td3' in self.exp_name.lower():
            self.algorithm = 'TD3'
        
        # 提取难度级别 (L1-L5)
        difficulty_match = re.search(r'_l(\d)', self.exp_name.lower())
        if difficulty_match:
            level = difficulty_match.group(1)
            self.env_difficulty = f'L{level}'
        
        # 提取超参数变体
        if '_lr' in self.exp_name:
            match = re.search(r'_lr(\w+)', self.exp_name)
            if match:
                self.hyperparam_variant = f'lr={match.group(1)}'
        elif '_batch' in self.exp_name:
            match = re.search(r'_batch(\d+)', self.exp_name)
            if match:
                self.hyperparam_variant = f'batch={match.group(1)}'
        elif '_clip' in self.exp_name:
            match = re.search(r'_clip(\w+)', self.exp_name)
            if match:
                self.hyperparam_variant = f'clip={match.group(1)}'
        elif '_buf' in self.exp_name:
            match = re.search(r'_buf(\w+)', self.exp_name)
            if match:
                self.hyperparam_variant = f'buffer={match.group(1)}'
        elif '_ent' in self.exp_name:
            match = re.search(r'_ent(\w+)', self.exp_name)
            if match:
                self.hyperparam_variant = f'entropy={match.group(1)}'
        elif '_delay' in self.exp_name:
            match = re.search(r'_delay(\d+)', self.exp_name)
            if match:
                self.hyperparam_variant = f'delay={match.group(1)}'
        elif '_noise' in self.exp_name:
            match = re.search(r'_noise(\w+)', self.exp_name)
            if match:
                self.hyperparam_variant = f'noise={match.group(1)}'
        
        # 提取奖励函数变体
        if 'reward' in self.exp_name:
            match = re.search(r'reward_(\w+)', self.exp_name)
            if match:
                self.reward_variant = match.group(1)
        
        # 提取环境维度
        if 'current' in self.exp_name:
            self.env_dimension = 'current'
        elif 'obs' in self.exp_name:
            self.env_dimension = 'obstacle'
        elif 'dist' in self.exp_name:
            self.env_dimension = 'distance'
    
    def get_mean_results(self) -> np.ndarray:
        """获取每次评估的平均回报
        
        Returns:
            shape: (n_evals,)
        """
        return np.mean(self.results, axis=1)
    
    def get_std_results(self) -> np.ndarray:
        """获取每次评估的标准差
        
        Returns:
            shape: (n_evals,)
        """
        return np.std(self.results, axis=1)
    
    def get_success_rate(self, success_threshold: float = 0.0) -> np.ndarray:
        """计算每次评估的成功率
        
        Args:
            success_threshold: 成功的回报阈值
        
        Returns:
            shape: (n_evals,)
        """
        # 假设回报>阈值即为成功
        # 注意：这是简化的计算，实际成功率应该从环境的info中获取
        # 但evaluations.npz中没有存储这个信息
        return np.mean(self.results > success_threshold, axis=1)


def load_evaluations(exp_dir: str) -> Dict[str, np.ndarray]:
    """从实验目录加载评估数据
    
    Args:
        exp_dir: 实验目录路径
    
    Returns:
        包含 timesteps, results, ep_lengths 的字典
    
    Raises:
        FileNotFoundError: 如果找不到 evaluations.npz
    """
    eval_path = os.path.join(exp_dir, 'logs', 'eval', 'evaluations.npz')
    
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluations file not found: {eval_path}")
    
    data = np.load(eval_path)
    
    return {
        'timesteps': data['timesteps'],
        'results': data['results'],
        'ep_lengths': data['ep_lengths'],
    }


def load_experiment_config(exp_dir: str) -> Dict[str, Any]:
    """从实验目录加载配置文件
    
    Args:
        exp_dir: 实验目录路径
    
    Returns:
        配置字典
    
    Raises:
        FileNotFoundError: 如果找不到 config.json
    """
    config_path = os.path.join(exp_dir, 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def load_eval_results(exp_dir: str) -> Optional[Dict[str, Any]]:
    """从实验目录加载eval.py生成的评估结果
    
    Args:
        exp_dir: 实验目录路径
    
    Returns:
        eval_results.json的内容，如果不存在则返回None
    
    Note:
        eval_results.json位于 {exp_dir}/eval/{timestamp}/eval_results.json
        如果有多个评估结果，加载最新的一个
    """
    eval_base_dir = os.path.join(exp_dir, 'eval')
    
    if not os.path.exists(eval_base_dir):
        return None
    
    # 查找所有时间戳目录
    eval_dirs = []
    for item in os.listdir(eval_base_dir):
        item_path = os.path.join(eval_base_dir, item)
        if os.path.isdir(item_path):
            eval_result_path = os.path.join(item_path, 'eval_results.json')
            if os.path.exists(eval_result_path):
                eval_dirs.append((item, eval_result_path))
    
    if not eval_dirs:
        return None
    
    # 按时间戳排序，取最新的
    eval_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_eval_path = eval_dirs[0][1]
    
    try:
        with open(latest_eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        return eval_data
    except Exception as e:
        print(f"Warning: Failed to load eval results from {latest_eval_path}: {e}")
        return None


def load_experiment_data(exp_dir: str) -> ExperimentData:
    """加载完整的实验数据
    
    Args:
        exp_dir: 实验目录路径
    
    Returns:
        ExperimentData 对象
    """
    # 获取实验名称
    exp_name = os.path.basename(exp_dir)
    
    # 加载评估数据（训练期间）
    eval_data = load_evaluations(exp_dir)
    
    # 加载配置
    config = load_experiment_config(exp_dir)
    
    # 加载eval.py生成的评估结果（可选）
    eval_results = load_eval_results(exp_dir)
    
    # 创建 ExperimentData 对象
    return ExperimentData(
        exp_name=exp_name,
        exp_dir=exp_dir,
        algorithm='',  # 将在 __post_init__ 中解析
        config=config,
        eval_data=eval_results,  # 添加eval数据
        **eval_data
    )


def scan_experiment_results(results_dir: str, 
                           experiment_group: Optional[str] = None) -> List[ExperimentData]:
    """扫描结果目录，加载所有实验数据
    
    Args:
        results_dir: 结果根目录 (例如: task_path/results)
        experiment_group: 实验组名称 (例如: 'a1', 'a2')，None表示所有实验
    
    Returns:
        ExperimentData 对象列表
    """
    experiments = []
    
    # 如果指定了实验组，只扫描该组
    if experiment_group:
        base_dir = os.path.join(results_dir, experiment_group)
        if not os.path.exists(base_dir):
            print(f"Warning: Experiment group '{experiment_group}' not found in {results_dir}")
            return experiments
    else:
        base_dir = results_dir
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(base_dir):
        # 检查是否是实验目录 (包含 config.json)
        if 'config.json' in files:
            try:
                # 尝试加载实验数据
                exp_data = load_experiment_data(root)
                experiments.append(exp_data)
                print(f"Loaded: {exp_data.exp_name} ({exp_data.algorithm})")
            except FileNotFoundError as e:
                print(f"Warning: Skipping {root}: {e}")
            except Exception as e:
                print(f"Error loading {root}: {e}")
    
    print(f"\nTotal experiments loaded: {len(experiments)}")
    return experiments


def filter_experiments(experiments: List[ExperimentData],
                       algorithm: Optional[str] = None,
                       difficulty: Optional[str] = None,
                       experiment_group: Optional[str] = None) -> List[ExperimentData]:
    """过滤实验数据
    
    Args:
        experiments: 实验数据列表
        algorithm: 算法名称 ('PPO', 'SAC', 'TD3')
        difficulty: 难度级别 ('L1', 'L2', 'L3', 'L4', 'L5')
        experiment_group: 实验组 ('a1', 'a2', 'a3', 'a4')
    
    Returns:
        过滤后的实验列表
    """
    filtered = experiments
    
    if algorithm:
        filtered = [exp for exp in filtered if exp.algorithm == algorithm.upper()]
    
    if difficulty:
        filtered = [exp for exp in filtered if exp.env_difficulty == difficulty.upper()]
    
    if experiment_group:
        filtered = [exp for exp in filtered 
                   if experiment_group.lower() in exp.exp_name.lower()]
    
    return filtered


def group_experiments_by(experiments: List[ExperimentData], 
                        by: str = 'algorithm') -> Dict[str, List[ExperimentData]]:
    """按指定字段分组实验
    
    Args:
        experiments: 实验数据列表
        by: 分组依据 ('algorithm', 'difficulty', 'hyperparam_variant', 'reward_variant')
    
    Returns:
        分组后的字典
    """
    groups = {}
    
    for exp in experiments:
        key = getattr(exp, by, 'unknown')
        if key is None:
            key = 'unknown'
        
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)
    
    return groups
