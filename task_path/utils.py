"""路径规划任务的工具函数

提供环境创建、模型加载、日志管理、配置管理等通用功能
"""
import os
import json
import numpy as np
import gymnasium as gym

from typing import Optional, Callable, Dict, Any, Type, List
from datetime import datetime
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor

from env.path_planning_2d import PathPlanning2DEnv
from algorithm import PPO as CustomPPO
from algorithm import SAC as CustomSAC
from algorithm import TD3 as CustomTD3

# ==================== 配置管理 ====================

# 配置文件目录
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

def load_config(name: str) -> Dict[str, Any]:
    """从JSON文件加载配置
    
    Args:
        name: 配置名称（不含.json后缀），或完整的配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: JSON格式错误
        
    Examples:
        >>> config = load_config('default')
        >>> config = load_config('hard')
        >>> config = load_config('/path/to/my_config.json')
    """
    # 判断是否为完整路径
    if name.endswith('.json') and os.path.exists(name):
        config_path = name
    else:
        # 从configs目录加载
        config_path = os.path.join(CONFIG_DIR, f"{name}.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """保存配置到JSON文件
    
    Args:
        config: 配置字典
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"配置已保存到: {path}")


def validate_config(config: Dict[str, Any]) -> None:
    """验证配置参数的合法性
    
    Args:
        config: 配置字典
        
    Raises:
        ValueError: 参数不合法
    """
    errors = []
    
    # 检查必需的顶层键（兼容旧格式和新格式）
    # 新格式：env, algorithm, training
    env_config = config['env']
    required_keys = ['task', 'robot', 'map', 'sensor', 'reward']
    for key in required_keys:
        if key not in env_config:
            errors.append(f"缺少必需的环境配置部分: env.{key}")

    if errors:
        raise ValueError("配置验证失败:\n  - " + "\n  - ".join(errors))
    
    # 验证任务参数
    task = env_config.get('task', {})
    if task.get('dt', 0) <= 0:
        errors.append(f"task.dt 必须为正数, 当前值: {task.get('dt')}")
    if task.get('max_episode_steps', 0) <= 0:
        errors.append(f"task.max_episode_steps 必须为正数, 当前值: {task.get('max_episode_steps')}")
    if task.get('success_threshold', 0) <= 0:
        errors.append(f"task.success_threshold 必须为正数, 当前值: {task.get('success_threshold')}")
    
    # 验证机器人参数
    robot = env_config.get('robot', {})
    if robot.get('mass', 0) <= 0:
        errors.append(f"robot.mass 必须为正数, 当前值: {robot.get('mass')}")
    if robot.get('max_thrust', -1) < 0:
        errors.append(f"robot.max_thrust 必须非负, 当前值: {robot.get('max_thrust')}")
    if robot.get('drag_coeff', -1) < 0:
        errors.append(f"robot.drag_coeff 必须非负, 当前值: {robot.get('drag_coeff')}")
    
    # 验证地图参数
    map_cfg = env_config.get('map', {})
    if map_cfg.get('x_min', 0) >= map_cfg.get('x_max', 0):
        errors.append("map.x_min 必须小于 map.x_max")
    if map_cfg.get('y_min', 0) >= map_cfg.get('y_max', 0):
        errors.append("map.y_min 必须小于 map.y_max")
    
    # 验证静态障碍物配置
    static_obstacle_cfg = map_cfg.get('static_obstacles', None)
    if static_obstacle_cfg:
        if static_obstacle_cfg.get('num_obstacles', 0) < 0:
            errors.append(f"map.static_obstacles.num_obstacles 必须非负, 当前值: {static_obstacle_cfg.get('num_obstacles')}")
        if static_obstacle_cfg.get('obstacle_type') not in ['circle', 'rectangle', None]:
            errors.append(f"map.static_obstacles.obstacle_type 必须是 'circle' 或 'rectangle', 当前值: {static_obstacle_cfg.get('obstacle_type')}")
        if static_obstacle_cfg.get('num_obstacles', 0) > 0:
            if static_obstacle_cfg.get('min_size', 0) <= 0:
                errors.append(f"map.static_obstacles.min_size 必须为正数, 当前值: {static_obstacle_cfg.get('min_size')}")
            if static_obstacle_cfg.get('max_size', 0) < static_obstacle_cfg.get('min_size', 0):
                errors.append("map.static_obstacles.max_size 必须大于等于 map.static_obstacles.min_size")
    
    # 验证动态障碍物配置
    dynamic_obstacle_cfg = map_cfg.get('dynamic_obstacles', None)
    if dynamic_obstacle_cfg:
        if dynamic_obstacle_cfg.get('num_obstacles', 0) < 0:
            errors.append(f"map.dynamic_obstacles.num_obstacles 必须非负, 当前值: {dynamic_obstacle_cfg.get('num_obstacles')}")
        if dynamic_obstacle_cfg.get('obstacle_type') not in ['circle', 'rectangle', None]:
            errors.append(f"map.dynamic_obstacles.obstacle_type 必须是 'circle' 或 'rectangle', 当前值: {dynamic_obstacle_cfg.get('obstacle_type')}")
        if dynamic_obstacle_cfg.get('num_obstacles', 0) > 0:
            if dynamic_obstacle_cfg.get('min_size', 0) <= 0:
                errors.append(f"map.dynamic_obstacles.min_size 必须为正数, 当前值: {dynamic_obstacle_cfg.get('min_size')}")
            if dynamic_obstacle_cfg.get('max_size', 0) < dynamic_obstacle_cfg.get('min_size', 0):
                errors.append("map.dynamic_obstacles.max_size 必须大于等于 map.dynamic_obstacles.min_size")
            
            # 验证动态障碍物特有的运动参数
            motion_type = dynamic_obstacle_cfg.get('motion_type')
            if motion_type not in ['linear', 'bounce', 'circular', 'random', None]:
                errors.append(f"map.dynamic_obstacles.motion_type 必须是 'linear', 'bounce', 'circular', 或 'random', 当前值: {motion_type}")
            if dynamic_obstacle_cfg.get('min_speed', 0) < 0:
                errors.append(f"map.dynamic_obstacles.min_speed 必须非负, 当前值: {dynamic_obstacle_cfg.get('min_speed')}")
            if dynamic_obstacle_cfg.get('max_speed', 0) < dynamic_obstacle_cfg.get('min_speed', 0):
                errors.append("map.dynamic_obstacles.max_speed 必须大于等于 map.dynamic_obstacles.min_speed")
    
    # 验证洋流配置
    ocean_current_cfg = map_cfg.get('ocean_current', {})
    if ocean_current_cfg.get('enabled'):
        current_type = ocean_current_cfg.get('type')
        if not current_type:
            errors.append("map.ocean_current.enabled=true 时必须指定 type")
        elif current_type not in ['uniform', 'vortex', 'gradient', 'oscillating', 'turbulent']:
            errors.append(f"map.ocean_current.type 不支持: {current_type}, 支持的类型: uniform, vortex, gradient, oscillating, turbulent")
        
        # 根据类型验证必需参数
        current_config = ocean_current_cfg.get('config', {})
        if current_type == 'uniform':
            if 'velocity' not in current_config:
                errors.append("uniform 类型洋流必须指定 velocity (例如: [0.5, 0.0])")
            elif not isinstance(current_config.get('velocity'), (list, tuple)) or len(current_config['velocity']) != 2:
                errors.append("uniform 类型洋流的 velocity 必须是包含两个数字的列表: [vx, vy]")
        elif current_type == 'vortex':
            if 'center' not in current_config:
                errors.append("vortex 类型洋流必须指定 center (例如: [0.0, 0.0])")
            elif not isinstance(current_config.get('center'), (list, tuple)) or len(current_config['center']) != 2:
                errors.append("vortex 类型洋流的 center 必须是包含两个数字的列表: [cx, cy]")
            if current_config.get('strength', 0) <= 0:
                errors.append("vortex 类型洋流的 strength 必须为正数")
            if current_config.get('radius', 0) <= 0:
                errors.append("vortex 类型洋流的 radius 必须为正数")
        elif current_type == 'gradient':
            if 'base_velocity' not in current_config:
                errors.append("gradient 类型洋流必须指定 base_velocity")
            if 'gradient' not in current_config:
                errors.append("gradient 类型洋流必须指定 gradient")
        elif current_type == 'oscillating':
            if 'base_velocity' not in current_config:
                errors.append("oscillating 类型洋流必须指定 base_velocity")
            if current_config.get('amplitude', 0) < 0:
                errors.append("oscillating 类型洋流的 amplitude 必须非负")
            if current_config.get('frequency', 0) <= 0:
                errors.append("oscillating 类型洋流的 frequency 必须为正数")
        elif current_type == 'turbulent':
            if 'base_velocity' not in current_config:
                errors.append("turbulent 类型洋流必须指定 base_velocity")
            if current_config.get('turbulence_scale', 0) < 0:
                errors.append("turbulent 类型洋流的 turbulence_scale 必须非负")
            if current_config.get('spatial_scale', 0) <= 0:
                errors.append("turbulent 类型洋流的 spatial_scale 必须为正数")
        
        # 验证强度系数
        if current_config.get('strength', 1.0) < 0:
            errors.append(f"map.ocean_current.config.strength 必须非负, 当前值: {current_config.get('strength')}")
    
    # 验证传感器参数
    sensor = env_config.get('sensor', {})
    if sensor.get('num_lidar_rays', 0) <= 0:
        errors.append(f"sensor.num_lidar_rays 必须为正数, 当前值: {sensor.get('num_lidar_rays')}")
    if sensor.get('lidar_range', 0) <= 0:
        errors.append(f"sensor.lidar_range 必须为正数, 当前值: {sensor.get('lidar_range')}")
    if sensor.get('safety_margin', -1) < 0:
        errors.append(f"sensor.safety_margin 必须非负, 当前值: {sensor.get('safety_margin')}")
    
    # 验证奖励参数
    reward = env_config.get('reward', {})
    if reward.get('progress_weight') is not None and reward.get('progress_weight') < 0:
        errors.append(f"reward.progress_weight 必须非负, 当前值: {reward.get('progress_weight')}")
    if reward.get('action_cost_weight', 0) < 0:
        errors.append(f"reward.action_cost_weight 必须非负, 当前值: {reward.get('action_cost_weight')}")
    if reward.get('obstacle_penalty_weight', 0) < 0:
        errors.append(f"reward.obstacle_penalty_weight 必须非负, 当前值: {reward.get('obstacle_penalty_weight')}")
    if reward.get('collision_penalty', 0) > 0:
        errors.append(f"reward.collision_penalty 必须为负数或零, 当前值: {reward.get('collision_penalty')}")
    if reward.get('success_reward', 0) < 0:
        errors.append(f"reward.success_reward 必须为非负数, 当前值: {reward.get('success_reward')}")
    if reward.get('obstacle_danger_radius', 0) <= 0:
        errors.append(f"reward.obstacle_danger_radius 必须为正数, 当前值: {reward.get('obstacle_danger_radius')}")
 
    if errors:
        raise ValueError("配置验证失败:\n  - " + "\n  - ".join(errors))


def get_config(name: str, validate: bool = True) -> Dict[str, Any]:
    """加载并验证配置（便捷函数）
    
    Args:
        name: 配置名称或路径
        validate: 是否验证配置
        
    Returns:
        配置字典
        
    Examples:
        >>> config = get_config('default')
        >>> config = get_config('hard')
    """
    config = load_config(name)
    
    if validate:
        validate_config(config)
    
    return config


# ==================== 环境创建工具 ====================

def make_env(config: Dict[str, Any],
             rank: int = 0,
             seed: int = 0,
             log_dir: Optional[str] = None) -> Callable[[], gym.Env]:
    """创建环境工厂函数
    
    Args:
        config: 完整配置字典（必须包含 'env' 键）
        rank: 环境编号（用于并行环境）
        seed: 随机种子
        log_dir: Monitor日志目录
        
    Returns:
        Monitor(gym.Env): 带监控功能的环境工厂
    """
    def _init() -> gym.Env:
        # 提取环境配置
        env_config = config['env']
        
        env = PathPlanning2DEnv(config=env_config)
        
        # 设置环境种子（在创建时设置，VecEnv会在初始化时自动reset）
        env.reset(seed=seed + rank)
        
        # 使用Monitor包装以记录训练统计
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        else:
            env = Monitor(env)
        
        return env
    
    return _init


def make_vec_env(config: Dict[str, Any],
                 n_envs: int = 4,
                 seed: int = 0,
                 vec_env_cls: Optional[Type[VecEnv]] = None,
                 log_dir: Optional[str] = None) -> VecEnv:
    """创建向量化环境
    
    Args:
        config: 完整配置字典（必须包含 'env' 键）
        n_envs: 并行环境数量
        seed: 随机种子
        vec_env_cls: 向量化环境类（None=自动选择）
        log_dir: Monitor日志目录
        
    Returns:
        向量化环境
    """
    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv
        # vec_env_cls = SubprocVecEnv
    
    # 创建环境工厂列表
    env_fns = [make_env(config, rank=i, seed=seed, log_dir=log_dir) 
               for i in range(n_envs)]
    
    return vec_env_cls(env_fns)


# ==================== 模型管理工具 ====================

# Stable-Baselines3算法注册表
_SB3_ALGORITHM_REGISTRY = {
    'ppo': PPO,
    'sac': SAC,
    'td3': TD3,
}

# 自定义算法注册表
_CUSTOM_ALGORITHM_REGISTRY = {
    'ppo': CustomPPO,
    'sac': CustomSAC,
    'td3': CustomTD3,
}

def create_model(algo_name: str,
                env: gym.Env,
                hyperparameters: Optional[Dict[str, Any]] = None,
                tensorboard_log: Optional[str] = None,
                seed: Optional[int] = None,
                device: str = 'auto',
                verbose: int = 1,
                use_custom: bool = False) -> Any:
    """创建RL算法模型
    
    Args:
        algo_name: 算法名称 ('ppo', 'sac', 'td3')
        env: 训练环境
        hyperparameters: 算法超参数字典
        tensorboard_log: TensorBoard日志目录
        seed: 随机种子
        device: 设备 ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...)
        verbose: 详细程度
        use_custom: 是否使用自定义算法实现（False=使用SB3）
        
    Returns:
        算法模型实例
        
    Raises:
        ValueError: 如果算法名称不支持
    """
    algo_name = algo_name.lower()
    
    # 选择算法注册表
    if use_custom:
        registry = _CUSTOM_ALGORITHM_REGISTRY
        impl_name = "Custom"
    else:
        registry = _SB3_ALGORITHM_REGISTRY
        impl_name = "Stable-Baselines3"
    
    if algo_name not in registry:
        raise ValueError(
            f"Unknown algorithm: {algo_name}. "
            f"Supported: {list(registry.keys())}"
        )
    
    algo_cls = registry[algo_name]
    
    if verbose >= 1:
        print(f"Using {impl_name} implementation of {algo_name.upper()}")
    
    # 合并默认参数和用户参数
    if use_custom:
        # 自定义算法：直接传递参数
        kwargs = {
            'env': env,
            'verbose': verbose,
            'device': device,
        }
        if hyperparameters:
            kwargs.update(hyperparameters)
        if tensorboard_log:
            kwargs['tensorboard_log'] = tensorboard_log
        if seed is not None:
            kwargs['seed'] = seed
    else:
        # SB3算法：需要policy参数
        kwargs = {
            'policy': 'MlpPolicy',
            'env': env,
            'verbose': verbose,
            'device': device,
        }
        if hyperparameters:
            kwargs.update(hyperparameters)
        if tensorboard_log:
            kwargs['tensorboard_log'] = tensorboard_log
        if seed is not None:
            kwargs['seed'] = seed
    
    return algo_cls(**kwargs)


def load_model(model_path: str, 
               algo_name: Optional[str] = None,
               env: Optional[gym.Env] = None,
               device: str = 'auto') -> Any:
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        algo_name: 算法名称（如果为None则从路径推断）
        env: 环境实例（可选，用于继续训练）
        device: 设备 ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...)
        
    Returns:
        加载的模型
        
    Raises:
        ValueError: 如果无法确定算法类型
    """
    algo_name = algo_name.lower()
    
    if algo_name not in _CUSTOM_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {algo_name}. "
            f"Supported: {list(_CUSTOM_ALGORITHM_REGISTRY.keys())}"
        )
    
    algo_cls = _CUSTOM_ALGORITHM_REGISTRY[algo_name]
    
    return algo_cls.load(model_path, env=env, device=device)


def save_model(model: Any, save_path: str, metadata: Optional[Dict] = None):
    """保存模型和元数据
    
    Args:
        model: 要保存的模型
        save_path: 保存路径（不含扩展名）
        metadata: 额外的元数据（将保存为JSON）
    """
    # 保存模型
    model.save(save_path)
    print(f"Model saved to: {save_path}.zip")
    
    # 保存元数据
    if metadata:
        metadata_path = save_path + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")


# ==================== 目录管理工具 ====================

def create_experiment_dir(base_dir: str,
                         experiment_name: str,
                         use_timestamp: bool = True) -> str:
    """创建实验目录
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称
        use_timestamp: 是否添加时间戳
        
    Returns:
        创建的目录路径
    """
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    else:
        exp_dir = os.path.join(base_dir, experiment_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建子目录
    subdirs = ['models', 'logs', 'plots', 'videos']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir


# ==================== 日志工具 ====================

class Logger:
    """简单的日志记录器"""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        """初始化日志记录器
        
        Args:
            log_file: 日志文件路径（None=仅打印到控制台）
            verbose: 是否打印到控制台
        """
        self.log_file = log_file
        self.verbose = verbose
        
        if log_file:
            # 确保目录存在
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            # 清空或创建日志文件
            with open(log_file, 'w') as f:
                f.write(f"Log started at {datetime.now()}\n")
                f.write("=" * 80 + "\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """记录消息
        
        Args:
            message: 要记录的消息
            level: 日志级别
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        if self.verbose:
            print(formatted_msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_msg + "\n")
    
    def info(self, message: str):
        """记录INFO级别消息"""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """记录WARNING级别消息"""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """记录ERROR级别消息"""
        self.log(message, "ERROR")
    
    def section(self, title: str, char: str = "="):
        """打印分隔线"""
        separator = char * 60
        self.log(f"\n{separator}")
        self.log(title)
        self.log(f"{separator}\n")


# ==================== 数据处理工具 ====================

def set_random_seed(seed: int):
    """设置所有随机种子
    
    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    # 如果需要，也可以设置torch的种子
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def print_model_info(model: Any):
    """打印模型信息
    
    Args:
        model: RL模型
    """
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    
    # 获取模型类型
    model_class = model.__class__.__name__
    print(f"Algorithm: {model_class}")
    
    # 获取策略架构
    if hasattr(model, 'policy'):
        policy = model.policy
        print(f"Policy: {policy.__class__.__name__}")
        
        # 打印网络架构
        if hasattr(policy, 'net_arch'):
            print(f"Network Architecture: {policy.net_arch}")
    
    # 学习率
    if hasattr(model, 'learning_rate'):
        print(f"Learning Rate: {model.learning_rate}")
    
    # 其他关键参数
    if hasattr(model, 'gamma'):
        print(f"Gamma: {model.gamma}")
    
    print("=" * 60 + "\n")


def format_time(seconds: float) -> str:
    """格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
