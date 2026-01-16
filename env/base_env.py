"""Gymnasium兼容的基础环境类

为所有鱼型机器人环境提供通用结构
"""
from typing import Optional, Dict, Any
import numpy as np
import gymnasium as gym


class FishEnvBase(gym.Env):
    """鱼型机器人环境基类
    
    提供通用功能：
    - 观测/动作空间定义
    - 基本的reset/step结构
    - 渲染接口
    - 边界检查
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化基础环境
        
        Args:
            config: 配置字典
        """
        super().__init__()
        
        # 配置
        self.config = config or {}
        self.dt = float(self.config.get("dt", 0.05))
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")

        self.max_episode_steps = int(self.config.get("max_episode_steps", 500))
        if self.max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be positive, got {self.max_episode_steps}"
            )
        
        # 边界（由子类设置）
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        
        # 状态管理
        self.state = None
        self.step_count = 0
        self._rng = np.random.default_rng()
        
        # 渲染器（延迟初始化）
        self.renderer = None
        
        # 子类应该定义这些
        self.observation_space = None
        self.action_space = None
    
    def _check_bounds(self, x: float, y: float) -> bool:
        """检查位置是否在边界内
        
        边界内定义为：x_min < x < x_max 且 y_min < y < y_max
        边界上或边界外返回False
        """
        return (self.x_min < x < self.x_max and 
                self.y_min < y < self.y_max)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境。子类应重写但调用super()
        
        Args:
            seed: 用于可重复性的随机种子
            options: 额外的重置选项
        
        Returns:
            应返回(observation, info)元组
        """
        # 调用父类的reset进行正确的Gymnasium种子设置
        super().reset(seed=seed)
        self._rng = self.np_random
        self.step_count = 0
        
        # 重置渲染器
        if self.renderer is not None:
            self.renderer.reset()
        
        # 子类必须实现完整的重置逻辑
        # 这个基础实现只处理通用设置
    
    def step(self, action):
        """执行一步。子类应重写"""
        raise NotImplementedError("Subclass must implement step()")
    
    def render(self, mode: str = "human", save_path: Optional[str] = None):
        """渲染环境。子类应重写或使用默认实现"""
        pass
    
    def close(self):
        """清理资源"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
