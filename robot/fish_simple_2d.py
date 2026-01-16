"""简化的二维鱼型机器人模型

这是一个高层次的机器人模型，组合了：
- 物理动力学(来自engine)
- 数值积分(来自engine)
- 机器人特定配置
"""
import numpy as np
from typing import Optional, Callable
from engine.dynamics_2d import PointMassDynamics2D
from engine.integrator import RK4Integrator, Integrator


class FishSimple2D:
    """二维质点鱼型机器人
    
    本类封装物理引擎，提供简洁的机器人仿真接口
    洋流速度通过get_current_velocity_fn回调函数动态获取
    """
    
    def __init__(self,
                 mass: float = 0.5,
                 max_thrust: float = 5.0,
                 drag_coeff: float = 0.5,
                 integrator: Integrator = None,
                 get_current_velocity_fn: Optional[Callable[[float, float], np.ndarray]] = None):
        """初始化机器鱼
        
        Args:
            mass: 机器鱼质量 (kg)
            max_thrust: 最大推力 (N)
            drag_coeff: 阻力系数
            integrator: 数值积分器 (默认: RK4)
            get_current_velocity_fn: 获取洋流速度的回调函数 (x, y) -> [vx, vy]
        """
        self.mass = mass
        # 物理模型（不再持有洋流对象）
        self.dynamics = PointMassDynamics2D(
            mass=self.mass,
            max_thrust=max_thrust,
            drag_coeff=drag_coeff
        )
        
        # 数值积分器
        self.integrator = integrator if integrator is not None else RK4Integrator()
        
        # 洋流速度查询函数（由外部提供，例如从Map获取）
        self.get_current_velocity_fn = get_current_velocity_fn
    
    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """仿真一个时间步
        
        Args:
            state: 当前状态 [x, y, vx, vy]
            action: 控制输入 [thrust_x, thrust_y] 范围 [-1, 1]
            dt: 时间步长（秒）
            
        Returns:
            下一时刻状态 [x, y, vx, vy]
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        state = np.asarray(state, dtype=float).reshape(4,)
        action = np.asarray(action, dtype=float).reshape(2,)
        
        # 传递洋流查询函数给积分器
        return self.integrator.step(
            self.dynamics.compute_derivative,
            state,
            action,
            dt,
            self.get_current_velocity_fn  # 传递回调函数而不是速度值
        )
    
    def get_config(self):
        """获取机器人配置"""
        return {
            'dynamics': self.dynamics.get_params(),
            'integrator': self.integrator.__class__.__name__,
            'has_current_velocity_fn': self.get_current_velocity_fn is not None
        }
    
    def set_current_velocity_fn(self, fn: Optional[Callable[[float, float], np.ndarray]]):
        """设置洋流速度查询函数
        
        Args:
            fn: 回调函数 (x, y) -> [vx, vy]，None表示无洋流
        """
        self.get_current_velocity_fn = fn

