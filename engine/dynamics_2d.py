"""二维质点动力学模型

实现水下机器人的推力和阻力动力学，支持洋流影响
"""
import numpy as np
from typing import Dict, Any, Optional

# 常量定义
VELOCITY_EPSILON = 1e-6  # 避免除零的速度阈值


class PointMassDynamics2D:
    """二维质点动力学，包含推力、阻力和洋流影响
    
    状态: [x, y, vx, vy]
    动作: [thrust_x, thrust_y] (归一化到 [-1, 1])
    
    动力学方程:
        F_thrust = action * max_thrust (action是控制推力的输入)
        F_drag = -k_drag * (v - v_current) * |v - v_current|  (相对速度的二次阻力)
        a = (F_thrust + F_drag) / mass
        
    洋流影响:
        机器人受到洋流的推动，阻力基于相对于水流的速度计算
        洋流作为参数传入compute_derivative，而不是作为属性存储
    """
    
    def __init__(self, 
                 mass: float = 0.5,
                 max_thrust: float = 5.0,
                 drag_coeff: float = 0.5):
        """初始化动力学参数
        
        Args:
            mass: 机器人质量 (kg)
            max_thrust: 最大推力 (N)
            drag_coeff: 二次阻力系数
        """
        self.mass = float(mass)
        if self.mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass}")

        self.max_thrust = float(max_thrust)
        if self.max_thrust < 0.0:
            raise ValueError(f"max_thrust must be non-negative, got {max_thrust}")

        self.drag_coeff = float(drag_coeff)
        if self.drag_coeff < 0.0:
            raise ValueError(f"drag_coeff must be non-negative, got {drag_coeff}")
    
    def compute_derivative(self, 
                          state: np.ndarray, 
                          action: np.ndarray,
                          current_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """计算状态导数 [dx, dy, dvx, dvy]
        
        Args:
            state: [x, y, vx, vy]
            action: [thrust_x, thrust_y] 范围 [-1, 1]
            current_velocity: 当前位置的洋流速度 [vcx, vcy]，None表示无洋流
            
        Returns:
            状态导数 [vx, vy, ax, ay]
        """
        x, y, vx, vy = state
        
        # 计算推力（将动作限制在有效范围内）
        action_clipped = np.clip(action, -1.0, 1.0)
        thrust_force = action_clipped * self.max_thrust
        fx_thrust, fy_thrust = thrust_force
        
        # 获取洋流速度（从参数传入）
        if current_velocity is not None:
            vcx, vcy = current_velocity
        else:
            vcx, vcy = 0.0, 0.0
        
        # 计算相对于水流的速度（机器人相对水的运动）
        relative_velocity = np.array([vx - vcx, vy - vcy])
        relative_speed = np.linalg.norm(relative_velocity)
        
        # 计算阻力（基于相对速度的二次型：-k * v_rel * |v_rel|）
        if relative_speed > VELOCITY_EPSILON:  # 避免除零
            drag_force = -self.drag_coeff * relative_velocity * relative_speed
        else:
            drag_force = np.zeros(2)
        
        fx_drag, fy_drag = drag_force
        
        # 合力和加速度
        fx_total = fx_thrust + fx_drag
        fy_total = fy_thrust + fy_drag
        
        ax = fx_total / self.mass
        ay = fy_total / self.mass
        
        # 返回 [dx/dt, dy/dt, dvx/dt, dvy/dt]
        return np.array([vx, vy, ax, ay], dtype=float)
    
    def get_params(self) -> Dict[str, Any]:
        """获取当前动力学参数"""
        return {
            'mass': self.mass,
            'max_thrust': self.max_thrust,
            'drag_coeff': self.drag_coeff
        }
    
    def set_params(self, **kwargs):
        """更新动力学参数"""
        if 'mass' in kwargs:
            mass = float(kwargs['mass'])
            if mass <= 0.0:
                raise ValueError(f"mass must be positive, got {mass}")
            self.mass = mass
        if 'max_thrust' in kwargs:
            max_thrust = float(kwargs['max_thrust'])
            if max_thrust < 0.0:
                raise ValueError(f"max_thrust must be non-negative, got {max_thrust}")
            self.max_thrust = max_thrust
        if 'drag_coeff' in kwargs:
            drag_coeff = float(kwargs['drag_coeff'])
            if drag_coeff < 0.0:
                raise ValueError(f"drag_coeff must be non-negative, got {drag_coeff}")
            self.drag_coeff = drag_coeff
