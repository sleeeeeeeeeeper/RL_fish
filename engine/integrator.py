"""常微分方程数值积分器

提供多种求解微分方程的积分方法
"""
from typing import Callable, Optional
import numpy as np


class Integrator:
    """数值积分器基类"""
    
    def step(self, 
             dynamics_fn: Callable,
             state: np.ndarray, 
             action: np.ndarray, 
             dt: float,
             get_current_velocity_fn: Optional[Callable] = None) -> np.ndarray:
        """积分一个时间步
        
        Args:
            dynamics_fn: 计算状态导数的函数 (state, action, current_velocity) -> derivative
            state: 当前状态向量
            action: 动作/控制输入
            dt: 时间步长
            get_current_velocity_fn: 获取洋流速度的回调函数 (x, y) -> [vx, vy]，None表示无洋流
            
        Returns:
            下一时刻的状态向量
        """
        raise NotImplementedError


class EulerIntegrator(Integrator):
    """简单的前向欧拉积分器（一阶）"""
    
    def step(self, dynamics_fn, state, action, dt, get_current_velocity_fn=None):
        """前向欧拉法: x_{n+1} = x_n + dt * f(x_n, u, v_current)"""
        # 查询当前位置的洋流速度
        if get_current_velocity_fn is not None:
            current_velocity = get_current_velocity_fn(state[0], state[1])
        else:
            current_velocity = None
        
        derivative = dynamics_fn(state, action, current_velocity)
        return state + dt * derivative


class RK4Integrator(Integrator):
    """四阶龙格-库塔积分器（四阶）"""
    
    def step(self, dynamics_fn, state, action, dt, get_current_velocity_fn=None):
        """RK4积分，使用4个评估点
        
        对于空间变化的洋流，在每个k点重新查询洋流速度以保证精度
        """
        # k1: 初始位置
        if get_current_velocity_fn is not None:
            cv1 = get_current_velocity_fn(state[0], state[1])
        else:
            cv1 = None
        k1 = dynamics_fn(state, action, cv1)
        
        # k2: 中点位置
        state2 = state + 0.5 * dt * k1
        if get_current_velocity_fn is not None:
            cv2 = get_current_velocity_fn(state2[0], state2[1])
        else:
            cv2 = None
        k2 = dynamics_fn(state2, action, cv2)
        
        # k3: 中点位置（不同方向）
        state3 = state + 0.5 * dt * k2
        if get_current_velocity_fn is not None:
            cv3 = get_current_velocity_fn(state3[0], state3[1])
        else:
            cv3 = None
        k3 = dynamics_fn(state3, action, cv3)
        
        # k4: 终点位置
        state4 = state + dt * k3
        if get_current_velocity_fn is not None:
            cv4 = get_current_velocity_fn(state4[0], state4[1])
        else:
            cv4 = None
        k4 = dynamics_fn(state4, action, cv4)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
