"""海洋洋流模拟模块

提供多种洋流模型，模拟真实的海洋流场环境
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
from enum import Enum

# 常量定义
POSITION_EPSILON = 1e-6  # 避免除零的位置阈值
TURBULENCE_FREQUENCY_COMPONENTS = 3  # 湍流频率分量数量


class CurrentType(Enum):
    """洋流类型枚举"""
    UNIFORM = "uniform"           # 均匀流
    VORTEX = "vortex"            # 涡流
    GRADIENT = "gradient"        # 梯度流
    OSCILLATING = "oscillating"  # 振荡流
    TURBULENT = "turbulent"      # 湍流
    NONE = "none"                # 无洋流


class OceanCurrent:
    """海洋洋流基类"""
    
    def __init__(self, strength: float = 1.0):
        """初始化洋流
        
        Args:
            strength: 洋流强度系数 (m/s)
        """
        self.strength = float(strength)
        self.time = 0.0
    
    def get_velocity(self, x: float, y: float) -> np.ndarray:
        """获取指定位置的洋流速度
        
        Args:
            x, y: 位置坐标
            
        Returns:
            洋流速度向量 [vx, vy] (m/s)
        """
        raise NotImplementedError
    
    def update(self, dt: float):
        """更新洋流状态（用于时变洋流）
        
        Args:
            dt: 时间步长
        """
        self.time += dt
    
    def reset(self):
        """重置洋流状态"""
        self.time = 0.0


class UniformCurrent(OceanCurrent):
    """均匀洋流 - 整个区域流速相同"""
    
    def __init__(self, 
                 velocity: Tuple[float, float] = (0.5, 0.0),
                 strength: float = 1.0):
        """初始化均匀流
        
        Args:
            velocity: 洋流速度 (vx, vy) (m/s)
            strength: 强度系数
        """
        super().__init__(strength)
        self.base_velocity = np.array(velocity, dtype=float)
    
    def get_velocity(self, x: float, y: float) -> np.ndarray:
        """返回均匀的洋流速度"""
        return self.base_velocity * self.strength


class VortexCurrent(OceanCurrent):
    """涡流 - 围绕中心点旋转的洋流"""
    
    def __init__(self,
                 center: Tuple[float, float] = (0.0, 0.0),
                 strength: float = 1.0,
                 radius: float = 5.0,
                 clockwise: bool = True):
        """初始化涡流
        
        Args:
            center: 涡流中心 (cx, cy)
            strength: 涡流强度 (m/s)
            radius: 影响半径 (m)
            clockwise: 是否顺时针旋转
        """
        super().__init__(strength)
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.clockwise = clockwise
    
    def get_velocity(self, x: float, y: float) -> np.ndarray:
        """计算涡流速度（切向速度）"""
        # 相对位置
        dx = x - self.center[0]
        dy = y - self.center[1]
        r = np.sqrt(dx**2 + dy**2)
        
        if r < POSITION_EPSILON:  # 避免除零
            return np.zeros(2)
        
        # 切向单位向量（垂直于径向）
        if self.clockwise:
            tangent_x = -dy / r
            tangent_y = dx / r
        else:
            tangent_x = dy / r
            tangent_y = -dx / r
        
        # 速度随距离衰减（高斯型）
        velocity_magnitude = self.strength * np.exp(-(r / self.radius)**2)
        
        return np.array([tangent_x * velocity_magnitude, 
                        tangent_y * velocity_magnitude])


class GradientCurrent(OceanCurrent):
    """梯度流 - 速度沿某方向线性变化"""
    
    def __init__(self,
                 base_velocity: Tuple[float, float] = (0.5, 0.0),
                 gradient: Tuple[float, float] = (0.1, 0.0),
                 strength: float = 1.0):
        """初始化梯度流
        
        Args:
            base_velocity: 原点处的基础速度 (vx, vy)
            gradient: 速度梯度 (dvx/dx, dvy/dy) (1/s)
            strength: 强度系数
        """
        super().__init__(strength)
        self.base_velocity = np.array(base_velocity, dtype=float)
        self.gradient = np.array(gradient, dtype=float)
    
    def get_velocity(self, x: float, y: float) -> np.ndarray:
        """计算梯度流速度"""
        # v(x,y) = v0 + gradient * position
        velocity = self.base_velocity + self.gradient * np.array([x, y])
        return velocity * self.strength


class OscillatingCurrent(OceanCurrent):
    """振荡流 - 周期性变化的洋流"""
    
    def __init__(self,
                 base_velocity: Tuple[float, float] = (0.5, 0.0),
                 amplitude: float = 0.3,
                 frequency: float = 0.5,
                 strength: float = 1.0):
        """初始化振荡流
        
        Args:
            base_velocity: 平均速度 (vx, vy)
            amplitude: 振幅 (m/s)
            frequency: 频率 (Hz)
            strength: 强度系数
        """
        super().__init__(strength)
        self.base_velocity = np.array(base_velocity, dtype=float)
        self.amplitude = float(amplitude)
        self.omega = 2 * np.pi * frequency  # 角频率
    
    def get_velocity(self, x: float, y: float) -> np.ndarray:
        """计算振荡流速度"""
        # v(t) = v0 + A * sin(ωt)
        oscillation = self.amplitude * np.sin(self.omega * self.time)
        velocity = self.base_velocity + np.array([oscillation, 0.0])
        return velocity * self.strength


class TurbulentCurrent(OceanCurrent):
    """湍流 - 随机扰动的洋流"""
    
    def __init__(self,
                 base_velocity: Tuple[float, float] = (0.5, 0.0),
                 turbulence_scale: float = 0.2,
                 spatial_scale: float = 2.0,
                 strength: float = 1.0,
                 seed: Optional[int] = None):
        """初始化湍流
        
        Args:
            base_velocity: 平均速度 (vx, vy)
            turbulence_scale: 湍流强度 (m/s)
            spatial_scale: 空间相关尺度 (m)
            strength: 强度系数
            seed: 随机种子
        """
        super().__init__(strength)
        self.base_velocity = np.array(base_velocity, dtype=float)
        self.turbulence_scale = float(turbulence_scale)
        self.spatial_scale = float(spatial_scale)
        self.rng = np.random.RandomState(seed)
    
    def get_velocity(self, x: float, y: float) -> np.ndarray:
        """计算湍流速度（基于Perlin噪声的简化版本）"""
        # 使用时间和空间的正弦组合生成伪随机场
        phase_x = (x / self.spatial_scale) + (self.time * 0.5)
        phase_y = (y / self.spatial_scale) + (self.time * 0.5)
        
        # 多个频率的叠加
        turbulence_x = 0.0
        turbulence_y = 0.0
        
        for i in range(TURBULENCE_FREQUENCY_COMPONENTS):  # 多个频率分量
            freq = 2 ** i
            turbulence_x += np.sin(freq * phase_x) * np.cos(freq * phase_y) / freq
            turbulence_y += np.cos(freq * phase_x) * np.sin(freq * phase_y) / freq
        
        turbulence = np.array([turbulence_x, turbulence_y]) * self.turbulence_scale
        velocity = self.base_velocity + turbulence
        
        return velocity * self.strength


class CompositeOceanCurrent(OceanCurrent):
    """复合洋流 - 多个洋流的叠加"""
    
    def __init__(self, currents: list = None):
        """初始化复合洋流
        
        Args:
            currents: 洋流对象列表
        """
        super().__init__(strength=1.0)
        self.currents = currents if currents is not None else []
    
    def add_current(self, current: OceanCurrent):
        """添加一个洋流分量"""
        self.currents.append(current)
    
    def get_velocity(self, x: float, y: float) -> np.ndarray:
        """计算叠加后的总洋流速度"""
        total_velocity = np.zeros(2)
        for current in self.currents:
            total_velocity += current.get_velocity(x, y)
        return total_velocity
    
    def update(self, dt: float):
        """更新所有洋流分量"""
        super().update(dt)
        for current in self.currents:
            current.update(dt)
    
    def reset(self):
        """重置所有洋流分量"""
        super().reset()
        for current in self.currents:
            current.reset()


def create_ocean_current(current_type: str = "none", 
                        config: Optional[Dict[str, Any]] = None) -> OceanCurrent:
    """创建洋流对象的工厂函数
    
    Args:
        current_type: 洋流类型 ("uniform", "vortex", "gradient", "oscillating", "turbulent", "none")
        config: 洋流配置参数
        
    Returns:
        OceanCurrent对象
        
    Examples:
        >>> # 创建均匀流
        >>> current = create_ocean_current("uniform", {"velocity": [0.5, 0.0], "strength": 1.0})
        
        >>> # 创建涡流
        >>> current = create_ocean_current("vortex", {"center": [0, 0], "strength": 1.0, "radius": 5.0})
    """
    if config is None:
        config = {}
    
    current_type = current_type.lower()
    
    if current_type == "none":
        return UniformCurrent(velocity=(0.0, 0.0), strength=0.0)
    
    elif current_type == "uniform":
        return UniformCurrent(
            velocity=config.get('velocity', (0.5, 0.0)),
            strength=config.get('strength', 1.0)
        )
    
    elif current_type == "vortex":
        return VortexCurrent(
            center=config.get('center', (0.0, 0.0)),
            strength=config.get('strength', 1.0),
            radius=config.get('radius', 5.0),
            clockwise=config.get('clockwise', True)
        )
    
    elif current_type == "gradient":
        return GradientCurrent(
            base_velocity=config.get('base_velocity', (0.5, 0.0)),
            gradient=config.get('gradient', (0.1, 0.0)),
            strength=config.get('strength', 1.0)
        )
    
    elif current_type == "oscillating":
        return OscillatingCurrent(
            base_velocity=config.get('base_velocity', (0.5, 0.0)),
            amplitude=config.get('amplitude', 0.3),
            frequency=config.get('frequency', 0.5),
            strength=config.get('strength', 1.0)
        )
    
    elif current_type == "turbulent":
        return TurbulentCurrent(
            base_velocity=config.get('base_velocity', (0.5, 0.0)),
            turbulence_scale=config.get('turbulence_scale', 0.2),
            spatial_scale=config.get('spatial_scale', 2.0),
            strength=config.get('strength', 1.0),
            seed=config.get('seed', None)
        )
    
    else:
        raise ValueError(f"Unknown current type: {current_type}")
