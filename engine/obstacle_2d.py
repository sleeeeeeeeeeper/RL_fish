"""二维障碍物模块

本模块定义各种类型的障碍物及其碰撞检测逻辑，支持静态和动态障碍物
"""
from typing import Tuple, Optional
import numpy as np


class Obstacle:
    """障碍物基类"""
    
    def __init__(self):
        """初始化障碍物"""
        self.is_dynamic = False  # 标记是否为动态障碍物
    
    def contains_point(self, x: float, y: float) -> bool:
        """检查点是否在障碍物内部
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            如果点在障碍物内返回True
        """
        raise NotImplementedError
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """获取障碍物的包围盒
        
        Returns:
            (x_min, x_max, y_min, y_max)元组
        """
        raise NotImplementedError
    
    def to_dict(self) -> dict:
        """转换为字典格式用于可视化
        
        Returns:
            包含障碍物信息的字典
        """
        raise NotImplementedError
    
    def distance_to_point(self, x: float, y: float) -> float:
        """计算点到障碍物表面的距离
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            距离（在外部为正，在内部为负）
        """
        raise NotImplementedError
    
    def update(self, dt: float, map_bounds: Optional[Tuple[float, float, float, float]] = None):
        """更新障碍物状态（用于动态障碍物）
        
        Args:
            dt: 时间步长
            map_bounds: 地图边界 (x_min, x_max, y_min, y_max)，用于边界处理
        """
        pass  # 静态障碍物不需要更新
    
    def reset(self):
        """重置障碍物到初始状态（用于动态障碍物）"""
        pass  # 静态障碍物不需要重置


class CircleObstacle(Obstacle):
    """圆形障碍物"""
    
    def __init__(self, center_x: float, center_y: float, radius: float):
        """初始化圆形障碍物
        
        Args:
            center_x: 圆心X坐标
            center_y: 圆心Y坐标
            radius: 圆的半径
        """
        super().__init__()
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
    
    def contains_point(self, x: float, y: float) -> bool:
        """检查点是否在圆内"""
        dist_sq = (x - self.center_x)**2 + (y - self.center_y)**2
        return dist_sq <= self.radius**2
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """获取包围盒"""
        return (
            self.center_x - self.radius,
            self.center_x + self.radius,
            self.center_y - self.radius,
            self.center_y + self.radius
        )
    
    def to_dict(self) -> dict:
        """转换为字典格式用于可视化"""
        return {
            'type': 'circle',
            'center': (self.center_x, self.center_y),
            'radius': self.radius
        }
    
    def distance_to_point(self, x: float, y: float) -> float:
        """计算点到圆表面的距离"""
        dist_to_center = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        return dist_to_center - self.radius


class RectangleObstacle(Obstacle):
    """矩形障碍物"""
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """初始化矩形障碍物
        
        Args:
            x_min: 最小X坐标
            x_max: 最大X坐标
            y_min: 最小Y坐标
            y_max: 最大Y坐标
        """
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    
    def contains_point(self, x: float, y: float) -> bool:
        """检查点是否在矩形内"""
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """获取包围盒"""
        return (self.x_min, self.x_max, self.y_min, self.y_max)
    
    def to_dict(self) -> dict:
        """转换为字典格式用于可视化"""
        return {
            'type': 'rectangle',
            'bounds': (self.x_min, self.x_max, self.y_min, self.y_max)
        }
    
    def distance_to_point(self, x: float, y: float) -> float:
        """计算点到矩形表面的距离"""
        # 到矩形边界的距离
        dx = max(self.x_min - x, 0, x - self.x_max)
        dy = max(self.y_min - y, 0, y - self.y_max)
        
        # 如果点在矩形内，距离为负
        if self.contains_point(x, y):
            # 到最近边的距离
            dist_to_edges = [
                x - self.x_min,      # 左边
                self.x_max - x,      # 右边
                y - self.y_min,      # 下边
                self.y_max - y       # 上边
            ]
            return -min(dist_to_edges)
        
        # 点在矩形外
        return np.sqrt(dx**2 + dy**2)


def create_random_obstacles(num_obstacles: int,
                           x_min: float, x_max: float,
                           y_min: float, y_max: float,
                           rng: np.random.Generator,
                           obstacle_type: str = 'circle',
                           min_size: float = 0.3,
                           max_size: float = 0.8) -> list:
    """创建随机障碍物列表
    
    Args:
        num_obstacles: 要创建的障碍物数量
        x_min: 地图最小X坐标
        x_max: 地图最大X坐标
        y_min: 地图最小Y坐标
        y_max: 地图最大Y坐标
        rng: 随机数生成器
        obstacle_type: 障碍物类型 ('circle' 或 'rectangle')
        min_size: 最小障碍物尺寸（半径或半宽）
        max_size: 最大障碍物尺寸（半径或半宽）
        
    Returns:
        障碍物对象列表
    """
    obstacles = []
    
    for _ in range(num_obstacles):
        if obstacle_type == 'circle':
            # 随机圆形
            center_x = rng.uniform(x_min + max_size, x_max - max_size)
            center_y = rng.uniform(y_min + max_size, y_max - max_size)
            radius = rng.uniform(min_size, max_size)
            obstacles.append(CircleObstacle(center_x, center_y, radius))
        
        elif obstacle_type == 'rectangle':
            # 随机矩形
            width = rng.uniform(min_size * 2, max_size * 2)
            height = rng.uniform(min_size * 2, max_size * 2)
            x_min_rect = rng.uniform(x_min, x_max - width)
            y_min_rect = rng.uniform(y_min, y_max - height)
            obstacles.append(RectangleObstacle(
                x_min_rect, x_min_rect + width,
                y_min_rect, y_min_rect + height
            ))
    
    return obstacles


class DynamicCircleObstacle(CircleObstacle):
    """动态圆形障碍物（可移动）"""
    
    def __init__(self, center_x: float, center_y: float, radius: float,
                 velocity_x: float = 0.0, velocity_y: float = 0.0,
                 motion_type: str = 'linear', motion_params: Optional[dict] = None):
        """初始化动态圆形障碍物
        
        Args:
            center_x: 初始圆心X坐标
            center_y: 初始圆心Y坐标
            radius: 圆的半径
            velocity_x: X方向初始速度
            velocity_y: Y方向初始速度
            motion_type: 运动类型 ('linear', 'bounce', 'circular', 'random')
            motion_params: 运动参数字典
                - 'linear': 无额外参数
                - 'bounce': 无额外参数（碰到边界反弹）
                - 'circular': {'center_x', 'center_y', 'angular_velocity'} 绕点旋转
                - 'random': {'change_interval', 'max_speed'} 随机游走
        """
        super().__init__(center_x, center_y, radius)
        self.is_dynamic = True
        
        # 运动状态
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.initial_center = (center_x, center_y)
        self.initial_velocity = (velocity_x, velocity_y)
        
        # 运动类型和参数
        self.motion_type = motion_type
        self.motion_params = motion_params or {}
        
        # 用于circular运动
        if motion_type == 'circular':
            self.orbit_center_x = self.motion_params.get('center_x', center_x)
            self.orbit_center_y = self.motion_params.get('center_y', center_y)
            self.angular_velocity = self.motion_params.get('angular_velocity', 0.5)
            self.orbit_radius = np.sqrt((center_x - self.orbit_center_x)**2 + 
                                       (center_y - self.orbit_center_y)**2)
            self.initial_angle = np.arctan2(center_y - self.orbit_center_y,
                                           center_x - self.orbit_center_x)
            self.current_angle = self.initial_angle
        
        # 用于random运动
        if motion_type == 'random':
            self.change_interval = self.motion_params.get('change_interval', 2.0)
            self.max_speed = self.motion_params.get('max_speed', 1.0)
            self.time_since_change = 0.0
    
    def update(self, dt: float, map_bounds: Optional[Tuple[float, float, float, float]] = None):
        """更新障碍物位置
        
        Args:
            dt: 时间步长
            map_bounds: 地图边界 (x_min, x_max, y_min, y_max)
        """
        if self.motion_type == 'linear':
            # 直线运动
            self.center_x += self.velocity_x * dt
            self.center_y += self.velocity_y * dt
        
        elif self.motion_type == 'bounce':
            # 反弹运动（碰到边界反向）
            self.center_x += self.velocity_x * dt
            self.center_y += self.velocity_y * dt
            
            if map_bounds is not None:
                x_min, x_max, y_min, y_max = map_bounds
                # 检查X边界碰撞
                if self.center_x - self.radius < x_min or self.center_x + self.radius > x_max:
                    self.velocity_x = -self.velocity_x
                    self.center_x = np.clip(self.center_x, x_min + self.radius, x_max - self.radius)
                
                # 检查Y边界碰撞
                if self.center_y - self.radius < y_min or self.center_y + self.radius > y_max:
                    self.velocity_y = -self.velocity_y
                    self.center_y = np.clip(self.center_y, y_min + self.radius, y_max - self.radius)
        
        elif self.motion_type == 'circular':
            # 圆周运动
            self.current_angle += self.angular_velocity * dt
            self.center_x = self.orbit_center_x + self.orbit_radius * np.cos(self.current_angle)
            self.center_y = self.orbit_center_y + self.orbit_radius * np.sin(self.current_angle)
        
        elif self.motion_type == 'random':
            # 随机游走
            self.time_since_change += dt
            if self.time_since_change >= self.change_interval:
                # 改变速度方向
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(0, self.max_speed)
                self.velocity_x = speed * np.cos(angle)
                self.velocity_y = speed * np.sin(angle)
                self.time_since_change = 0.0
            
            self.center_x += self.velocity_x * dt
            self.center_y += self.velocity_y * dt
            
            # 保持在地图内
            if map_bounds is not None:
                x_min, x_max, y_min, y_max = map_bounds
                self.center_x = np.clip(self.center_x, x_min + self.radius, x_max - self.radius)
                self.center_y = np.clip(self.center_y, y_min + self.radius, y_max - self.radius)
    
    def reset(self):
        """重置障碍物到初始状态"""
        self.center_x, self.center_y = self.initial_center
        self.velocity_x, self.velocity_y = self.initial_velocity
        if self.motion_type == 'circular':
            self.current_angle = self.initial_angle
        if self.motion_type == 'random':
            self.time_since_change = 0.0
    
    def to_dict(self) -> dict:
        """转换为字典格式用于可视化"""
        base_dict = super().to_dict()
        base_dict['is_dynamic'] = True
        base_dict['velocity'] = (self.velocity_x, self.velocity_y)
        base_dict['motion_type'] = self.motion_type
        return base_dict


class DynamicRectangleObstacle(RectangleObstacle):
    """动态矩形障碍物（可移动）"""
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 velocity_x: float = 0.0, velocity_y: float = 0.0,
                 motion_type: str = 'linear', motion_params: Optional[dict] = None):
        """初始化动态矩形障碍物
        
        Args:
            x_min: 初始最小X坐标
            x_max: 初始最大X坐标
            y_min: 初始最小Y坐标
            y_max: 初始最大Y坐标
            velocity_x: X方向速度
            velocity_y: Y方向速度
            motion_type: 运动类型 ('linear', 'bounce')
            motion_params: 运动参数字典
        """
        super().__init__(x_min, x_max, y_min, y_max)
        self.is_dynamic = True
        
        # 运动状态
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.initial_bounds = (x_min, x_max, y_min, y_max)
        self.initial_velocity = (velocity_x, velocity_y)
        
        # 运动类型
        self.motion_type = motion_type
        self.motion_params = motion_params or {}
    
    def update(self, dt: float, map_bounds: Optional[Tuple[float, float, float, float]] = None):
        """更新障碍物位置
        
        Args:
            dt: 时间步长
            map_bounds: 地图边界 (x_min, x_max, y_min, y_max)
        """
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        
        if self.motion_type == 'linear':
            # 直线运动
            self.x_min += self.velocity_x * dt
            self.x_max += self.velocity_x * dt
            self.y_min += self.velocity_y * dt
            self.y_max += self.velocity_y * dt
        
        elif self.motion_type == 'bounce':
            # 反弹运动
            self.x_min += self.velocity_x * dt
            self.x_max += self.velocity_x * dt
            self.y_min += self.velocity_y * dt
            self.y_max += self.velocity_y * dt
            
            if map_bounds is not None:
                map_x_min, map_x_max, map_y_min, map_y_max = map_bounds
                # 检查X边界碰撞
                if self.x_min < map_x_min or self.x_max > map_x_max:
                    self.velocity_x = -self.velocity_x
                    if self.x_min < map_x_min:
                        self.x_min = map_x_min
                        self.x_max = map_x_min + width
                    else:
                        self.x_max = map_x_max
                        self.x_min = map_x_max - width
                
                # 检查Y边界碰撞
                if self.y_min < map_y_min or self.y_max > map_y_max:
                    self.velocity_y = -self.velocity_y
                    if self.y_min < map_y_min:
                        self.y_min = map_y_min
                        self.y_max = map_y_min + height
                    else:
                        self.y_max = map_y_max
                        self.y_min = map_y_max - height
    
    def reset(self):
        """重置障碍物到初始状态"""
        self.x_min, self.x_max, self.y_min, self.y_max = self.initial_bounds
        self.velocity_x, self.velocity_y = self.initial_velocity
    
    def to_dict(self) -> dict:
        """转换为字典格式用于可视化"""
        base_dict = super().to_dict()
        base_dict['is_dynamic'] = True
        base_dict['velocity'] = (self.velocity_x, self.velocity_y)
        base_dict['motion_type'] = self.motion_type
        return base_dict


def create_random_dynamic_obstacles(num_obstacles: int,
                                   x_min: float, x_max: float,
                                   y_min: float, y_max: float,
                                   rng: np.random.Generator,
                                   obstacle_type: str = 'circle',
                                   min_size: float = 0.3,
                                   max_size: float = 0.8,
                                   motion_type: str = 'bounce',
                                   min_speed: float = 0.1,
                                   max_speed: float = 0.5) -> list:
    """创建随机动态障碍物列表
    
    Args:
        num_obstacles: 要创建的障碍物数量
        x_min: 地图最小X坐标
        x_max: 地图最大X坐标
        y_min: 地图最小Y坐标
        y_max: 地图最大Y坐标
        rng: 随机数生成器
        obstacle_type: 障碍物类型 ('circle' 或 'rectangle')
        min_size: 最小障碍物尺寸
        max_size: 最大障碍物尺寸
        motion_type: 运动类型 ('linear', 'bounce', 'circular', 'random')
        min_speed: 最小速度
        max_speed: 最大速度
        
    Returns:
        动态障碍物对象列表
    """
    obstacles = []
    
    for _ in range(num_obstacles):
        # 随机速度
        speed = rng.uniform(min_speed, max_speed)
        angle = rng.uniform(0, 2 * np.pi)
        velocity_x = speed * np.cos(angle)
        velocity_y = speed * np.sin(angle)
        
        if obstacle_type == 'circle':
            # 随机动态圆形
            center_x = rng.uniform(x_min + max_size, x_max - max_size)
            center_y = rng.uniform(y_min + max_size, y_max - max_size)
            radius = rng.uniform(min_size, max_size)
            
            motion_params = {}
            if motion_type == 'circular':
                # 圆周运动参数
                motion_params = {
                    'center_x': rng.uniform(x_min + 2, x_max - 2),
                    'center_y': rng.uniform(y_min + 2, y_max - 2),
                    'angular_velocity': rng.uniform(-0.5, 0.5)
                }
            elif motion_type == 'random':
                # 随机游走参数
                motion_params = {
                    'change_interval': rng.uniform(1.0, 3.0),
                    'max_speed': max_speed
                }
            
            obstacles.append(DynamicCircleObstacle(
                center_x, center_y, radius,
                velocity_x, velocity_y,
                motion_type, motion_params
            ))
        
        elif obstacle_type == 'rectangle':
            # 随机动态矩形
            width = rng.uniform(min_size * 2, max_size * 2)
            height = rng.uniform(min_size * 2, max_size * 2)
            x_min_rect = rng.uniform(x_min, x_max - width)
            y_min_rect = rng.uniform(y_min, y_max - height)
            
            obstacles.append(DynamicRectangleObstacle(
                x_min_rect, x_min_rect + width,
                y_min_rect, y_min_rect + height,
                velocity_x, velocity_y,
                motion_type
            ))
    
    return obstacles

