"""二维地图管理模块

本模块提供地图表示，集成障碍物和洋流管理功能，
用于水下机器人路径规划任务
"""
from typing import List, Tuple, Optional
import numpy as np

from engine.obstacle_2d import Obstacle
from engine.ocean_current_2d import OceanCurrent


class Map2D:
    """带障碍物和洋流的二维地图，用于路径规划
    
    管理地图边界、障碍物集合和洋流场，提供碰撞检测和环境查询功能
    """
    
    def __init__(self, 
                 x_min: float, 
                 x_max: float, 
                 y_min: float, 
                 y_max: float,
                 obstacles: Optional[List[Obstacle]] = None,
                 ocean_current: Optional[OceanCurrent] = None):
        """初始化地图
        
        Args:
            x_min: 地图最小X坐标
            x_max: 地图最大X坐标
            y_min: 地图最小Y坐标
            y_max: 地图最大Y坐标
            obstacles: 障碍物列表（可选）
            ocean_current: 洋流对象（可选）
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.obstacles: List[Obstacle] = obstacles if obstacles is not None else []
        self.ocean_current: Optional[OceanCurrent] = ocean_current
    
    # ==================== 障碍物管理 ====================
    
    def add_obstacle(self, obstacle: Obstacle):
        """向地图添加障碍物
        
        Args:
            obstacle: 要添加的障碍物
        """
        self.obstacles.append(obstacle)
    
    def set_obstacles(self, obstacles: List[Obstacle]):
        """设置地图的障碍物列表
        
        Args:
            obstacles: 新的障碍物列表
        """
        self.obstacles = obstacles
    
    def clear_obstacles(self):
        """移除地图中的所有障碍物"""
        # 显式删除障碍物对象以帮助垃圾回收
        for obs in self.obstacles:
            del obs
        self.obstacles.clear()
    
    def get_obstacles(self) -> List[Obstacle]:
        """获取障碍物列表
        
        Returns:
            障碍物对象列表
        """
        return self.obstacles
    
    def get_obstacles_info(self) -> List[dict]:
        """获取所有障碍物信息用于可视化
        
        Returns:
            障碍物字典列表
        """
        return [obs.to_dict() for obs in self.obstacles]
    
    # ==================== 洋流管理 ====================
    
    def set_ocean_current(self, ocean_current: Optional[OceanCurrent]):
        """设置地图的洋流
        
        Args:
            ocean_current: 洋流对象，None表示无洋流
        """
        self.ocean_current = ocean_current
    
    def get_ocean_current(self) -> Optional[OceanCurrent]:
        """获取地图的洋流对象
        
        Returns:
            洋流对象，如果没有则返回None
        """
        return self.ocean_current
    
    def get_current_velocity(self, x: float, y: float) -> np.ndarray:
        """获取指定位置的洋流速度
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            速度向量 [vx, vy]，如果没有洋流则返回零向量
        """
        if self.ocean_current is None:
            return np.zeros(2, dtype=np.float32)
        return self.ocean_current.get_velocity(x, y)
    
    def update_ocean_current(self, dt: float):
        """更新洋流状态（用于时变洋流）
        
        Args:
            dt: 时间步长
        """
        if self.ocean_current is not None:
            self.ocean_current.update(dt)
    
    def update_obstacles(self, dt: float):
        """更新所有动态障碍物的位置
        
        Args:
            dt: 时间步长
        """
        map_bounds = self.get_bounds()
        for obstacle in self.obstacles:
            if hasattr(obstacle, 'is_dynamic') and obstacle.is_dynamic:
                obstacle.update(dt, map_bounds)
    
    def reset_ocean_current(self):
        """重置洋流状态"""
        if self.ocean_current is not None:
            self.ocean_current.reset()
    
    def reset_obstacles(self):
        """重置所有动态障碍物到初始状态"""
        for obstacle in self.obstacles:
            if hasattr(obstacle, 'is_dynamic') and obstacle.is_dynamic:
                if hasattr(obstacle, 'reset'):
                    obstacle.reset()
    
    # ==================== 碰撞检测 ====================
    
    def is_collision(self, x: float, y: float, safety_margin: float = 0.0) -> bool:
        """检查点是否与障碍物或边界碰撞
        
        Args:
            x: X坐标
            y: Y坐标
            safety_margin: 障碍物周围的额外安全距离
            
        Returns:
            如果发生碰撞返回True
        """
        # 检查地图边界（使用严格不等号，边界上或边界外视为碰撞）
        if x <= self.x_min or x >= self.x_max or y <= self.y_min or y >= self.y_max:
            return True
        
        # 检查每个障碍物（直接访问属性，避免函数调用开销）
        for obstacle in self.obstacles:
            # 使用障碍物的距离函数检测碰撞
            distance = obstacle.distance_to_point(x, y)
            if distance <= safety_margin:
                return True
        
        return False
    
    def get_nearest_obstacle_distance(self, x: float, y: float) -> float:
        """获取到最近障碍物的距离
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            到最近障碍物的距离（在外部为正，在内部为负）
        """
        if not self.obstacles:
            return float('inf')
        
        min_dist = float('inf')
        for obstacle in self.obstacles:
            dist = obstacle.distance_to_point(x, y)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    # ==================== 采样和查询 ====================
    
    def sample_free_position(self, 
                            rng: np.random.Generator, 
                            max_attempts: int = 100,
                            safety_margin: float = 0.0) -> Optional[Tuple[float, float]]:
        """在地图中采样无碰撞位置
        
        Args:
            rng: 随机数生成器
            max_attempts: 最大采样尝试次数
            safety_margin: 与障碍物的最小安全距离
            
        Returns:
            成功时返回(x, y)元组，否则返回None
        """
        for _ in range(max_attempts):
            x = rng.uniform(self.x_min, self.x_max)
            y = rng.uniform(self.y_min, self.y_max)
            
            if not self.is_collision(x, y, safety_margin):
                return (x, y)
        
        return None
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """获取地图边界
        
        Returns:
            (x_min, x_max, y_min, y_max)元组
        """
        return (self.x_min, self.x_max, self.y_min, self.y_max)
    
    def is_inside_bounds(self, x: float, y: float) -> bool:
        """检查点是否在地图边界内（不包括边界上）
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            如果点在边界内返回True（边界上返回False）
        """
        return (self.x_min < x < self.x_max and 
                self.y_min < y < self.y_max)
    
    # ==================== 批量设置 ====================
    
    def create_random_obstacles_separate(self,
                                        rng: np.random.Generator,
                                        static_config: Optional[dict] = None,
                                        dynamic_config: Optional[dict] = None):
        """分别创建静态和动态障碍物（独立配置）
        
        Args:
            rng: 随机数生成器
            static_config: 静态障碍物配置字典
                - num_obstacles: 数量
                - obstacle_type: 类型 ('circle' 或 'rectangle')
                - min_size: 最小尺寸
                - max_size: 最大尺寸
            dynamic_config: 动态障碍物配置字典
                - num_obstacles: 数量
                - obstacle_type: 类型 ('circle' 或 'rectangle')
                - min_size: 最小尺寸
                - max_size: 最大尺寸
                - motion_type: 运动类型
                - min_speed: 最小速度
                - max_speed: 最大速度
        """
        from engine.obstacle_2d import create_random_obstacles, create_random_dynamic_obstacles
        
        # 先清理旧障碍物
        self.clear_obstacles()
        # 重新初始化列表
        self.obstacles = []
        
        # 创建静态障碍物
        if static_config and static_config.get('num_obstacles', 0) > 0:
            static_obstacles = create_random_obstacles(
                num_obstacles=static_config['num_obstacles'],
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                rng=rng,
                obstacle_type=static_config.get('obstacle_type', 'circle'),
                min_size=static_config.get('min_size', 0.3),
                max_size=static_config.get('max_size', 0.8)
            )
            self.obstacles.extend(static_obstacles)
        
        # 创建动态障碍物
        if dynamic_config and dynamic_config.get('num_obstacles', 0) > 0:
            dynamic_obstacles = create_random_dynamic_obstacles(
                num_obstacles=dynamic_config['num_obstacles'],
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                rng=rng,
                obstacle_type=dynamic_config.get('obstacle_type', 'circle'),
                min_size=dynamic_config.get('min_size', 0.3),
                max_size=dynamic_config.get('max_size', 0.8),
                motion_type=dynamic_config.get('motion_type', 'bounce'),
                min_speed=dynamic_config.get('min_speed', 0.1),
                max_speed=dynamic_config.get('max_speed', 0.5)
            )
            self.obstacles.extend(dynamic_obstacles)
    
    # ==================== 环境重置 ====================
    
    def reset(self):
        """重置地图状态（清空障碍物和重置洋流及动态障碍物）"""
        self.clear_obstacles()
        self.reset_ocean_current()
        self.reset_obstacles()

