"""Gymnasium环境：带障碍物的二维路径规划

本环境实现了一个路径规划任务，机器人需要从起点导航到目标点
同时避开障碍物。与传统方法（A*、Dijkstra）不同，本环境使用强化学习
来学习无碰撞导航，无需预先构建全局地图。
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
from gymnasium import spaces

from env.base_env import FishEnvBase
from robot.fish_simple_2d import FishSimple2D
from engine.map_2d import Map2D
from engine.ocean_current_2d import create_ocean_current

# 常量定义
LIDAR_RAY_SAMPLES = 10  # 激光雷达采样点数量（减少以提高性能）
MIN_START_GOAL_DISTANCE = 2.0  # 起点和目标点的最小距离
POSITION_SAMPLING_MAX_ATTEMPTS = 200  # 位置采样最大尝试次数
GOAL_RESAMPLE_MAX_ATTEMPTS = 50  # 目标点重新采样最大尝试次数


class PathPlanning2DEnv(FishEnvBase):
    """带障碍物的二维路径规划环境
    
    观测: [x, y, vx, vy, goal_x, goal_y, d_obs_1, d_obs_2, ..., d_obs_n]
    - 机器人的位置和速度
    - 目标位置
    - 到最近障碍物的距离（用于局部感知）
    
    动作: [thrust_x, thrust_y] 归一化到 [-1, 1]
    
    任务: 从起点导航到目标点同时避开障碍物
    相比A*/Dijkstra的优势:
    - 不需要全局地图
    - 在线感知和反应
    - 可泛化到动态环境
    - 在大规模环境中扩展性更好
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, record_trajectory: bool = False):
        """初始化路径规划环境
        
        Args:
            config: 配置字典，采用分层结构:
                - task: 任务参数
                    - dt: 仿真时间步长 (默认: 0.05)
                    - max_episode_steps: 每个回合的最大步数 (默认: 1000)
                    - success_threshold: 到达目标的距离阈值 (默认: 0.3)
                - reward: 奖励参数
                    - progress_weight: 进度奖励权重 (默认: 1.0)
                    - action_cost_weight: 动作代价权重 (默认: 0.01)
                    - obstacle_penalty_weight: 障碍物惩罚权重 (默认: 1.0)
                    - collision_penalty: 碰撞惩罚 (默认: -10.0)
                    - success_reward: 成功奖励 (默认: 10.0)
                    - obstacle_danger_radius: 障碍物危险半径 (默认: 0.5)
                - map: 地图配置
                    - x_min, x_max, y_min, y_max: 地图边界
                    - obstacle: 障碍物子配置
                        - num_obstacles: 障碍物数量 (默认: 5)
                        - obstacle_type: 'circle' 或 'rectangle' (默认: 'circle')
                        - min_size: 最小尺寸 (默认: 0.3)
                        - max_size: 最大尺寸 (默认: 0.8)
                    - ocean_current: 洋流子配置
                        - enabled: 是否启用 (默认: False)
                        - type: 洋流类型
                        - config: 洋流参数
                - robot: 机器人配置
                    - mass: 质量 (默认: 0.5)
                    - max_thrust: 最大推力 (默认: 5.0)
                    - drag_coeff: 阻力系数 (默认: 0.5)
                - sensor: 传感器配置
                    - num_lidar_rays: 激光雷达方向数 (默认: 8)
                    - lidar_range: 最大感知距离 (默认: 3.0)
                    - safety_margin: 安全边距 (默认: 0.1)
            record_trajectory: 是否记录轨迹（训练时False，评估时True）
        """
        super().__init__(config)
        
        # 轨迹记录控制（训练时不记录以节省内存）
        self.record_trajectory = record_trajectory
        
        # ==================== 任务参数 ====================
        task_config = self.config.get("task", {})
        
        # 修正dt和max_episode_steps的读取（base_env从扁平结构读取，但实际配置是嵌套的）
        dt = task_config.get("dt")
        if dt is not None:
            self.dt = float(dt)
            if self.dt <= 0.0:
                raise ValueError(f"dt must be positive, got {self.dt}")
        
        max_steps = task_config.get("max_episode_steps")
        if max_steps is not None:
            self.max_episode_steps = int(max_steps)
            if self.max_episode_steps <= 0:
                raise ValueError(f"max_episode_steps must be positive, got {self.max_episode_steps}")
        
        self.success_threshold = float(task_config.get("success_threshold", 0.3))
        if self.success_threshold <= 0:
            raise ValueError(f"success_threshold must be positive, got {self.success_threshold}")
        
        # ==================== 奖励参数 ====================
        reward_config = self.config.get("reward", {})
        self.progress_weight = float(reward_config.get("progress_weight", 1.0))
        self.action_cost_weight = float(reward_config.get("action_cost_weight", 0.01))
        if self.action_cost_weight < 0:
            raise ValueError(f"action_cost_weight must be non-negative, got {self.action_cost_weight}")
        
        self.obstacle_penalty_weight = float(reward_config.get("obstacle_penalty_weight", 1.0))
        if self.obstacle_penalty_weight < 0:
            raise ValueError(f"obstacle_penalty_weight must be non-negative, got {self.obstacle_penalty_weight}")
        
        self.collision_penalty = float(reward_config.get("collision_penalty", -10.0))
        self.success_reward = float(reward_config.get("success_reward", 10.0))
        self.obstacle_danger_radius = float(reward_config.get("obstacle_danger_radius", 0.5))
        if self.obstacle_danger_radius <= 0:
            raise ValueError(f"obstacle_danger_radius must be positive, got {self.obstacle_danger_radius}")
        
        self.step_penalty_weight = float(reward_config.get("step_penalty_weight", 0.0))
        if self.step_penalty_weight < 0:
            raise ValueError(f"step_penalty_weight must be non-negative, got {self.step_penalty_weight}")
        
        # ==================== 传感器参数 ====================
        sensor_config = self.config.get("sensor", {})
        self.num_lidar_rays = int(sensor_config.get("num_lidar_rays", 8))
        if self.num_lidar_rays <= 0:
            raise ValueError(f"num_lidar_rays must be positive, got {self.num_lidar_rays}")
        
        self.lidar_range = float(sensor_config.get("lidar_range", 3.0))
        if self.lidar_range <= 0:
            raise ValueError(f"lidar_range must be positive, got {self.lidar_range}")
        
        self.safety_margin = float(sensor_config.get("safety_margin", 0.1))
        if self.safety_margin < 0:
            raise ValueError(f"safety_margin must be non-negative, got {self.safety_margin}")
        
        # ==================== 初始化地图 ====================
        map_config = self.config.get("map", {})
        self.map = self._init_map(map_config)
        
        # ==================== 初始化机器人 ====================
        robot_config = self.config.get("robot", {})
        self.robot = self._init_robot(robot_config)
        
        # ==================== 定义空间 ====================
        # 动作空间: [thrust_x, thrust_y] 归一化到 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # 观测空间: [x, y, vx, vy, goal_x, goal_y, lidar_readings...]
        obs_dim = 6 + self.num_lidar_rays
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # ==================== 状态变量 ====================
        self.goal_position = None
        self.start_position = None
        self.initial_distance = None
        self.collision = False
        self.previous_distance = None  # 用于计算进度
        self.trajectory = []  # 存储轨迹用于渲染
    
    def _init_map(self, map_config: Dict[str, Any]) -> Map2D:
        """初始化地图（包含边界、障碍物配置、洋流配置）
        
        Args:
            map_config: 地图配置字典
            
        Returns:
            初始化的Map2D对象
        """
        # 地图边界配置
        self.x_min = float(map_config.get("x_min", -10.0))
        self.x_max = float(map_config.get("x_max", 10.0))
        self.y_min = float(map_config.get("y_min", -10.0))
        self.y_max = float(map_config.get("y_max", 10.0))

        # 障碍物配置（保存用于reset时生成）
        self.static_obstacle_config = map_config.get("static_obstacles", None)
        self.dynamic_obstacle_config = map_config.get("dynamic_obstacles", None)
        
        # 洋流配置
        ocean_current_config = map_config.get("ocean_current", {})
        self.ocean_current_enabled = ocean_current_config.get("enabled", False)
        ocean_current = None
        
        if self.ocean_current_enabled:
            current_type = ocean_current_config.get("type", "uniform")
            current_params = ocean_current_config.get("config", {})
            ocean_current = create_ocean_current(current_type, current_params)
        
        # 创建地图对象（使用环境的边界属性）
        return Map2D(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            obstacles=None,  # 初始为空，在reset时生成
            ocean_current=ocean_current
        )
    
    def _init_robot(self, robot_config: Dict[str, Any]) -> FishSimple2D:
        """初始化机器人（需要地图已初始化以获取洋流回调）
        
        Args:
            robot_config: 机器人配置字典
            
        Returns:
            初始化的FishSimple2D对象
        """
        return FishSimple2D(
            mass=robot_config.get("mass", 0.5),
            max_thrust=robot_config.get("max_thrust", 5.0),
            drag_coeff=robot_config.get("drag_coeff", 0.5),
            get_current_velocity_fn=self.map.get_current_velocity  # 传入地图的洋流查询回调
        )
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """重置环境到初始状态
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            (observation, info)元组
        """
        super().reset(seed=seed)
        
        # 使用地图的方法生成随机障碍物（使用初始化时保存的配置）
        self.map.create_random_obstacles_separate(
            rng=self.np_random,
            static_config=self.static_obstacle_config,
            dynamic_config=self.dynamic_obstacle_config
        )
        
        # 采样无碰撞的起点和目标点
        self.start_position = self.map.sample_free_position(
            self.np_random, max_attempts=POSITION_SAMPLING_MAX_ATTEMPTS, 
            safety_margin=self.safety_margin
        )
        if self.start_position is None:
            # 如果采样失败则使用备用位置
            backup_start = (self.x_min + 1.0, self.y_min + 1.0)
            # 验证备用位置是否安全
            if not self.map.is_collision(backup_start[0], backup_start[1], self.safety_margin):
                self.start_position = backup_start
                import warnings
                warnings.warn(
                    "Failed to sample start position after 200 attempts, using backup position. "
                    "Map may be too crowded with obstacles."
                )
            else:
                raise RuntimeError(
                    "Cannot find valid start position. "
                    "Map is too crowded with obstacles or obstacles are too large."
                )
        
        self.goal_position = self.map.sample_free_position(
            self.np_random, max_attempts=POSITION_SAMPLING_MAX_ATTEMPTS, 
            safety_margin=self.safety_margin
        )
        if self.goal_position is None:
            # 备用位置
            backup_goal = (self.x_max - 1.0, self.y_max - 1.0)
            # 验证备用位置是否安全
            if not self.map.is_collision(backup_goal[0], backup_goal[1], self.safety_margin):
                self.goal_position = backup_goal
                import warnings
                warnings.warn(
                    "Failed to sample goal position after 200 attempts, using backup position. "
                    "Map may be too crowded with obstacles."
                )
            else:
                raise RuntimeError(
                    "Cannot find valid goal position. "
                    "Map is too crowded with obstacles or obstacles are too large."
                )
        
        # 确保起点和目标点之间距离足够远
        for attempt in range(GOAL_RESAMPLE_MAX_ATTEMPTS):
            distance = np.linalg.norm(
                np.array(self.goal_position) - np.array(self.start_position)
            )
            if distance >= MIN_START_GOAL_DISTANCE:
                break
            
            # 重新采样目标点
            self.goal_position = self.map.sample_free_position(
                self.np_random, max_attempts=100, safety_margin=self.safety_margin
            )
            if self.goal_position is None:
                backup_goal = (self.x_max - 1.0, self.y_max - 1.0)
                if not self.map.is_collision(backup_goal[0], backup_goal[1], self.safety_margin):
                    self.goal_position = backup_goal
                else:
                    # 如果备用位置也失败，使用对角线远端
                    self.goal_position = (
                        self.x_max - 1.0 if self.start_position[0] < 0 else self.x_min + 1.0,
                        self.y_max - 1.0 if self.start_position[1] < 0 else self.y_min + 1.0
                    )
                    break
        
        # 最终检查：如果仍然太近，警告但继续
        final_distance = np.linalg.norm(
            np.array(self.goal_position) - np.array(self.start_position)
        )
        if final_distance < MIN_START_GOAL_DISTANCE:
            import warnings
            warnings.warn(
                f"Start and goal positions are close ({final_distance:.2f}m < {MIN_START_GOAL_DISTANCE}m). "
                "Task may be too easy or map is too crowded."
            )
        
        # 初始化状态: [x, y, vx, vy]
        self.state = np.array([
            self.start_position[0],
            self.start_position[1],
            0.0,  # vx
            0.0   # vy
        ], dtype=np.float32)
        
        self.initial_distance = np.linalg.norm(
            self.state[:2] - np.array(self.goal_position)
        )
        self.previous_distance = self.initial_distance
        self.collision = False
        
        # 重置地图状态（包括洋流）
        self.map.reset_ocean_current()
        
        # 用起始位置初始化轨迹（仅在需要时）
        if self.record_trajectory:
            self.trajectory = [self.state[:2].copy()]
        else:
            self.trajectory = []  # 训练时不记录
        
        # 重置渲染器
        if self.renderer is not None:
            self.renderer.reset()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一个时间步
        
        Args:
            action: 控制输入 [thrust_x, thrust_y]
            
        Returns:
            (observation, reward, terminated, truncated, info)元组
        """
        # 将动作限制在有效范围
        action = np.clip(action, -1.0, 1.0)
        
        # 更新洋流状态（用于时变洋流）
        self.map.update_ocean_current(self.dt)
        
        # 更新动态障碍物位置（Map2D会自动检查每个障碍物是否为动态）
        self.map.update_obstacles(self.dt)
        
        # 物理仿真
        self.state = self.robot.step(self.state, action, self.dt)
        
        # 更新步数计数器
        self.step_count += 1
        
        # 检查碰撞
        x, y = self.state[0], self.state[1]
        self.collision = self.map.is_collision(x, y, safety_margin=self.safety_margin)
        
        # 检查是否出界（已由地图碰撞检查处理）
        out_of_bounds = not self._check_bounds(x, y)
        
        # 计算奖励
        reward = self._compute_reward(action)
        
        # 检查终止条件
        distance_to_goal = np.linalg.norm(self.state[:2] - np.array(self.goal_position))
        success = distance_to_goal < self.success_threshold
        terminated = success or self.collision or out_of_bounds
        
        # 检查截断（超时）
        truncated = self.step_count >= self.max_episode_steps
        
        # 更新previous_distance用于下一步
        self.previous_distance = distance_to_goal
        
        # 记录轨迹点（仅在需要时）
        if self.record_trajectory:
            self.trajectory.append(self.state[:2].copy())
        
        observation = self._get_observation()
        info = self._get_info()
        info['success'] = success
        info['collision'] = self.collision
        info['distance_to_goal'] = distance_to_goal
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观测
        
        Returns:
            包含机器人状态、目标和激光雷达读数的观测数组
        """
        x, y, vx, vy = self.state
        
        # 基本状态和目标
        obs = [x, y, vx, vy, self.goal_position[0], self.goal_position[1]]
        
        # 类似激光雷达的距离传感器
        lidar_readings = self._get_lidar_readings()
        obs.extend(lidar_readings)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_lidar_readings(self) -> list:
        """获取类激光雷达传感器的距离读数（向量化优化版）
        
        模拟指向不同方向的距离传感器。
        返回归一化的到最近障碍物或边界的距离。
        
        注意：此方法在每次调用时读取最新的障碍物位置，不影响动态障碍物。
        障碍物在step()中通过map.update_obstacles(dt)更新，然后此方法读取更新后的位置。
        
        Returns:
            距离读数列表（0=机器人处有障碍物，1=范围内无障碍物）
        """
        x, y = self.state[0], self.state[1]
        
        # 预先计算所有射线方向（向量化）
        angles = np.linspace(0, 2 * np.pi, self.num_lidar_rays, endpoint=False)
        dx = np.cos(angles)
        dy = np.sin(angles)
        
        # 预先计算采样距离
        sample_dists = np.linspace(0.1, self.lidar_range, LIDAR_RAY_SAMPLES)
        
        # 缓存障碍物列表（局部变量，每次调用都会重新读取最新状态）
        # 这个缓存只是避免在当前方法内部多次调用get_obstacles()
        # 并不会影响动态障碍物的更新（因为每次step都会先update_obstacles再调用此方法）
        obstacles = self.map.obstacles
        
        readings = []
        
        for i in range(self.num_lidar_rays):
            min_dist = self.lidar_range
            
            # 计算该射线上所有采样点的坐标（向量化）
            ray_x = x + sample_dists * dx[i]
            ray_y = y + sample_dists * dy[i]
            
            # 检查边界碰撞（向量化）
            out_of_bounds = ((ray_x < self.x_min) | (ray_x > self.x_max) | 
                            (ray_y < self.y_min) | (ray_y > self.y_max))
            
            if np.any(out_of_bounds):
                # 找到第一个越界点
                first_out = np.argmax(out_of_bounds)
                min_dist = sample_dists[first_out]
            else:
                # 检查障碍物碰撞
                for j, d in enumerate(sample_dists):
                    collision = False
                    for obstacle in obstacles:
                        if obstacle.contains_point(ray_x[j], ray_y[j]):
                            collision = True
                            break
                    
                    if collision:
                        min_dist = d
                        break
            
            # 归一化到[0, 1]
            readings.append(min_dist / self.lidar_range)
        
        return readings
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """计算当前状态和动作的奖励
        
        奖励组成:
        - 向目标前进的进度（塑形奖励）
        - 碰撞惩罚（大负值）
        - 动作成本（小负值）
        - 成功奖励（大正值）
        - 到障碍物的距离（距离过近时的惩罚）
        - 时间步惩罚（鼓励快速完成）
        
        Args:
            action: 控制输入
            
        Returns:
            标量奖励
        """
        x, y = self.state[0], self.state[1]
        
        # 到目标的距离
        distance = np.linalg.norm(self.state[:2] - np.array(self.goal_position))
        
        # 进度奖励：塑形以鼓励向目标移动
        progress_reward = (self.previous_distance - distance) * self.progress_weight
        
        # 碰撞惩罚
        if self.collision:
            return self.collision_penalty
        
        # 成功奖励
        if distance < self.success_threshold:
            return self.success_reward
        
        # 动作成本（鼓励能量效率）
        action_cost = self.action_cost_weight * np.sum(action**2)
        
        # 接近障碍物的惩罚（平滑惩罚）
        nearest_obstacle_dist = self.map.get_nearest_obstacle_distance(x, y)
        obstacle_penalty = 0.0
        if 0 < nearest_obstacle_dist < self.obstacle_danger_radius:
            # 随着距离变近而增加的平滑二次惩罚
            obstacle_penalty = (
                -self.obstacle_penalty_weight * 
                (1.0 - nearest_obstacle_dist / self.obstacle_danger_radius)**2
            )
        
        # 时间步惩罚（鼓励快速完成任务）
        step_penalty = self.step_penalty_weight
        
        reward = progress_reward - action_cost + obstacle_penalty - step_penalty
        
        return reward
    
    def _get_info(self) -> dict:
        """获取额外信息
        
        Returns:
            包含状态信息的字典
        """
        return {
            'position': self.state[:2].copy(),
            'velocity': self.state[2:4].copy(),
            'goal': np.array(self.goal_position),
            'obstacles': self.map.get_obstacles_info(),
            'num_obstacles': len(self.map.obstacles)
        }
    
    def render(self, save_path: Optional[str] = None):
        """渲染环境
        
        Args:
            save_path: 保存渲染图像的路径
        """
        if self.renderer is None:
            from env.renderer_2d import Renderer2D
            self.renderer = Renderer2D(
                x_lim=(self.x_min, self.x_max),
                y_lim=(self.y_min, self.y_max)
            )
        
        # 设置渲染器的轨迹
        self.renderer.set_trajectory(self.trajectory)
        
        # 向渲染器添加目标和障碍物
        x, y = self.state[0], self.state[1]
        
        self.renderer.render(
            current_pos=(x, y),
            target_pos=self.goal_position,
            obstacles=self.map.get_obstacles_info(),
            ocean_current=self.map.get_ocean_current(),
            show_current_field=self.ocean_current_enabled,
            save_path=save_path,
            success_threshold=self.success_threshold
        )
    
    def close(self):
        """清理环境资源"""
        # 关闭渲染器（如果存在）
        if hasattr(self, 'renderer') and self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        
        # 清理地图资源
        if hasattr(self, 'map'):
            self.map.clear_obstacles()
        
        super().close()
