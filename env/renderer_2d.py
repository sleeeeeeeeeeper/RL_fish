"""使用matplotlib的简单2D渲染器

此渲染器设计为轻量级，默认在检测不到显示时以无头模式工作
"""
from typing import Iterable, Sequence, Optional
import os
import matplotlib

if "MPLBACKEND" not in os.environ and os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter


class Renderer2D:
    def __init__(self, width: int = 6, height: int = 6, 
                 x_lim: tuple = None, y_lim: tuple = None):
        self.width = width
        self.height = height
        self.x_lim = x_lim
        self.y_lim = y_lim
        self._fig, self._ax = plt.subplots(figsize=(self.width, self.height))
        self._trajectory = []
    
    def close(self):
        """关闭渲染器，释放matplotlib资源"""
        if hasattr(self, '_fig') and self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
    
    def __del__(self):
        """析构时确保资源释放"""
        self.close()

    def set_trajectory(self, points: Iterable[Sequence[float]]):
        """用提供的点替换存储的轨迹"""
        self._trajectory = [np.array(p, dtype=float) for p in points]

    def add_point(self, x: float, y: float):
        """向轨迹追加单个点"""
        self._trajectory.append(np.array([x, y], dtype=float))

    def render(
        self,
        current_pos: tuple = None,
        target_pos: tuple = None,
        obstacles: list = None,
        ocean_current = None,
        show_current_field: bool = False,
        show: bool = False,
        save_path: Optional[str] = None,
        return_rgb_array: bool = False,
        success_threshold: float = 0.3,
    ) -> Optional[np.ndarray]:
        """渲染当前位置、目标和障碍物。可选保存或返回像素数据

        current_pos: (x, y)元组 - 当前位置
        target_pos: (x, y)元组 - 目标位置
        obstacles: 障碍物字典列表，包含'type', 'center'/'bounds', 'radius'
        ocean_current: 洋流对象，用于可视化流场
        show_current_field: 是否显示洋流流场
        """
        self._ax.clear()
        
        # 设置背景颜色（海洋色）
        self._ax.set_facecolor('#E0F2FB')
        self._fig.patch.set_facecolor('white')
        
        # 绘制洋流流场（如果启用）
        if show_current_field and ocean_current is not None:
            self._render_ocean_current(self._ax, ocean_current, obstacles)
        
        # 绘制轨迹
        traj = np.array(self._trajectory)
        
        if traj.shape[0] > 0:
            self._ax.plot(traj[:, 0], traj[:, 1], '-', color='#4A90E2', linewidth=2, alpha=0.8, label='Trajectory')
            # 绘制起点（橙色实心小圆圈）
            self._ax.plot(traj[0, 0], traj[0, 1], 'o', markersize=10, 
                         markerfacecolor='#F6B073', markeredgecolor='#E89A56', 
                         linewidth=1.5, label='Starting Point', zorder=10)
        
        # 绘制障碍物（带边界裁剪）
        if obstacles:
            # 创建裁剪区域（地图边界）
            if self.x_lim and self.y_lim:
                clip_box = patches.Rectangle(
                    (self.x_lim[0], self.y_lim[0]),
                    self.x_lim[1] - self.x_lim[0],
                    self.y_lim[1] - self.y_lim[0],
                    transform=self._ax.transData,
                    fill=False,
                    visible=False
                )
                self._ax.add_patch(clip_box)
            else:
                clip_box = None
            
            for i, obs in enumerate(obstacles):
                if obs['type'] == 'circle':
                    circle = plt.Circle(obs['center'], obs['radius'], 
                                      facecolor='#737AAF', alpha=0.8, 
                                      edgecolor='#5A6096', linewidth=1.5,
                                      label='Obstacle' if i == 0 else '')
                    # 应用裁剪，只显示在边界内的部分
                    if clip_box is not None:
                        circle.set_clip_path(clip_box)
                    self._ax.add_patch(circle)
                elif obs['type'] == 'rectangle':
                    x_min, x_max, y_min, y_max = obs['bounds']
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                        facecolor='#737AAF', alpha=0.8,
                                        edgecolor='#5A6096', linewidth=1.5,
                                        label='Obstacle' if i == 0 else '')
                    # 应用裁剪，只显示在边界内的部分
                    if clip_box is not None:
                        rect.set_clip_path(clip_box)
                    self._ax.add_patch(rect)
        
        # 绘制终点（半透明圆形区域，虚线边缘）
        if target_pos:
            # 绘制半透明圆形区域
            goal_circle = plt.Circle(
                target_pos, success_threshold,
                facecolor='#2FBD9C', alpha=0.3,
                edgecolor='#26A085', linewidth=2,
                linestyle='--',  # 虚线
                label='End Point', zorder=10
            )
            self._ax.add_patch(goal_circle)
            # 在中心添加一个小标记点
            self._ax.plot(target_pos[0], target_pos[1], 'x', markersize=8,
                         markeredgecolor='#26A085', markeredgewidth=2, zorder=11)
        
        # 绘制当前位置
        if current_pos:
            self._ax.plot(current_pos[0], current_pos[1], 'o', markersize=8, 
                         markerfacecolor='#FF6B6B', markeredgecolor='#E85555', 
                         linewidth=1.5, label='Current Position', zorder=10)
        
        self._ax.set_xlabel('X (m)', fontsize=11, fontweight='bold', color='#313D78')
        self._ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold', color='#313D78')
        self._ax.set_title('2D Path Planning', fontsize=14, fontweight='bold', color='#313D78', pad=15)
        
        # 如果提供则设置坐标轴范围
        if self.x_lim:
            self._ax.set_xlim(self.x_lim)
        if self.y_lim:
            self._ax.set_ylim(self.y_lim)
        
        # 去掉网格线
        self._ax.grid(False)
        
        # 设置边框颜色
        for spine in self._ax.spines.values():
            spine.set_edgecolor('#313D78')
            spine.set_linewidth(2)
        
        # 设置刻度颜色
        self._ax.tick_params(colors='#313D78', which='both')
        
        # 设置等比例，但不改变坐标轴范围
        self._ax.set_aspect('equal', adjustable='box')
        
        # 再次确保坐标轴范围符合设置（在set_aspect之后）
        if self.x_lim:
            self._ax.set_xlim(self.x_lim)
        if self.y_lim:
            self._ax.set_ylim(self.y_lim)
        
        # 创建图例，使用自动定位避免遮挡元素
        handles, labels = self._ax.get_legend_handles_labels()
        # 移除重复的标签
        by_label = dict(zip(labels, handles))
        legend = self._ax.legend(
            by_label.values(), 
            by_label.keys(),
            loc='best',                  # 自动选择最佳位置，避免遮挡数据
            frameon=True,                # 显示边框
            framealpha=0.95,             # 边框透明度
            edgecolor='#313D78',         # 边框颜色
            facecolor='white',           # 背景颜色
            fontsize=9,                  # 字体大小
            shadow=True,                 # 添加阴影
            borderpad=0.8,               # 内边距
            labelspacing=0.6,            # 标签间距
            handlelength=1.5,            # 图例标记长度
            handleheight=0.7,            # 图例标记高度
            ncol=1                       # 单列显示
        )

        if save_path:
            self._fig.savefig(save_path, dpi=150, bbox_inches='tight')

        # 对于非阻塞显示，使用plt.pause()而非plt.show()
        if show and matplotlib.get_backend() != 'Agg':
            plt.pause(0.001)  # 非阻塞更新

        if return_rgb_array:
            self._fig.canvas.draw()
            width, height = self._fig.canvas.get_width_height()
            image = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return image.reshape((height, width, 3))

        return None

    def render_animation(
        self,
        trajectory: np.ndarray,
        obstacles: list,
        start_pos: tuple,
        goal_pos: tuple,
        save_path: str,
        episode_num: int = 0,
        info: dict = None,
        ocean_current = None,
        show_current_field: bool = False,
        success_threshold: float = 0.3,
    ) -> bool:
        """为单个episode创建动画视频

        Args:
            trajectory: 轨迹数组 (n_steps, 2)
            obstacles: 障碍物信息列表（字典格式）
            start_pos: 起始位置
            goal_pos: 目标位置
            save_path: 保存路径
            episode_num: episode编号
            info: episode信息字典
            ocean_current: 洋流对象
            show_current_field: 是否显示洋流流场
        """
        # 创建新的figure以避免干扰现有状态
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        
        # 设置背景颜色（海洋色）
        ax.set_facecolor('#E0F2FB')
        fig.patch.set_facecolor('white')
        
        # 设置坐标轴
        if self.x_lim:
            ax.set_xlim(self.x_lim)
        if self.y_lim:
            ax.set_ylim(self.y_lim)
            
        
        # 绘制洋流流场（如果启用）
        if show_current_field and ocean_current is not None:
            self._render_ocean_current(ax, ocean_current, obstacles)
        
        # 标题
        success = info.get('success', False) if info else False
        collision = info.get('collision', False) if info else False
        status = "成功 ✓" if success else ("碰撞 ✗" if collision else "超时")
        ax.set_title(f'Episode {episode_num} - {status}', fontsize=14, fontweight='bold', color='#313D78', pad=15)
        
        # 绘制障碍物（带边界裁剪）
        # 创建裁剪区域（地图边界）
        if self.x_lim and self.y_lim:
            clip_box = patches.Rectangle(
                (self.x_lim[0], self.y_lim[0]),
                self.x_lim[1] - self.x_lim[0],
                self.y_lim[1] - self.y_lim[0],
                transform=ax.transData,
                fill=False,
                visible=False
            )
            ax.add_patch(clip_box)
        else:
            clip_box = None
        
        for i, obs in enumerate(obstacles):
            if obs['type'] == 'circle':
                circle = patches.Circle(
                    obs['center'], obs['radius'],
                    facecolor='#737AAF', alpha=0.8,
                    edgecolor='#5A6096', linewidth=1.5,
                    label='Obstacle' if i == 0 else '', zorder=3
                )
                # 应用裁剪，只显示在边界内的部分
                if clip_box is not None:
                    circle.set_clip_path(clip_box)
                ax.add_patch(circle)
            elif obs['type'] == 'rectangle':
                bounds = obs['bounds']
                width = bounds[1] - bounds[0]
                height = bounds[3] - bounds[2]
                rect = patches.Rectangle(
                    (bounds[0], bounds[2]), width, height,
                    facecolor='#737AAF', alpha=0.8,
                    edgecolor='#5A6096', linewidth=1.5,
                    label='Obstacle' if i == 0 else '', zorder=3
                )
                # 应用裁剪，只显示在边界内的部分
                if clip_box is not None:
                    rect.set_clip_path(clip_box)
                ax.add_patch(rect)
        
        # 绘制起点和目标
        ax.plot(start_pos[0], start_pos[1], 'o', markersize=10, 
                markerfacecolor='#F6B073', markeredgecolor='#E89A56', 
                linewidth=1.5, label='Starting Point', zorder=5)
        
        # 绘制终点（半透明圆形区域，虚线边缘）
        goal_circle = patches.Circle(
            goal_pos, success_threshold,
            facecolor='#2FBD9C', alpha=0.3,
            edgecolor='#26A085', linewidth=2,
            linestyle='--',  # 虚线
            label='End Point', zorder=5
        )
        ax.add_patch(goal_circle)
        # 在中心添加一个小标记点
        ax.plot(goal_pos[0], goal_pos[1], 'x', markersize=8,
                markeredgecolor='#26A085', markeredgewidth=2, zorder=6)
        
        # 初始化轨迹线和机器人位置
        trajectory_line, = ax.plot([], [], '-', color='#4A90E2', linewidth=2, alpha=0.8, 
                                   label='Trajectory', zorder=4)
        robot_dot, = ax.plot([], [], 'o', markersize=8, 
                            markerfacecolor='#FF6B6B', markeredgecolor='#E85555', 
                            linewidth=1.5, label='Current Position', zorder=6)
        
        # 时间文本
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           verticalalignment='top', fontsize=11, color='#313D78',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#313D78'))
        
        # 标签和样式设置
        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold', color='#313D78')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold', color='#313D78')
        
        # 去掉网格线
        ax.grid(False)
        
        # 设置边框颜色
        for spine in ax.spines.values():
            spine.set_edgecolor('#313D78')
            spine.set_linewidth(2)
        
        # 设置刻度颜色
        ax.tick_params(colors='#313D78', which='both')
        
        # 设置等比例
        ax.set_aspect('equal', adjustable='box')
        
        # 再次确保坐标轴范围符合设置
        if self.x_lim:
            ax.set_xlim(self.x_lim)
        if self.y_lim:
            ax.set_ylim(self.y_lim)
            
        # 图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), 
            by_label.keys(),
            loc='best',
            frameon=True,
            framealpha=0.95,
            edgecolor='#313D78',
            facecolor='white',
            fontsize=9,
            shadow=True,
            borderpad=0.8,
            labelspacing=0.6,
            handlelength=1.5,
            handleheight=0.7,
            ncol=1
        )
        
        # 动画更新函数
        def update(frame):
            # 更新轨迹
            trajectory_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
            # 更新机器人位置
            robot_dot.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
            # 更新时间
            time_text.set_text(f'Step: {frame+1}/{len(trajectory)}')
            return trajectory_line, robot_dot, time_text
        
        # 创建动画
        n_frames = len(trajectory)
        # 根据轨迹长度调整帧率，确保视频时长在3-10秒之间
        fps = max(10, min(30, n_frames // 5))
        
        anim = FuncAnimation(fig, update, frames=n_frames, 
                            interval=1000/fps, blit=True, repeat=False)
        
        # 保存视频
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(save_path, writer=writer)
            plt.close(fig)
            return True
        except Exception as e:
            print(f"警告: 无法保存视频 {save_path}: {e}")
            print("提示: 请确保已安装 ffmpeg (conda install ffmpeg 或 apt install ffmpeg)")
            plt.close(fig)
            return False

    def reset(self):
        """清空轨迹以开始新回合"""
        self._trajectory = []

    def close(self):
        try:
            plt.close(self._fig)
        except Exception:
            pass

    def _render_ocean_current(self, ax, ocean_current, obstacles):
        """渲染洋流流场，支持稀疏显示和避障"""
        if not (self.x_lim and self.y_lim):
            return

        # 创建稀疏网格 (8x8) - 更稀疏，箭头更清晰
        grid_size = 8
        x_range = np.linspace(self.x_lim[0], self.x_lim[1], grid_size)
        y_range = np.linspace(self.y_lim[0], self.y_lim[1], grid_size)
        X, Y = np.meshgrid(x_range, y_range)
        
        # 计算每个网格点的流速
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        valid_mask = np.ones_like(X, dtype=bool)  # 标记有效点
        
        # 障碍物检测（带安全边距）
        def is_near_obstacle(x, y, safety_margin=0.15):
            """检查点是否在障碍物内或附近
            
            Args:
                x, y: 检查点坐标
                safety_margin: 安全边距（米），箭头不会在障碍物这个范围内显示
            """
            if not obstacles:
                return False
            for obs in obstacles:
                if obs['type'] == 'circle':
                    dist = np.sqrt((x - obs['center'][0])**2 + (y - obs['center'][1])**2)
                    # 圆形障碍物：检查是否在半径+安全边距内
                    if dist < obs['radius'] + safety_margin:
                        return True
                elif obs['type'] == 'rectangle':
                    x_min, x_max, y_min, y_max = obs['bounds']
                    # 矩形障碍物：扩展边界
                    if (x_min - safety_margin <= x <= x_max + safety_margin and 
                        y_min - safety_margin <= y <= y_max + safety_margin):
                        return True
            return False

        # 计算所有网格点的速度
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # 检查是否在障碍物附近
                if is_near_obstacle(X[i, j], Y[i, j]):
                    valid_mask[i, j] = False
                    U[i, j] = np.nan
                    V[i, j] = np.nan
                else:
                    velocity = ocean_current.get_velocity(X[i, j], Y[i, j])
                    U[i, j] = velocity[0]
                    V[i, j] = velocity[1]
        
        # 计算流速大小（用于颜色映射）
        speed = np.sqrt(U**2 + V**2)
        
        # 找到有效的非零速度点
        has_flow = np.any(~np.isnan(U) & (speed > 0.01))
        
        if not has_flow:
            # 如果没有显著的流速，不绘制任何箭头
            return
        
        # 绘制矢量场 - 半透明箭头
        ax.quiver(X, Y, U, V, speed, 
                 cmap='winter_r', alpha=0.85,  # 使用PuBu配色，降低透明度
                 scale=3,
                 scale_units='xy',
                 width=0.003,
                 headwidth=3,      # 头部更小
                 headlength=4,   # 头部更短
                 zorder=2)  # 在背景之上，障碍物之下
