"""路径规划任务评估脚本

用于评估训练好的模型并生成详细的性能报告
"""
import argparse
import os
import time
import numpy as np
import sys
import pathlib
import gc
try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from task_path.metrics import (
    PathPlanningMetrics,
    compute_episode_metrics,
)
from task_path.utils import (
    make_env,
    load_model,
    Logger,
    get_config,
    save_config,
    _CUSTOM_ALGORITHM_REGISTRY
)
from env.path_planning_2d import PathPlanning2DEnv
from env.renderer_2d import Renderer2D

def evaluate_model(
    model_path: str,
    config_name: str = "default",
    n_episodes: int = 100,
    render: bool = False,
    save_dir: Optional[str] = None,
    seed: int = 42,
    deterministic: bool = True,
    device: str = 'auto',
    verbose: bool = True,
) -> Dict[str, Any]:
    """评估训练好的模型
    
    Args:
        model_path: 模型文件路径
        config_name: 环境配置名称
        n_episodes: 评估episode数
        render: 是否渲染（保存每个episode最终画面，每10个episode创建动画视频）
        save_dir: 保存结果的目录
        seed: 随机种子
        deterministic: 是否使用确定性策略
        device: 设备 ('auto', 'cpu', 'cuda')
        verbose: 是否打印详细信息
        
    Returns:
        评估结果字典
    """
    print(f'保存路径: {save_dir}')
    algo_name = ''
    model_path_lower = model_path.lower()
    for name in _CUSTOM_ALGORITHM_REGISTRY:
        if name in model_path_lower:
            algo_name = name
            break
    
    if algo_name is None:
        raise ValueError(
            f"Cannot infer algorithm from path: {model_path}. "
            f"Please specify algo_name parameter."
        )

    if algo_name == 'ppo':
        device = 'cpu'
    else:
        device = 'cuda'

    # 创建日志
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "eval_log.txt")
    else:
        log_file = None
    
    logger = Logger(log_file=log_file, verbose=verbose)
    
    logger.section("路径规划模型评估")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"环境配置: {config_name}")
    logger.info(f"评估episodes: {n_episodes}")
    logger.info(f"确定性策略: {deterministic}")
    logger.info(f"随机种子: {seed}")
    logger.info(f"计算设备: {device}")
    
    # 加载环境配置
    config = get_config(config_name)
    
    # 保存配置
    if save_dir:
        config_save_path = os.path.join(save_dir, "eval_config.json")
        eval_config = {
            'name': config.get('name', config_name),
            'description': config.get('description', ''),
            'env': config['env'],
            'model_path': model_path,
            'eval_settings': {
                'n_episodes': n_episodes,
                'deterministic': deterministic,
                'seed': seed,
            }
        }
        save_config(eval_config, config_save_path)
    
    # 加载模型
    logger.info("\n加载模型...")
    
    try:
        model = load_model(model_path, algo_name=algo_name, device=device)
        logger.info("模型加载成功")
        logger.info(f"使用设备: {model.device}")
        
        # 创建环境
        logger.info("创建环境...")
        env = PathPlanning2DEnv(config=config['env'], record_trajectory=True)  # 评估时启用轨迹记录
        
        # 初始化渲染器（如果需要渲染）
        renderer = None
        if render:
            renderer = Renderer2D(
                width=8, 
                height=8,
                x_lim=(env.x_min, env.x_max),
                y_lim=(env.y_min, env.y_max)
            )
        
        # 初始化指标跟踪器
        metrics_tracker = PathPlanningMetrics()
        
        # 评估循环
        logger.section("开始评估")
        
        start_time = time.time()
        
        for episode in range(n_episodes):
            obs, info = env.reset(seed=seed + episode)
            
            done = False
            truncated = False
            total_reward = 0.0
            steps = 0
            
            # 记录轨迹和动作
            trajectory = [obs[:2].copy()]
            actions = []
            
            # 记录episode信息（用于视频）
            start_position = env.start_position
            goal_position = env.goal_position
            
            episode_start_time = time.time()
            
            while not (done or truncated):
                # 预测动作
                action, _states = model.predict(obs, deterministic=deterministic)
                
                # 执行动作
                obs, reward, done, truncated, info = env.step(action)
                
                # 记录
                total_reward += reward
                steps += 1
                trajectory.append(obs[:2].copy())
                actions.append(action.copy())
            
            episode_time = time.time() - episode_start_time
            
            # 转换轨迹为numpy数组
            trajectory = np.array(trajectory)
            actions = np.array(actions)
            
            # 保存最终画面（所有episode）
            if render and save_dir and renderer:
                # 设置轨迹到渲染器
                renderer.reset()
                renderer.set_trajectory(trajectory)
                
                # 获取障碍物信息（字典格式）
                obstacles_info = [obs.to_dict() for obs in env.map.get_obstacles()]
                
                # 保存最终帧
                render_path = os.path.join(
                    save_dir, 'renders', f'episode_{episode + 1:03d}_final.png'
                )
                # 确保目录存在
                os.makedirs(os.path.dirname(render_path), exist_ok=True)
                
                # 直接调用渲染器保存画面
                renderer.render(
                    current_pos=tuple(obs[:2]),
                    target_pos=goal_position,
                    obstacles=obstacles_info,
                    ocean_current=env.map.get_ocean_current(),
                    show_current_field=True,
                    show=False,
                    save_path=render_path,
                    success_threshold=env.success_threshold
                )
            
            # 计算episode指标
            obstacles = env.map.get_obstacles()
            
            episode_metrics = compute_episode_metrics(
                trajectory=trajectory,
                actions=actions,
                obstacles=obstacles,
                success=info.get('success', False),
                collision=info.get('collision', False),
                episode_return=total_reward,
                episode_length=steps,
                computation_time=episode_time,
            )
            
            # 更新指标
            metrics_tracker.update(episode_metrics)
            
            # 为特定episode创建动画视频（episode 1, 11, 21, 31...）
            if render and save_dir and ((episode + 1) % 10 == 1):
                video_dir = os.path.join(save_dir, 'videos')
                video_path = os.path.join(
                    video_dir,
                    f'episode_{episode + 1:03d}_animation.mp4'
                )
                logger.info(f"创建动画视频: Episode {episode + 1}...")
                
                obstacles_info = [obs.to_dict() for obs in obstacles]

                success = renderer.render_animation(
                    trajectory=trajectory,
                    obstacles=obstacles_info,
                    start_pos=start_position,
                    goal_pos=goal_position,
                    save_path=video_path,
                    episode_num=episode + 1,
                    info=info,
                    ocean_current=env.map.get_ocean_current(),
                    show_current_field=True,
                    success_threshold=env.success_threshold
                )
                
                if success:
                    logger.info(f"  ✓ 视频已保存: {video_path}")
                else:
                    logger.info(f"  ✗ 视频保存失败")
            
            # 定期打印进度
            if (episode + 1) % 10 == 0 or episode == 0:
                success_status = "✓" if info.get('success', False) else "✗"
                logger.info(
                    f"Episode {episode + 1:3d}/{n_episodes} | "
                    f"Success: {success_status} | "
                    f"Steps: {steps:4d} | "
                    f"Reward: {total_reward:7.2f} | "
                    f"Path Length: {episode_metrics.get('path_length', 0):.2f}m"
                )
        
        total_time = time.time() - start_time
        
        # 打印总结
        logger.section("评估完成")
        logger.info(f"总耗时: {total_time:.2f}s")
        logger.info(f"平均每episode: {total_time / n_episodes:.2f}s")
        
        # 打印详细指标
        metrics_tracker.print_summary()
        
        # 获取汇总统计
        summary = metrics_tracker.get_summary()
        summary['total_evaluation_time'] = total_time
        summary['config_name'] = config_name
        summary['model_path'] = model_path
        
        # 保存结果
        if save_dir:
            import json
            results_path = os.path.join(save_dir, "eval_results.json")
            with open(results_path, 'w') as f:
                # 将numpy类型转换为Python类型
                summary_serializable = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in summary.items()
                }
                json.dump(summary_serializable, f, indent=2)
            logger.info(f"结果已保存到: {results_path}")
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {str(e)}")
        raise
        
    finally:
        # 清理资源
        if 'env' in locals():
            env.close()
        
        if 'renderer' in locals() and renderer:
            renderer.close()
            
        # 显式删除大对象
        if 'model' in locals():
            del model
        if 'env' in locals():
            del env
        if 'renderer' in locals():
            del renderer
            
        # 强制垃圾回收
        gc.collect()
        
        # 清理GPU缓存
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("环境已关闭，内存已释放")
    
    return summary


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="评估路径规划任务的训练模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本参数
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='训练好的模型路径'
    )
    parser.add_argument(
        '--algo', type=str, default=None,
        choices=['ppo', 'sac', 'td3'],
        help='算法名称（如果为None则自动推断）'
    )
    
    # 环境配置（动态获取所有可用配置）
    parser.add_argument(
        '--config', type=str, default='default',
        help='环境配置名称或完整路径（必需）'
    )
    
    # 评估参数
    parser.add_argument(
        '--n-episodes', type=int, default=100,
        help='评估episode数'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--deterministic', action='store_true',
        help='使用确定性策略'
    )
    parser.add_argument(
        '--no-deterministic', dest='deterministic', action='store_false',
        help='使用随机策略'
    )
    parser.set_defaults(deterministic=True)
    
    # 可视化
    parser.add_argument(
        '--render', action='store_true',
        help='渲染评估过程（保存每个episode最终画面，每10个episode创建动画视频）'
    )
    
    # 保存选项
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='保存评估结果的目录'
    )
    
    # 设备参数
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='计算设备: auto(自动选择), cpu(强制CPU), cuda(使用GPU)'
    )
    
    args = parser.parse_args()
    
    # 如果未指定保存目录，使用模型所在实验目录
    if args.save_dir is None:
        from datetime import datetime
        
        # 获取模型路径：例如 task_path/results/ppo_default_20251102_223235/models/best_model.zip
        model_path = os.path.abspath(args.model_path)
        # 获取 models 目录：task_path/results/ppo_default_20251102_223235/models
        models_dir = os.path.dirname(model_path)
        # 获取实验目录：task_path/results/ppo_default_20251102_223235
        experiment_dir = os.path.dirname(models_dir)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 检查路径是否符合预期结构（models 目录存在）
        if os.path.basename(models_dir) == 'models':
            # 在实验目录下创建 eval 子目录
            args.save_dir = os.path.join(experiment_dir, 'eval', timestamp)
        else:
            assert False, (
                "无法推断保存目录。请手动指定 --save-dir 参数。"
            )
    
    # 运行评估
    evaluate_model(
        model_path=args.model_path,
        config_name=args.config,
        n_episodes=args.n_episodes,
        render=args.render,
        save_dir=args.save_dir,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
    )
    
    print("\n评估完成！")
