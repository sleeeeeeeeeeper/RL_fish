"""路径规划任务训练脚本

使用Stable-Baselines3训练强化学习模型
支持PPO、SAC、TD3等算法
"""
import argparse
import os
import time
import sys
import pathlib
import gc
try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from datetime import datetime
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from task_path.utils import (
    make_vec_env,
    create_model,
    save_model,
    create_experiment_dir,
    Logger,
    set_random_seed,
    print_model_info,
    format_time,
    get_config,
    save_config,
)


def train(args):
    """训练主函数
    
    Args:
        args: 命令行参数
        
    Returns:
        实验目录路径
    """
    # 获取配置文件（所有参数从配置文件读取）
    config = get_config(args.config)
    
    # 从配置文件获取算法名称
    if 'algorithm' in config and len(config.get('algorithm', {})) > 0:
        # 使用配置文件中的第一个算法
        args.algo = list(config['algorithm'].keys())[0]
    else:
        raise ValueError("配置文件中未指定算法，且命令行未提供 --algo 参数")
    
    # 验证算法是否在配置文件中
    if 'algorithm' not in config or args.algo not in config['algorithm']:
        raise ValueError(f"配置文件中未找到算法 '{args.algo}' 的配置")
    
    # 从配置文件读取所有参数
    training_config = config.get('training', {})
    if not training_config:
        raise ValueError("配置文件中缺少 'training' 配置部分")
    
    # 必需的训练参数
    required_params = ['total_timesteps', 'n_envs', 'seed', 'eval_freq', 'n_eval_episodes', 'checkpoint_freq', 'log_interval']
    for param in required_params:
        if param not in training_config:
            raise ValueError(f"配置文件的 'training' 部分缺少必需参数: {param}")
    
    # 获取算法超参数
    algo_hyperparams = config['algorithm'][args.algo].copy()
    
    # 智能设备选择：对于MlpPolicy，强制使用CPU（避免GPU-CPU数据传输开销）
    if args.device == 'auto':
        if args.algo == 'ppo': # PPO算法默认使用CPU
            args.device = 'cpu'
        else:
            args.device = 'cuda'  # 默认使用GPU
    
    # 设置随机种子
    set_random_seed(training_config['seed'])
    
    # 创建实验目录
    # 提取配置名称（去除路径和.json后缀）
    if args.config.endswith('.json'):
        config_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        config_name = args.config
    experiment_name = f"{args.algo}_{config_name}"
    exp_dir = create_experiment_dir(
        base_dir=args.save_dir,
        experiment_name=experiment_name,
        use_timestamp=True
    )
    
    # 创建日志
    log_file = os.path.join(exp_dir, "logs", "train.log")
    logger = Logger(log_file=log_file, verbose=True)
    
    # 打印训练配置
    logger.section("路径规划任务训练")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"算法: {args.algo.upper()}")
    logger.info(f"总训练步数: {training_config['total_timesteps']:,}")
    logger.info(f"并行环境数: {training_config['n_envs']}")
    logger.info(f"随机种子: {training_config['seed']}")
    logger.info(f"计算设备: {args.device} (MlpPolicy在CPU上性能更好)")
    logger.info(f"实验目录: {exp_dir}")
    
    # 保存配置到实验目录
    config_save_path = os.path.join(exp_dir, "config.json")
    save_config(config, config_save_path)
    
    # 创建训练环境
    logger.info("\n创建训练环境...")
    train_env = make_vec_env(
        config=config,
        n_envs=training_config['n_envs'],
        seed=training_config['seed'],
        log_dir=os.path.join(exp_dir, "logs", "train_monitor"),
    )
    logger.info(f"训练环境创建完成 ({training_config['n_envs']} 个并行环境)")
    
    # 创建评估环境
    logger.info("创建评估环境...")
    eval_env = make_vec_env(
        config=config,
        n_envs=1,
        seed=training_config['seed'] + 999,
        log_dir=os.path.join(exp_dir, "logs", "eval_monitor"),
    )
    logger.info("评估环境创建完成")
    
    # 创建模型
    logger.info(f"\n初始化 {args.algo.upper()} 模型...")
    logger.info(f"实现方式: {'自定义实现' if args.use_custom else 'Stable-Baselines3'}")
    tensorboard_log = os.path.join(exp_dir, "logs", "tensorboard")
    
    model = create_model(
        algo_name=args.algo,
        env=train_env,
        hyperparameters=algo_hyperparams,
        tensorboard_log=tensorboard_log,
        seed=training_config['seed'],
        device=args.device,
        verbose=1,
        use_custom=args.use_custom,
    )
    
    logger.info("模型创建完成")
    logger.info(f"使用设备: {model.device}")
    if not args.use_custom:
        print_model_info(model)
    
    # 创建回调
    callbacks = []
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(exp_dir, "models"),
        log_path=os.path.join(exp_dir, "logs", "eval"),
        eval_freq=max(training_config['eval_freq'] // training_config['n_envs'], 1),
        n_eval_episodes=training_config['n_eval_episodes'],
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks.append(eval_callback)
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=max(training_config['checkpoint_freq'] // training_config['n_envs'], 1),
        save_path=os.path.join(exp_dir, "models", "checkpoints"),
        name_prefix=f"{args.algo}_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    
    callback = CallbackList(callbacks)
    
    # 提示训练中的"卡顿"是正常现象
    logger.info("\n训练提示:")
    logger.info(f"  - 每 {training_config['eval_freq']:,} 步会暂停训练进行评估（约10-15秒）")
    logger.info(f"  - 每 {training_config['checkpoint_freq']:,} 步会保存检查点（约1-2秒）")
    logger.info(f"  - PPO显示的速度包含模型更新时间，环境纯运行速度约4000 steps/s")
    logger.info(f"  - 预计完成时间: {training_config['total_timesteps'] / (2000 * training_config['n_envs']) / 60:.1f} 分钟\n")
    
    # 开始训练
    logger.section("开始训练")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=training_config['total_timesteps'],
            callback=callback,
            progress_bar=True,
            log_interval=training_config['log_interval'],
        )
        
        training_time = time.time() - start_time
        
        logger.section("训练完成")
        logger.info(f"训练耗时: {format_time(training_time)}")
        logger.info(f"平均速度: {training_config['total_timesteps'] / training_time:.0f} steps/s")
        
        # 保存最终模型
        final_model_path = os.path.join(exp_dir, "models", "final_model")
        save_model(
            model,
            final_model_path,
            metadata={
                'algorithm': args.algo,
                'config': args.config,
                'total_timesteps': training_config['total_timesteps'],
                'training_time': training_time,
                'timestamp': datetime.now().isoformat(),
            }
        )
        logger.info(f"最终模型已保存: {final_model_path}.zip")
        
    except KeyboardInterrupt:
        logger.warning("\n训练被用户中断")
        training_time = time.time() - start_time
        logger.info(f"已训练时间: {format_time(training_time)}")
        
        # 保存中断的模型
        interrupted_model_path = os.path.join(exp_dir, "models", "interrupted_model")
        save_model(model, interrupted_model_path)
        logger.info(f"中断模型已保存: {interrupted_model_path}.zip")
    
    except Exception as e:
        logger.error(f"\n训练过程中发生错误: {str(e)}")
        raise
    
    finally:
        # 清理环境和释放内存
        if 'train_env' in locals():
            train_env.close()
        if 'eval_env' in locals():
            eval_env.close()
            
        # 显式删除大对象
        if 'model' in locals():
            del model
        if 'train_env' in locals():
            del train_env
        if 'eval_env' in locals():
            del eval_env
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理GPU缓存
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("环境已关闭，内存已释放")
    
    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="路径规划任务强化学习训练 - 所有参数从配置文件读取",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数：配置文件
    parser.add_argument(
        '--config', type=str, required=True,
        help=f'配置文件名称或完整路径（必需）'
    )
    
    # 可选参数：算法实现
    parser.add_argument(
        '--use-custom', action='store_true',
        help='使用自定义算法实现（默认使用Stable-Baselines3）'
    )
    
    # 可选参数：保存目录
    parser.add_argument(
        '--save-dir', type=str, default='task_path/results/',
        help='模型和日志保存根目录'
    )
    
    # 可选参数：设备
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='计算设备: auto(自动选择), cpu(强制CPU), cuda(使用GPU)'
    )
    
    args = parser.parse_args()
    
    # 运行训练
    try:
        exp_dir = train(args)
        
        print(f"\n{'='*60}")
        print("训练完成！")
        print(f"实验目录: {exp_dir}")
        print(f"{'='*60}\n")
        
        # 提示如何评估
        best_model_path = os.path.join(exp_dir, "models", "best_model.zip")
        print("评估最佳模型:")
        print(f"  python task_path/eval.py --model-path {best_model_path} --config {args.config} --render")
        print()
        
    except ValueError as e:
        print(f"\n错误: {e}")
        print("\n提示: 请确保配置文件包含所有必需的参数：")
        print("  - env: 环境配置")
        print("  - algorithm: 算法配置（至少包含一个算法：ppo/sac/td3）")
        print("  - training: 训练配置（total_timesteps, n_envs, seed等）")
        print(f"\n示例: python task_path/train.py --config none-default --algo ppo")
        sys.exit(1)