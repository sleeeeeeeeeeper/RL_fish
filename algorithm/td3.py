"""Twin Delayed Deep Deterministic Policy Gradient (TD3) 算法实现
自定义算法的示例，要满足 Stable Baselines3 (SB3) 框架的要求，
须继承自 OnPolicyAlgorithm 或 OffPolicyAlgorithm，
且仅重写 __init__() 和 train() 方法。
该实现完全兼容 SB3 生态，支持 Callback、Logger、VecEnv、Monitor 等所有 SB3 工具。
"""

import torch as th
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from stable_baselines3.common.save_util import load_from_zip_file


class TD3(OffPolicyAlgorithm):
    """Twin Delayed DDPG (TD3)
    
    继承自 OffPolicyAlgorithm，完全兼容 SB3 生态
    支持 Callback、Logger、VecEnv、Monitor 等所有 SB3 工具
    """
    
    policy_aliases = {
        "MlpPolicy": TD3Policy,
    }
    
    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """初始化 TD3 算法
        
        Args:
            policy: 策略网络类型 ('MlpPolicy' 或自定义 TD3Policy)
            env: 训练环境
            learning_rate: 学习率
            buffer_size: 经验回放缓冲区大小
            learning_starts: 开始训练前收集的步数
            batch_size: 小批量大小
            tau: 软更新系数
            gamma: 折扣因子
            train_freq: 训练频率
            gradient_steps: 每次训练的梯度步数
            action_noise: 动作噪声
            optimize_memory_usage: 优化内存使用
            policy_delay: Actor更新延迟(每多少步更新一次)
            target_policy_noise: 目标策略噪声标准差
            target_noise_clip: 目标策略噪声裁剪范围
            tensorboard_log: TensorBoard 日志目录
            policy_kwargs: 策略网络的额外参数
            verbose: 日志级别
            seed: 随机种子
            device: 设备
            _init_setup_model: 是否初始化模型
        """
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
        )
        
        # TD3 特有参数
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        """设置模型（网络、优化器、buffer等）
        
        由 SB3 基类在 learn() 开始前自动调用
        """
        super()._setup_model()
        self._create_aliases()
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    @classmethod
    def load(cls, path, env=None, device='auto', **kwargs):
        """加载模型，自动处理参数不匹配（如多余的优化器参数）"""
        try:
            return super().load(path, env=env, device=device, **kwargs)
        except (ValueError, KeyError) as e:
            # check if it is the specific ValueError we are looking for
            if isinstance(e, ValueError) and "Names of parameters do not match" not in str(e):
                raise e
            
            # 如果是参数不匹配（通常是因为保存了优化器状态但加载时不需要），
            # 或者是KeyError（SB3内部处理不匹配时可能抛出policy.optimizer缺失）
            # 则尝试手动加载并过滤参数
            print(f"[Warning] Loading model with parameter mismatch handling: {e}")
            print("[Info] Attempting to load only matching parameters (policy)...")
            
            data, params, pytorch_variables = load_from_zip_file(
                path, device=device, custom_objects=kwargs.get("custom_objects")
            )
            
            if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data.get("policy_kwargs"):
                raise ValueError("The policy_kwargs passed must be the same as the one used for training.")
            
            model = cls(
                policy=data["policy_class"],
                env=env,
                device=device,
                _init_setup_model=False,
            )
            
            # 加载属性
            model.__dict__.update(data)
            model.__dict__.update(kwargs)
            
            model._setup_model()
            
            # 过滤参数：只保留当前模型期望的参数（通常是'policy'）
            expected_params = model.get_parameters()
            filtered_params = {}
            for k in expected_params.keys():
                if k in params:
                    filtered_params[k] = params[k]
            
            # 强制加载
            model.set_parameters(filtered_params, exact_match=True, device=device)
            
            return model

    
    
    def _create_aliases(self) -> None:
        """创建别名以方便访问网络"""
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """训练策略和价值网络，来自Stable Baselines3源码
        
        由 SB3 基类在收集经验后自动调用
        
        Args:
            gradient_steps: 梯度更新步数
            batch_size: 小批量大小
        """
        # 切换到训练模式
        self.policy.set_training_mode(True)
        
        # 更新学习率
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        self._update_learning_rate(optimizers)
        
        actor_losses, critic_losses = [], []
        
        for gradient_step in range(gradient_steps):
            self._n_updates += 1
            
            # 从replay buffer采样
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # 训练Critic
            with th.no_grad():
                # 目标策略平滑: 添加噪声到目标动作
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                
                # 计算下一个Q值(取最小值)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            
            # 获取当前Q值估计
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            
            # Critic损失
            critic_loss = sum([th.nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())
            
            # 优化Critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # 延迟更新Actor
            if self._n_updates % self.policy_delay == 0:
                # Actor损失: -E[Q(s, π(s))]
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())
                
                # 优化Actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                
                # 软更新目标网络
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        # 记录日志
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
