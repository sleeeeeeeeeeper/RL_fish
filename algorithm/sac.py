"""Soft Actor-Critic (SAC) 算法实现
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
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from stable_baselines3.common.save_util import load_from_zip_file


class SAC(OffPolicyAlgorithm):
    """Soft Actor-Critic (SAC)
    
    继承自 OffPolicyAlgorithm，完全兼容 SB3 生态
    支持 Callback、Logger、VecEnv、Monitor 等所有 SB3 工具
    """
    
    policy_aliases = {
        "MlpPolicy": SACPolicy,
    }
    
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """初始化 SAC 算法
        
        Args:
            policy: 策略网络类型 ('MlpPolicy' 或自定义 SACPolicy)
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
            ent_coef: 熵系数 ('auto' 表示自动调整)
            target_update_interval: 目标网络更新间隔
            target_entropy: 目标熵 ('auto' 表示 -dim(A))
            use_sde: 是否使用状态依赖探索
            sde_sample_freq: SDE 采样频率
            use_sde_at_warmup: 预热期是否使用 SDE
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
        
        # SAC 特有参数
        self.target_update_interval = target_update_interval
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.ent_coef_optimizer = None
        
        if _init_setup_model:
            self._setup_model()

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

    
    def _setup_model(self) -> None:
        """设置模型（网络、优化器、buffer等）
        
        由 SB3 基类在 learn() 开始前自动调用
        """
        super()._setup_model()
        
        # 创建Actor和Critic（由SACPolicy实现）
        self._create_aliases()

        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        
        # 初始化目标熵
        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
        else:
            self.target_entropy = float(self.target_entropy)
        
        # 初始化熵系数
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # 注意: 这里使用 target_entropy 的相反数作为初始值
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            
            # 可学习的对数熵系数
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)
    
    def _create_aliases(self) -> None:
        """创建别名以方便访问网络"""
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
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
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        
        # 更新学习率
        self._update_learning_rate(optimizers)
        
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        
        for gradient_step in range(gradient_steps):
            # 从replay buffer采样
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # 重要: 使用detach避免梯度回传到Actor
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())


            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # 训练Critic
            # 计算目标Q值
            with th.no_grad():
                # 采样下一个动作
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # 计算下一个Q值
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # 添加熵项
                ent_coef = self.ent_coef if isinstance(self.ent_coef, float) else th.exp(self.log_ent_coef.detach())
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # 计算目标Q值
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            
            # 获取当前Q值估计
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            
            # Critic损失
            critic_loss = 0.5 * sum([th.nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())
            
            # 优化Critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # 计算Actor损失
            # 最小化 E[α·log π(a|s) - Q(s,a)]
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # 优化Actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            # 软更新目标网络
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        
        self._n_updates += gradient_steps
        
        # 记录日志
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
