"""Proximal Policy Optimization (PPO) 算法实现示例，
自定义算法的示例，要满足 Stable Baselines3 (SB3) 框架的要求，
须继承自 OnPolicyAlgorithm 或 OffPolicyAlgorithm，
且仅重写 __init__() 和 train() 方法。
该实现完全兼容 SB3 生态，支持 Callback、Logger、VecEnv、Monitor 等所有 SB3 工具。
"""

import torch as th
import numpy as np
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.save_util import load_from_zip_file
from gymnasium import spaces


class PPO(OnPolicyAlgorithm):
    """Proximal Policy Optimization (PPO)
    
    继承自 OnPolicyAlgorithm，完全兼容 SB3 生态
    支持 Callback、Logger、VecEnv、Monitor 等所有 SB3 工具
    """
    
    policy_aliases = {
        "MlpPolicy": ActorCriticPolicy,
    }
    
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Optional[Union[float, Schedule]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """初始化 PPO 算法
        
        Args:
            policy: 策略网络类型 ('MlpPolicy' 或自定义 ActorCriticPolicy)
            env: 训练环境
            learning_rate: 学习率 (可以是函数)
            n_steps: 每次 rollout 收集的步数
            batch_size: 小批量大小
            n_epochs: 每次 rollout 数据更新的 epoch 数
            gamma: 折扣因子
            gae_lambda: GAE 参数 λ
            clip_range: 策略裁剪范围 ε
            clip_range_vf: 价值函数裁剪范围 (None 表示不裁剪)
            normalize_advantage: 是否标准化优势函数
            ent_coef: 熵正则化系数
            vf_coef: 价值损失系数
            max_grad_norm: 梯度裁剪阈值
            use_sde: 是否使用状态依赖探索
            sde_sample_freq: SDE 采样频率
            target_kl: 目标 KL 散度 (用于早停)
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
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,  # 我们手动初始化
        )
        
        # PPO 特有参数
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        """设置模型（网络、优化器、buffer等）
        
        由 SB3 基类在 learn() 开始前自动调用
        """
        super()._setup_model()
        
        # 初始化 schedules（学习率、clip_range 可以是函数）
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive"
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

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

    
    def train(self) -> None:
        """训练策略和价值网络，来自Stable Baselines3源码
        
        由 SB3 基类在每次 rollout 后自动调用
        这是算法的核心：从 rollout_buffer 取数据并更新网络
        """
        # 切换到训练模式
        self.policy.set_training_mode(True)
        
        # 更新学习率和 clip_range
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        
        # 如果使用价值裁剪
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        
        # 统计信息
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        
        continue_training = True
        
        # 多个 epoch 更新
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # 遍历所有小批量
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # 离散动作空间
                    actions = rollout_data.actions.long().flatten()
                
                # 评估动作
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                
                # 标准化优势函数
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 策略损失 (Clipped Surrogate Objective)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                
                # 记录策略损失
                pg_losses.append(policy_loss.item())
                
                # 计算 clip fraction
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                # 价值损失
                if self.clip_range_vf is None:
                    # 不裁剪
                    values_pred = values
                else:
                    # 裁剪价值函数
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                
                # 熵损失
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                
                entropy_losses.append(entropy_loss.item())
                
                # 总损失
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # 计算近似 KL 散度 (用于早停)
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                
                # 提前停止 (如果 KL 散度过大)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                
                # 梯度更新
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            
            self._n_updates += 1
            if not continue_training:
                break
        
        # 计算 explained variance (用于评估价值函数质量)
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        
        # 记录日志 (SB3 的 Logger 会自动处理)
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
