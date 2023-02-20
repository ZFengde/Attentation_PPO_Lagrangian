import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from attentation_ppo_lagrangian.common.on_policy_algorithm import OnPolicyAlgorithm
from attentation_ppo_lagrangian.common.policies import BasePolicy, ActorCriticPolicy

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, safe_mean

class Atte_PPO_Lagrangian(OnPolicyAlgorithm):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 20480,
        batch_size: int = 2048,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        cost_lim: float = 25,
        net_arch_dim: int = 64,
        obstacle_num: int = 5,
        use_sde: bool = False,
        use_constraint: bool = True,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            cost_lim=cost_lim,
            net_arch_dim=net_arch_dim,
            obstacle_num=obstacle_num,
            use_sde=use_sde,
            use_constraint=use_constraint,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.cost_lim = cost_lim
        self.net_arch_dim = net_arch_dim
        self.obstacle_num = obstacle_num
        self.use_constraint = use_constraint

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # Only for logger
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        if self.use_constraint:
            ep_cost_mean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            # Penalty Loss, this term we will only care about it at the begining of the episode
            penalty_loss = - self.policy.penalty_lambda * (ep_cost_mean - self.cost_lim)
            
            # Optimization step
            self.policy.lambda_optimizer.zero_grad()
            penalty_loss.backward()
            self.policy.lambda_optimizer.step()

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            update_per_epoch = 0
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, c_values, attention = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                c_values = c_values.flatten()
                
                # Normalize advantage
                # TODO, think about detach here
                advantages = rollout_data.advantages * attention
                c_advantages = rollout_data.c_advantages * (1 - attention)
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # so this is the normalize method that make mean to 0, while max of the advantaged value could be large
                    # c_advantages = (c_advantages - c_advantages.mean()) / (c_advantages.std() + 1e-8)
                    c_advantages = th.nn.functional.normalize(c_advantages, dim=0) # this this the normalization method use baselines max|v|p, while the max of the advantaged value no greater than 1

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                surr_cost = th.mean(ratio * c_advantages) - c_advantages.mean()
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                
                # surr_adv is very small while 
                surr_adv = th.min(policy_loss_1, policy_loss_2).mean()

                if self.use_constraint:
                    pi_objective =  surr_adv - self.policy.penalty_lambda * surr_cost
                    pi_objective = pi_objective / (1 + self.policy.penalty_lambda)
                    policy_loss = - pi_objective
                else:
                    policy_loss = - surr_adv

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                    c_values_pred = c_values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    c_values_pred = rollout_data.old_c_values + th.clamp(
                        c_values - rollout_data.old_c_values, -clip_range_vf, clip_range_vf
                    )
                
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                c_value_loss = F.mse_loss(rollout_data.c_returns, c_values_pred)
                value_losses.append(value_loss.item())

                # Attentaion loss calculated here
                # TODO, why this become inf, it's because c_returns could be 0
                # importance = th.softmax(th.stack((rollout_data.returns, rollout_data.c_returns)), dim=1)

                # attention_loss = F.mse_loss(rollout_data.returns/rollout_data.c_returns, attention.squeeze(0))

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # This become very large since c_value_loss is high
                if self.use_constraint:
                    # loss = policy_loss + self.vf_coef * (value_loss + c_value_loss) + 0.5 * attention_loss
                    loss = policy_loss + self.vf_coef * (value_loss + c_value_loss)
                else:
                    loss = policy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                update_per_epoch += 1

            if not continue_training:
                break

        self._n_updates += self.n_epochs * update_per_epoch
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
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
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self,
        total_timesteps: int = None,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO_Lagrangian",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "Atte_PPO_Lagrangian":

        return super().learn(
            total_timesteps=self.env.num_envs * self.n_steps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def test(self, env):
        
        while True:
            obs = env.reset()
            ep_reward = 0
            ep_len = 0
            ep_cost = 0
            while True:
                env.render()
                with th.no_grad():
                    action = self.policy.predict(obs)[0]

                clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, done, info = env.step(clipped_action)

                ep_reward += reward
                ep_cost += info['cost']
                ep_len += 1
                if done:
                    print(ep_len, ep_reward, ep_cost)
                    break