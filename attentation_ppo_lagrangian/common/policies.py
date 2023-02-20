"""Policies: abstract base class and concrete implementations."""

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import time
import dgl
import dgl.function as fn

import math
import gym
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy

class multi_head(nn.Module):
    def __init__(self, in_dim, dim_feedforward):
        super().__init__()
        self.feature_layer_1 = nn.Linear(in_dim, dim_feedforward)
        self.multi_head_1 = nn.MultiheadAttention(embed_dim=dim_feedforward, 
                                                num_heads=4,
                                                batch_first=True)
        
        self.multi_head_2 = nn.MultiheadAttention(embed_dim=dim_feedforward, 
                                                num_heads=4,
                                                batch_first=True)
        
        self.output_layer = nn.Linear(dim_feedforward, 2)
        
    def forward(self, input):
        x1 = th.relu(self.feature_layer_1(input))
        x, _ = self.multi_head_1(x1, x1, x1)
        x2 = x1 + th.nn.functional.normalize(x, dim=1)

        x, _ = self.multi_head_2(x2, x2, x2)
        x = x2 + th.nn.functional.normalize(x, dim=1)

        output =  self.output_layer(x)

        return output
    
class attention_network(nn.Module):
    def __init__(self, in_dim, dim_feedforward):
        super().__init__()
        self.linear_layer_1 = nn.Linear(in_dim, dim_feedforward)
        self.linear_layer_2 = nn.Linear(dim_feedforward, dim_feedforward)
        self.output_layer = nn.Linear(dim_feedforward, 1)
        
    def forward(self, input):

        x = th.tanh(self.linear_layer_1(input))
        x = th.tanh(self.linear_layer_2(x))
        # make sure attention is among 0-1
        attention = (th.tanh(self.output_layer(x)) + 1) / 2

        return attention

class attention_mask_layer(nn.Module):
    def __init__(self, feat_dim, out_dim):
        super().__init__()
        self.policy_net = nn.Linear(feat_dim, out_dim)
        self.c_policy_net = nn.Linear(feat_dim, out_dim)

    def forward(self, feature, attention): # 3, 64 & 3, 1
        output = th.relu((1 + attention) * self.policy_net(feature) + (1 - attention) * self.c_policy_net(feature))
        return output

class ActorCriticPolicy(BasePolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        cost_lim: float = 25.,
        net_arch_dim: int = 64,
        obstacle_num: int = 5,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        self.cost_lim = cost_lim
        self.net_arch_dim = net_arch_dim
        self.obstacle_num = obstacle_num
        # Default network architecture, from stable-baselines
        net_arch = [dict(pi=[self.net_arch_dim, self.net_arch_dim], vf=[self.net_arch_dim, self.net_arch_dim])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        # Build lagrangian lambda term here
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        # Lagrangian method here
        self.lag_lambda_param = th.tensor(0.5, requires_grad=True, device=device)
        self.c_value_net = nn.Sequential(
                                        nn.Linear(self.features_dim, self.net_arch_dim),
                                        nn.Tanh(),
                                        nn.Linear(self.net_arch_dim, self.net_arch_dim),
                                        nn.Tanh(),
                                        nn.Linear(self.net_arch_dim, 1),).to(device)
        self.lambda_optimizer = self.optimizer_class([self.lag_lambda_param], lr=4e-3)

        # Attentation mechanism here
        self.attentation_net = attention_network(self.features_dim, 64)
        self.attention_mask = attention_mask_layer(64, 64)
        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        
        self.multi_head = multi_head(
            self.features_dim, 64
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.c_value_net: np.sqrt(2),
                self.attentation_net: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # TODO, need to watch what parameters are included
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        attention = self.attentation_net(features)
        latent_pi, latent_vf = self.mlp_extractor(features)
        masked_latent_pi = self.attention_mask(latent_pi, attention)
        
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        # Calculate c_values
        c_values = self.c_value_net(features)
        distribution = self._get_action_dist_from_latent(masked_latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob, c_values, attention

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        attention = self.attentation_net(features)
        latent_pi, latent_vf = self.mlp_extractor(features)
        masked_latent_pi = self.attention_mask(latent_pi, attention)
        distribution = self._get_action_dist_from_latent(masked_latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        c_values = self.c_value_net(features)
        return values, log_prob, distribution.entropy(), c_values, attention

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        attention = self.attentation_net(features)
        latent_pi = self.mlp_extractor.forward_actor(features)
        masked_latent_pi = self.attention_mask(latent_pi, attention)

        return self._get_action_dist_from_latent(masked_latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        c_values = self.c_value_net(features)
        
        return self.value_net(latent_vf), c_values

    @property
    def penalty_lambda(self):
        return F.softplus(self.lag_lambda_param)
