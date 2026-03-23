"""Proximal Policy Optimization (PPO) algorithm implementation.

Improvements over baseline PPO, aligned with tianshou v1.1.0:
  - Per-mini-batch advantage normalization (not global)
  - Value function clipping (arXiv:1811.02553v3)
  - Dual-clip PPO (arXiv:1912.09729)
  - Recompute advantage each epoch (arXiv:2006.05990)
  - Default actor uses state-independent log_std with orthogonal init
"""

from typing import Any, Dict, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from seenerl.algorithms.base import BaseAlgorithm
from seenerl.algorithms.registry import register_algorithm
from seenerl.models import build_actor_model, build_value_model
from seenerl.utils import resolve_device


@register_algorithm("PPO", trainer_kind="on_policy")
class PPO(BaseAlgorithm):
    """
    PPO with clipped surrogate objective.

    On-policy algorithm: collects rollout_steps of data using the current
    policy, then performs ppo_epoch optimization epochs on the collected data,
    splitting it into num_mini_batch mini-batches per epoch.

    Supports:
      - Clipped surrogate objective (standard PPO)
      - Value function clipping (prevents large value updates)
      - Dual-clip PPO (extra clipping for negative advantages)
      - Per-mini-batch advantage normalization
      - Recompute advantage at each optimization epoch
    """

    def __init__(self, obs_dim: int, action_space, config):
        """
        Args:
            obs_dim: Observation space dimension.
            action_space: Gymnasium action space.
            config: Configuration object with PPO parameters.
        """
        super().__init__(resolve_device(config.device))

        self.clip_param = config.get("clip_param", 0.2)
        self.ppo_epoch = config.get("ppo_epoch", 10)
        self.num_mini_batch = config.get("num_mini_batch", 32)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.action_scaling = config.get("action_scaling", True)
        self.action_bound_method = config.get("action_bound_method", "clip")

        # New PPO features (aligned with tianshou)
        self.value_clip = config.get("value_clip", False)
        self.dual_clip = config.get("dual_clip", None)
        self.norm_adv = config.get("advantage_normalization", True)
        self.recompute_advantage = config.get("recompute_advantage", False)

        # GAE parameters (stored for recompute_advantage)
        self._gamma = config.gamma
        self._gae_lambda = config.gae_lambda

        if self.dual_clip is not None:
            assert self.dual_clip > 1.0, (
                f"Dual-clip PPO parameter should be > 1.0, got {self.dual_clip}"
            )

        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"PPO only supports Box action spaces, got {type(action_space)!r}")
        self.action_low = np.asarray(action_space.low, dtype=np.float32)
        self.action_high = np.asarray(action_space.high, dtype=np.float32)

        # Actor: default to gaussian_fixed_std (PPO-specific, state-independent log_std)
        self.actor = build_actor_model(
            config,
            obs_dim,
            action_space,
            default_name="gaussian_fixed_std",
            default_kwargs={"squash": False, "unbounded": False, "max_action": 1.0},
        ).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=config.lr)

        # Critic (Value network)
        self.value_net = build_value_model(
            config,
            obs_dim,
            default_name="value_network",
        ).to(self.device)
        self.value_optim = Adam(self.value_net.parameters(), lr=config.lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """
        Select action and return mapped env action plus rollout statistics.

        During rollout collection, the policy samples a raw action, stores that
        raw action and its log-probability for PPO updates, and separately maps
        the action into the environment action range before stepping the env.

        Returns:
            If evaluate: action numpy array
            Otherwise: (env_action, log_prob, value, raw_action) tuple
        """
        state_t, single = self._prepare_state_tensor(state)

        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state_t)
                mapped_action = self._map_action(action)
                return mapped_action[0] if single else mapped_action
            else:
                raw_action, log_prob, _ = self.actor.sample(state_t)
                value = self.value_net(state_t)
                mapped_action = self._map_action(raw_action)
                actions = mapped_action[0] if single else mapped_action
                raw_actions = self._format_action_output(raw_action, single)
                log_probs = self._format_scalar_output(log_prob, single)
                values = self._format_scalar_output(value, single)
                return actions, log_probs, values, raw_actions

    def get_value(self, state: np.ndarray) -> float:
        """Estimate V(s) for the given state."""
        state_t, single = self._prepare_state_tensor(state)
        with torch.no_grad():
            value = self.value_net(state_t)
        return self._format_scalar_output(value, single)

    def _map_action(self, action: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Map raw policy actions into the environment action space."""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        mapped = np.asarray(action, dtype=np.float32)
        if self.action_bound_method == "clip":
            mapped = np.clip(mapped, -1.0, 1.0)
        elif self.action_bound_method == "tanh":
            mapped = np.tanh(mapped)
        elif self.action_bound_method is not None:
            raise ValueError(
                f"Unsupported PPO action_bound_method: {self.action_bound_method!r}"
            )

        if self.action_scaling:
            mapped = self.action_low + (self.action_high - self.action_low) * (
                mapped + 1.0
            ) / 2.0

        return mapped.astype(np.float32)

    def update_parameters(self, rollout_buffer) -> Dict[str, float]:
        """
        Update PPO parameters from rollout buffer.

        Performs ppo_epoch optimization epochs, each splitting the rollout
        data into num_mini_batch mini-batches per epoch.

        Features:
          - Per-mini-batch advantage normalization
          - Value function clipping (if value_clip=True)
          - Dual-clip PPO (if dual_clip is set)
          - Recompute advantages each epoch (if recompute_advantage=True)

        Args:
            rollout_buffer: RolloutBuffer with computed advantages and returns.

        Returns:
            Dict with averaged losses over all epochs: policy_loss, value_loss, entropy
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.ppo_epoch):
            # Recompute advantages each epoch (arXiv:2006.05990)
            if self.recompute_advantage and epoch > 0:
                rollout_buffer.compute_returns_and_advantages(
                    gamma=self._gamma,
                    gae_lambda=self._gae_lambda,
                )

            for (
                states, actions, old_log_probs, advantages, returns, old_values
            ) in rollout_buffer.get_mini_batches(self.num_mini_batch, self.device):

                # Per-mini-batch advantage normalization (tianshou standard)
                if self.norm_adv:
                    adv_mean = advantages.mean()
                    adv_std = advantages.std()
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

                # Evaluate actions under current policy
                log_probs, entropy = self.actor.evaluate_actions(states, actions)

                # Importance sampling ratio
                ratio = torch.exp(log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                ) * advantages

                if self.dual_clip is not None:
                    # Dual-clip PPO (arXiv:1912.09729)
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    policy_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with optional clipping
                values = self.value_net(states)
                if self.value_clip:
                    # Value function clipping (arXiv:1811.02553v3)
                    v_clip = old_values + (values - old_values).clamp(
                        -self.clip_param, self.clip_param
                    )
                    vf1 = (returns - values).pow(2)
                    vf2 = (returns - v_clip).pow(2)
                    value_loss = torch.max(vf1, vf2).mean()
                else:
                    value_loss = F.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Update actor and critic
                self.actor_optim.zero_grad()
                self.value_optim.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    nn.utils.clip_grad_norm_(
                        self.value_net.parameters(), self.max_grad_norm
                    )

                self.actor_optim.step()
                self.value_optim.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()
                num_updates += 1

        num_updates = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "value_net": self.value_net.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "value_optim": self.value_optim.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any], evaluate: bool = False) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.value_net.load_state_dict(state_dict["value_net"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.value_optim.load_state_dict(state_dict["value_optim"])

        if evaluate:
            self.actor.eval()
            self.value_net.eval()
        else:
            self.actor.train()
            self.value_net.train()
