from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..base import TorchModelTemplateBase


class TorchPPOModel(TorchModelTemplateBase):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        actor: nn.Module,
        value_head: nn.Module,
        policy_version: int = 0,
    ) -> None:
        super().__init__(policy_version=policy_version)
        self.encoder = encoder
        self.actor = actor
        self.value_head = value_head

    def forward_act(self, observation_batch: dict[str, Any], policy_state: Any | None = None) -> dict[str, Any]:
        features = self.encoder(self._observation_tensor(observation_batch))
        policy_outputs = self.actor.sample(features, deterministic=self._deterministic_flag(policy_state))
        value_estimate = self.value_head(features)
        return {
            "action": policy_outputs["action"],
            "policy_version": self.policy_version,
            "log_prob": policy_outputs["log_prob"],
            "value_estimate": value_estimate,
            "diagnostics": policy_outputs["distribution_params"],
        }

    def forward_train(self, train_request: dict[str, Any]) -> dict[str, Any]:
        features = self.encoder(self._observation_tensor(train_request))
        distribution_params = self.actor(features)
        state_values = self.value_head(features)
        return {
            "policy": {"distribution_params": distribution_params},
            "value": {"state_values": state_values},
            "q": {},
            "target": {},
            "aux": {"diagnostics": {"feature_shape": tuple(features.shape)}},
        }


class TorchSACModel(TorchModelTemplateBase):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        target_encoder: nn.Module,
        target_critic: nn.Module,
        alpha: float = 0.2,
        policy_version: int = 0,
    ) -> None:
        super().__init__(policy_version=policy_version)
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.target_encoder = target_encoder
        self.target_critic = target_critic
        self.register_buffer("_alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward_act(self, observation_batch: dict[str, Any], policy_state: Any | None = None) -> dict[str, Any]:
        features = self.encoder(self._observation_tensor(observation_batch))
        policy_outputs = self.actor.sample(features, deterministic=self._deterministic_flag(policy_state))
        return {
            "action": policy_outputs["action"],
            "policy_version": self.policy_version,
            "log_prob": policy_outputs["log_prob"],
            "diagnostics": policy_outputs["distribution_params"],
        }

    def forward_train(self, train_request: dict[str, Any]) -> dict[str, Any]:
        features = self.encoder(self._observation_tensor(train_request))
        next_features = self.target_encoder(self._next_observation_tensor(train_request))
        distribution_params = self.actor(features)
        target_policy = self.actor.sample(next_features)
        online_q = self.critic(features, self._action_tensor(train_request))
        target_q = self.target_critic(next_features, target_policy["action"])
        return {
            "policy": {"distribution_params": distribution_params},
            "value": {},
            "q": {"online": online_q, "target": target_q},
            "target": {},
            "aux": {"alpha": self._alpha.clone()},
        }


class TorchTD3Model(TorchModelTemplateBase):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        target_encoder: nn.Module,
        target_actor: nn.Module,
        target_critic: nn.Module,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_version: int = 0,
    ) -> None:
        super().__init__(policy_version=policy_version)
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.target_encoder = target_encoder
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.register_buffer("_target_policy_noise", torch.tensor(target_policy_noise, dtype=torch.float32))
        self.register_buffer("_target_noise_clip", torch.tensor(target_noise_clip, dtype=torch.float32))

    def forward_act(self, observation_batch: dict[str, Any], policy_state: Any | None = None) -> dict[str, Any]:
        features = self.encoder(self._observation_tensor(observation_batch))
        return {
            "action": self.actor(features),
            "policy_version": self.policy_version,
            "diagnostics": {"deterministic": True},
        }

    def forward_train(self, train_request: dict[str, Any]) -> dict[str, Any]:
        features = self.encoder(self._observation_tensor(train_request))
        next_features = self.target_encoder(self._next_observation_tensor(train_request))
        policy_actions = self.actor(features)
        online_q = self.critic(features, self._action_tensor(train_request))
        target_actions = self.target_actor(next_features)
        target_noise = torch.randn_like(target_actions) * self._target_policy_noise
        clipped_noise = target_noise.clamp(-self._target_noise_clip, self._target_noise_clip)
        smoothed_actions = (target_actions + clipped_noise).clamp(-1.0, 1.0)
        target_q = self.target_critic(next_features, smoothed_actions)
        return {
            "policy": {"actions": policy_actions},
            "value": {},
            "q": {"online": online_q, "target": target_q},
            "target": {},
            "aux": {
                "target_policy_smoothing": {
                    "noise": clipped_noise,
                    "clip": self._target_noise_clip.clone(),
                    "smoothed_actions": smoothed_actions,
                }
            },
        }
