from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from ..base import build_mlp, ensure_2d_float_tensor


class GaussianActor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (),
        activation_cls: type[nn.Module] = nn.ReLU,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.backbone = build_mlp(input_dim, self.hidden_sizes, activation_cls=activation_cls)
        feature_dim = self.hidden_sizes[-1] if self.hidden_sizes else input_dim
        self.mean_head = nn.Linear(feature_dim, action_dim)
        self.log_std_head = nn.Linear(feature_dim, action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def distribution_params(self, features: Tensor) -> dict[str, Tensor]:
        encoded_features = self.backbone(ensure_2d_float_tensor(features))
        mean = self.mean_head(encoded_features)
        log_std = self.log_std_head(encoded_features).clamp(self.min_log_std, self.max_log_std)
        return {"mean": mean, "log_std": log_std}

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return self.distribution_params(features)

    def sample(self, features: Tensor, *, deterministic: bool = False) -> dict[str, Any]:
        distribution_params = self.distribution_params(features)
        distribution = Normal(distribution_params["mean"], distribution_params["log_std"].exp())
        action = distribution_params["mean"] if deterministic else distribution.rsample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return {
            "action": action,
            "log_prob": log_prob,
            "distribution_params": distribution_params,
        }


class DeterministicActor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (),
        activation_cls: type[nn.Module] = nn.ReLU,
        squash_output: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.backbone = build_mlp(input_dim, self.hidden_sizes, activation_cls=activation_cls)
        feature_dim = self.hidden_sizes[-1] if self.hidden_sizes else input_dim
        self.action_head = nn.Linear(feature_dim, action_dim)
        self.squash_output = squash_output

    def forward(self, features: Tensor) -> Tensor:
        encoded_features = self.backbone(ensure_2d_float_tensor(features))
        actions = self.action_head(encoded_features)
        if self.squash_output:
            return torch.tanh(actions)
        return actions
