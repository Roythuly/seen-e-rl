from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ..base import build_mlp, ensure_2d_float_tensor


class ValueHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_sizes: Sequence[int] = (),
        activation_cls: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.network = build_mlp(
            input_dim,
            tuple(hidden_sizes),
            output_dim=1,
            activation_cls=activation_cls,
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(ensure_2d_float_tensor(features)).squeeze(-1)


class TwinQCritic(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (),
        activation_cls: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        critic_input_dim = input_dim + action_dim
        critic_hidden_sizes = tuple(hidden_sizes)
        self.q1_network = build_mlp(
            critic_input_dim,
            critic_hidden_sizes,
            output_dim=1,
            activation_cls=activation_cls,
        )
        self.q2_network = build_mlp(
            critic_input_dim,
            critic_hidden_sizes,
            output_dim=1,
            activation_cls=activation_cls,
        )

    def forward(self, features: Tensor, actions: Tensor) -> dict[str, Tensor]:
        critic_input = torch.cat(
            [ensure_2d_float_tensor(features), ensure_2d_float_tensor(actions)],
            dim=-1,
        )
        return {
            "q1": self.q1_network(critic_input).squeeze(-1),
            "q2": self.q2_network(critic_input).squeeze(-1),
        }
