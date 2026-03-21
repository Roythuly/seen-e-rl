from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .templates import ModelTemplate


def ensure_2d_float_tensor(value: Tensor) -> Tensor:
    tensor = value.float()
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim > 2:
        return tensor.flatten(start_dim=1)
    return tensor


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int] | None = None,
    *,
    output_dim: int | None = None,
    activation_cls: type[nn.Module] = nn.ReLU,
    final_activation_cls: type[nn.Module] | None = None,
) -> nn.Sequential:
    hidden_layers = tuple(hidden_sizes or ())
    layers: list[nn.Module] = []
    current_dim = input_dim

    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation_cls())
        current_dim = hidden_dim

    if output_dim is not None:
        layers.append(nn.Linear(current_dim, output_dim))
        if final_activation_cls is not None:
            layers.append(final_activation_cls())
    elif not layers:
        layers.append(nn.Identity())

    return nn.Sequential(*layers)


def batch_tensor_from_dict(batch: dict[str, Any], *preferred_keys: str) -> Tensor:
    for key in preferred_keys:
        value = batch.get(key)
        if isinstance(value, Tensor):
            return ensure_2d_float_tensor(value)

    raise KeyError(f"Expected tensor payload under one of {preferred_keys or ('<any tensor field>',)}")


class TorchModelTemplateBase(ModelTemplate, nn.Module):
    def __init__(self, *, policy_version: int = 0) -> None:
        super().__init__()
        self.register_buffer("_policy_version", torch.tensor(policy_version, dtype=torch.long))

    @property
    def policy_version(self) -> int:
        return int(self._policy_version.item())

    def _observation_tensor(self, batch: dict[str, Any]) -> Tensor:
        return batch_tensor_from_dict(batch, "observations", "observation")

    def _next_observation_tensor(self, batch: dict[str, Any]) -> Tensor:
        return batch_tensor_from_dict(batch, "next_observations", "next_observation")

    def _action_tensor(self, batch: dict[str, Any]) -> Tensor:
        return batch_tensor_from_dict(batch, "actions", "action")

    @staticmethod
    def _deterministic_flag(policy_state: Any | None) -> bool:
        return bool(isinstance(policy_state, dict) and policy_state.get("deterministic"))
