from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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


def _coerce_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return ensure_2d_float_tensor(value)
    if isinstance(value, Mapping):
        parts = [_coerce_tensor(item) for item in value.values()]
        if not parts:
            raise KeyError("mapping payload cannot be empty")
        return torch.cat(parts, dim=-1)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise KeyError("sequence payload cannot be empty")
        try:
            return ensure_2d_float_tensor(torch.as_tensor(value))
        except Exception:
            return torch.cat([_coerce_tensor(item) for item in value], dim=-1)
    return ensure_2d_float_tensor(torch.as_tensor(value))


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
        if key in batch:
            return _coerce_tensor(batch[key])

    raise KeyError(f"Expected tensor payload under one of {preferred_keys or ('<any tensor field>',)}")


class TorchModelTemplateBase(ModelTemplate, nn.Module):
    def __init__(self, *, policy_version: int = 0) -> None:
        super().__init__()
        self.register_buffer("_policy_version", torch.tensor(policy_version, dtype=torch.long))

    @property
    def policy_version(self) -> int:
        return int(self._policy_version.item())

    def set_policy_version(self, policy_version: int) -> None:
        self._policy_version.fill_(int(policy_version))

    def _observation_tensor(self, batch: dict[str, Any]) -> Tensor:
        return batch_tensor_from_dict(batch, "observations", "observation")

    def _next_observation_tensor(self, batch: dict[str, Any]) -> Tensor:
        return batch_tensor_from_dict(batch, "next_observations", "next_observation")

    def _action_tensor(self, batch: dict[str, Any]) -> Tensor:
        return batch_tensor_from_dict(batch, "actions", "action")

    @staticmethod
    def _deterministic_flag(policy_state: Any | None) -> bool:
        return bool(isinstance(policy_state, dict) and policy_state.get("deterministic"))

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "metadata": dict(metadata or {}),
            "policy_version": self.policy_version,
        }
        torch.save(payload, checkpoint_path)
        return {"path": str(checkpoint_path), "metadata": payload["metadata"], "policy_version": self.policy_version}

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location="cpu")
        state_dict = payload.get("state_dict", payload)
        self.load_state_dict(state_dict)
        if "policy_version" in payload:
            self.set_policy_version(int(payload["policy_version"]))
        return dict(payload.get("metadata", {}))
