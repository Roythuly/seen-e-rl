from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from typing import Any

import gymnasium as gym
import torch

from rl_training_infra.common import ensure_headless_mujoco_backend


def _detach_output(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 0:
            return float(tensor.item())
        return tensor.numpy()
    if isinstance(value, Mapping):
        return {key: _detach_output(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class GymEnvFactory:
    def create(self, env_spec: dict[str, Any], seed: int | None = None) -> Any:
        ensure_headless_mujoco_backend()
        env = gym.make(env_spec["id"], **dict(env_spec.get("kwargs", {})))
        if seed is not None:
            env.reset(seed=seed)
        return env


@dataclass(slots=True)
class TorchActorHandle:
    model: Any
    deterministic: bool = False

    def act(self, observation_batch: dict[str, Any], policy_version: int | None = None) -> dict[str, Any]:
        del policy_version
        with torch.no_grad():
            outputs = self.model.forward_act(observation_batch, policy_state={"deterministic": self.deterministic})
        normalized = dict(outputs)
        normalized["action"] = _detach_output(outputs["action"])
        for key in ("log_prob", "value_estimate", "diagnostics"):
            if key in outputs:
                normalized[key] = _detach_output(outputs[key])
        return normalized
