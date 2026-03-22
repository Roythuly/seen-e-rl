from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Normal

from rl_training_infra.contracts import build_checkpoint_manifest, build_policy_snapshot, build_update_result

from .templates import LearnerTemplate


def _batch_length(batch: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            return int(value.shape[0]) if value.ndim > 0 else 1
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            nested = _batch_length(value, *tuple(value))
            if nested:
                return nested
    return 0


def _soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def as_float_tensor(value: Any) -> torch.Tensor:
    try:
        import numpy as np

        if isinstance(value, list):
            value = np.asarray(value)
    except Exception:
        pass
    return torch.as_tensor(value, dtype=torch.float32)


def as_bool_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.bool)


def distribution_log_prob(
    distribution_params: dict[str, torch.Tensor], actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    distribution = Normal(distribution_params["mean"], distribution_params["log_std"].exp())
    log_prob = distribution.log_prob(actions).sum(dim=-1)
    entropy = distribution.entropy().sum(dim=-1)
    return log_prob, entropy


@dataclass(slots=True)
class TorchLearnerBase(LearnerTemplate, ABC):
    model: Any
    run_id: str
    backend: str
    algorithm: str
    artifacts_dir: str | Path
    config: dict[str, Any] = field(default_factory=dict)
    env_steps: int = 0
    grad_steps: int = 0
    _checkpoint_index: int = 0
    _latest_checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir)
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def policy_dir(self) -> Path:
        return self.artifacts_dir / "policies"

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    def publish_policy(self) -> dict[str, Any]:
        policy_path = self.policy_dir / f"{self.algorithm}-policy-v{self.model.policy_version}.pt"
        self.model.save_checkpoint(policy_path, {"artifact": "policy_snapshot", "algorithm": self.algorithm})
        return build_policy_snapshot(
            run_id=self.run_id,
            policy_version=self.model.policy_version,
            actor_ref=str(policy_path),
            backend=self.backend,
            algorithm=self.algorithm,
            checkpoint_id=self._latest_checkpoint_id,
        )

    def save_checkpoint(self) -> dict[str, Any]:
        self._checkpoint_index += 1
        checkpoint_id = f"{self.algorithm}-ckpt-{self._checkpoint_index:04d}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state": self.optimizer_state(),
            "runtime_state": {
                "env_steps": self.env_steps,
                "grad_steps": self.grad_steps,
                "policy_version": self.model.policy_version,
            },
            "config": dict(self.config),
            "algorithm": self.algorithm,
            "backend": self.backend,
        }
        torch.save(payload, checkpoint_path)
        self._latest_checkpoint_id = checkpoint_id
        return build_checkpoint_manifest(
            checkpoint_id=checkpoint_id,
            run_id=self.run_id,
            policy_version=self.model.policy_version,
            path=str(checkpoint_path),
            components=self.checkpoint_components(),
            backend=self.backend,
            algorithm=self.algorithm,
        )

    @abstractmethod
    def optimizer_state(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def checkpoint_components(self) -> list[str]:
        raise NotImplementedError

    def _resolved_env_steps(self, batch: dict[str, Any], objective: dict[str, Any] | None = None) -> int:
        if objective is not None and "env_steps" in objective:
            return int(objective["env_steps"])
        return self.env_steps + _batch_length(batch, "observations", "rewards", "actions")

    def _advance_policy_version(self) -> int:
        next_policy_version = self.model.policy_version + 1
        self.model.set_policy_version(next_policy_version)
        return next_policy_version

    def _build_update_result(self, *, status: str, published_policy: bool, metrics: dict[str, Any]) -> dict[str, Any]:
        return build_update_result(
            run_id=self.run_id,
            policy_version=self.model.policy_version,
            checkpoint_id=self._latest_checkpoint_id,
            env_steps=self.env_steps,
            grad_steps=self.grad_steps,
            status=status,
            published_policy=published_policy,
            metrics=metrics,
        )
