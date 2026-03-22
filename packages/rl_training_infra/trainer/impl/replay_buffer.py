from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import random
from typing import Any

import numpy as np
import torch

from ..templates import ReplayBufferTemplate


def _stack_values(values: list[Any]) -> Any:
    if not values:
        raise ValueError("cannot stack empty values")

    first = values[0]
    if isinstance(first, Mapping):
        return {key: _stack_values([value[key] for value in values]) for key in first}
    if isinstance(first, torch.Tensor):
        return torch.stack([torch.as_tensor(value) for value in values])
    if isinstance(first, np.ndarray):
        return torch.as_tensor(np.asarray(values))
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
        return torch.as_tensor(np.asarray(values))
    if isinstance(first, (bool, int, float)):
        return torch.as_tensor(values)
    return deepcopy(values)


@dataclass(slots=True)
class ReplayBuffer(ReplayBufferTemplate):
    capacity: int
    batch_size: int = 256
    sampling_mode: str = "random"
    seed: int | None = None
    _records: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self._records)

    def write(self, record: dict[str, Any]) -> None:
        if len(self._records) >= self.capacity:
            self._records.pop(0)
        self._records.append(deepcopy(record))

    def sample(self, spec: dict[str, Any]) -> dict[str, Any]:
        batch_size = int(spec.get("batch_size", self.batch_size))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._records) < batch_size:
            raise LookupError("replay buffer does not have enough samples")

        if self.sampling_mode == "fifo":
            records = self._records[-batch_size:]
        elif self.sampling_mode == "random":
            records = self._rng.sample(self._records, batch_size)
        else:
            raise ValueError(f"unsupported sampling_mode: {self.sampling_mode}")

        batch = {
            "observations": _stack_values([record["observations"] for record in records]),
            "actions": _stack_values([record["actions"] for record in records]),
            "rewards": _stack_values([record["rewards"] for record in records]),
            "next_observations": _stack_values([record["next_observations"] for record in records]),
            "terminated": _stack_values([record["terminated"] for record in records]),
            "truncated": _stack_values([record["truncated"] for record in records]),
            "policy_version": _stack_values([record["policy_version"] for record in records]),
            "sample_info": {
                "size": batch_size,
                "capacity": self.capacity,
                "available": len(self._records),
            },
        }
        if "env_step" in records[0]:
            batch["sample_info"]["max_env_step"] = max(record.get("env_step", 0) for record in records)
        return batch
