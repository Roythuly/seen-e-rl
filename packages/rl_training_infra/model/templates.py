from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelTemplate(ABC):
    """Base template for model implementations used by algorithm assemblies."""

    @abstractmethod
    def forward_act(self, observation_batch: dict[str, Any], policy_state: Any | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward_train(self, train_request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
