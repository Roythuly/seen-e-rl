from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CheckpointSelectorTemplate(ABC):
    @abstractmethod
    def select(self, selector: str, policy_version: int | None = None) -> dict[str, Any]:
        raise NotImplementedError


class EvaluatorTemplate(ABC):
    @abstractmethod
    def evaluate(self, checkpoint_manifest: dict[str, Any], seeds: list[int], env_spec: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
