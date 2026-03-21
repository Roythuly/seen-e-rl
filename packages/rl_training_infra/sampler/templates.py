from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ActorHandleTemplate(ABC):
    @abstractmethod
    def act(self, observation_batch: dict[str, Any], policy_version: int | None = None) -> dict[str, Any]:
        raise NotImplementedError


class RolloutWorkerTemplate(ABC):
    @abstractmethod
    def collect(self, amount: int) -> dict[str, Any]:
        raise NotImplementedError


class BatchAssemblerTemplate(ABC):
    @abstractmethod
    def build(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError
