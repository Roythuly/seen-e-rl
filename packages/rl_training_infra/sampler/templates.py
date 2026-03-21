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


class EnvFactoryTemplate(ABC):
    @abstractmethod
    def create(self, env_spec: dict[str, Any], seed: int | None = None) -> Any:
        raise NotImplementedError


class EnvAdapterTemplate(ABC):
    @abstractmethod
    def reset(self, env: Any, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def step(self, env: Any, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        raise NotImplementedError


class TrajectoryAssemblerTemplate(BatchAssemblerTemplate):
    pass


class ReplayAssemblerTemplate(ABC):
    @abstractmethod
    def build(self, record: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
