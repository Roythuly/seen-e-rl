from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LearnerTemplate(ABC):
    @abstractmethod
    def update(self, batch: dict[str, Any], objective: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def publish_policy(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError


class RuntimeLoopTemplate(ABC):
    @abstractmethod
    def run(
        self,
        runtime_spec: dict[str, Any],
        actor_handle: Any | None = None,
        sampler: Any | None = None,
        learner: Any | None = None,
        info: Any | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class ReplayBufferTemplate(ABC):
    @abstractmethod
    def write(self, record: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, spec: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
