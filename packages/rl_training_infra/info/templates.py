from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetricSinkTemplate(ABC):
    @abstractmethod
    def write(self, metric_event: dict[str, Any]) -> None:
        raise NotImplementedError


class HealthSinkTemplate(MetricSinkTemplate):
    @abstractmethod
    def write(self, health_event: dict[str, Any]) -> None:
        raise NotImplementedError


class CheckpointSinkTemplate(MetricSinkTemplate):
    @abstractmethod
    def write(self, checkpoint_event: dict[str, Any]) -> None:
        raise NotImplementedError


class InfoHubTemplate(ABC):
    @abstractmethod
    def record(self, event: dict[str, Any]) -> None:
        raise NotImplementedError
