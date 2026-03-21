from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetricSinkTemplate(ABC):
    @abstractmethod
    def write(self, metric_event: dict[str, Any]) -> None:
        raise NotImplementedError


class InfoHubTemplate(ABC):
    @abstractmethod
    def record(self, event: dict[str, Any]) -> None:
        raise NotImplementedError
