from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Registry:
    """Small named registry used by algorithm and module assemblies."""

    values: dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, value: Any) -> Any:
        self.values[name] = value
        return value

    def get(self, name: str) -> Any:
        return self.values[name]
