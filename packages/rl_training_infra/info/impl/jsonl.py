from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..templates import MetricSinkTemplate


@dataclass(slots=True)
class JSONLMetricSink(MetricSinkTemplate):
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, metric_event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric_event, sort_keys=True))
            handle.write("\n")
