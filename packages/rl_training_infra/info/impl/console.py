from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from ..templates import MetricSinkTemplate


@dataclass(slots=True)
class ConsoleMetricSink(MetricSinkTemplate):
    stream: TextIO | None = None

    def write(self, metric_event: dict[str, Any]) -> None:
        stream = self.stream or sys.stdout
        stream.write(json.dumps(metric_event, sort_keys=True))
        stream.write("\n")
        flush = getattr(stream, "flush", None)
        if callable(flush):
            flush()
