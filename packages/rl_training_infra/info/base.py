from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .templates import InfoHubTemplate, MetricSinkTemplate


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(slots=True)
class MetricEventBuilderBase:
    run_id: str
    algorithm: str
    backend: str
    env_id: str
    module: str = "info"

    def build(
        self,
        *,
        event_type: str,
        event_category: str,
        status: str,
        policy_version: int | None = None,
        checkpoint_id: str | None = None,
        env_steps: int = 0,
        grad_steps: int = 0,
        metrics: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
        error_code: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        event = {
            "run_id": self.run_id,
            "event_type": event_type,
            "event_category": event_category,
            "algorithm": self.algorithm,
            "backend": self.backend,
            "env_id": self.env_id,
            "policy_version": policy_version,
            "checkpoint_id": checkpoint_id,
            "env_steps": env_steps,
            "grad_steps": grad_steps,
            "timestamp": timestamp or _utc_timestamp(),
            "status": status,
            "metrics": _as_dict(metrics),
            "error_code": error_code,
            "module": self.module,
        }
        event.update(extra)
        return event

    def coerce(self, event: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(event)
        normalized.setdefault("run_id", self.run_id)
        normalized.setdefault("algorithm", self.algorithm)
        normalized.setdefault("backend", self.backend)
        normalized.setdefault("env_id", self.env_id)
        normalized.setdefault("policy_version", None)
        normalized.setdefault("checkpoint_id", None)
        normalized.setdefault("env_steps", 0)
        normalized.setdefault("grad_steps", 0)
        normalized.setdefault("timestamp", _utc_timestamp())
        normalized.setdefault("status", "ok")
        normalized.setdefault("metrics", {})
        normalized.setdefault("error_code", None)
        normalized.setdefault("module", self.module)
        return normalized

    def training_event(
        self,
        *,
        event_type: str = "update_finished",
        status: str = "ok",
        policy_version: int | None = None,
        env_steps: int = 0,
        grad_steps: int = 0,
        metrics: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        return self.build(
            event_type=event_type,
            event_category="training",
            status=status,
            policy_version=policy_version,
            env_steps=env_steps,
            grad_steps=grad_steps,
            metrics=metrics,
            timestamp=timestamp,
            **extra,
        )

    def checkpoint_event(
        self,
        *,
        checkpoint_id: str,
        path: str,
        policy_version: int,
        event_type: str = "checkpoint_saved",
        status: str = "saved",
        env_steps: int = 0,
        grad_steps: int = 0,
        metrics: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        return self.build(
            event_type=event_type,
            event_category="checkpoint",
            status=status,
            policy_version=policy_version,
            checkpoint_id=checkpoint_id,
            env_steps=env_steps,
            grad_steps=grad_steps,
            metrics=metrics,
            timestamp=timestamp,
            path=path,
            **extra,
        )

    def evaluation_event(
        self,
        *,
        checkpoint_id: str,
        policy_version: int,
        selector: str,
        aggregate: Mapping[str, Any],
        per_seed: Sequence[Mapping[str, Any]],
        event_type: str = "evaluation_finished",
        status: str = "complete",
        env_steps: int = 0,
        grad_steps: int = 0,
        metrics: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        return self.build(
            event_type=event_type,
            event_category="evaluation",
            status=status,
            policy_version=policy_version,
            checkpoint_id=checkpoint_id,
            env_steps=env_steps,
            grad_steps=grad_steps,
            metrics=metrics,
            timestamp=timestamp,
            selector=selector,
            aggregate=_as_dict(aggregate),
            per_seed=[dict(item) for item in per_seed],
            **extra,
        )


@dataclass(slots=True)
class InfoHubBase(InfoHubTemplate):
    builder: MetricEventBuilderBase
    sinks: Sequence[MetricSinkTemplate] = field(default_factory=tuple)

    def record(self, event: Mapping[str, Any]) -> dict[str, Any]:
        normalized = self.builder.coerce(event)
        for sink in self.sinks:
            sink.write(normalized)
        return normalized

    def record_training(self, **event: Any) -> dict[str, Any]:
        return self.record(self.builder.training_event(**event))

    def record_checkpoint(self, **event: Any) -> dict[str, Any]:
        return self.record(self.builder.checkpoint_event(**event))

    def record_evaluation(self, **event: Any) -> dict[str, Any]:
        return self.record(self.builder.evaluation_event(**event))
