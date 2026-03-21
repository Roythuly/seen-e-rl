from __future__ import annotations

import json
from pathlib import Path

from rl_training_infra.info import (
    ConsoleMetricSink,
    InfoHubBase,
    InfoHubTemplate,
    JSONLMetricSink,
    MetricEventBuilderBase,
    MetricSinkTemplate,
)


class RecordingSink(MetricSinkTemplate):
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def write(self, metric_event: dict[str, object]) -> None:
        self.events.append(metric_event)


class FailingSink(MetricSinkTemplate):
    def write(self, metric_event: dict[str, object]) -> None:
        raise RuntimeError("sink unavailable")


def test_info_package_exports_template_and_base_types() -> None:
    assert InfoHubTemplate.__name__ == "InfoHubTemplate"
    assert MetricSinkTemplate.__name__ == "MetricSinkTemplate"
    assert InfoHubBase.__name__ == "InfoHubBase"
    assert MetricEventBuilderBase.__name__ == "MetricEventBuilderBase"
    assert ConsoleMetricSink.__name__ == "ConsoleMetricSink"
    assert JSONLMetricSink.__name__ == "JSONLMetricSink"


def test_info_hub_records_training_checkpoint_and_evaluation_events() -> None:
    sink = RecordingSink()
    hub = InfoHubBase(
        builder=MetricEventBuilderBase(
            run_id="run-1",
            algorithm="ppo",
            backend="torch",
            env_id="CartPole-v1",
        ),
        sinks=[sink],
    )

    training_event = hub.record_training(
        event_type="update_finished",
        policy_version=7,
        env_steps=32,
        grad_steps=4,
        metrics={"loss": 1.25},
        status="ok",
    )
    checkpoint_event = hub.record_checkpoint(
        checkpoint_id="ckpt-001",
        path="/tmp/checkpoint",
        policy_version=7,
        status="saved",
        metrics={"size_bytes": 1024},
    )
    evaluation_event = hub.record_evaluation(
        checkpoint_id="ckpt-001",
        policy_version=7,
        selector="latest",
        aggregate={"reward_mean": 10.5},
        per_seed=[{"seed": 1, "reward": 10.0}],
        status="complete",
    )

    assert sink.events == [training_event, checkpoint_event, evaluation_event]
    assert training_event["event_type"] == "update_finished"
    assert training_event["event_category"] == "training"
    assert checkpoint_event["event_category"] == "checkpoint"
    assert evaluation_event["event_category"] == "evaluation"
    assert evaluation_event["checkpoint_id"] == "ckpt-001"
    assert evaluation_event["metrics"] == {}


def test_info_hub_ignores_sink_failures_and_keeps_dispatching() -> None:
    recording_sink = RecordingSink()
    hub = InfoHubBase(
        builder=MetricEventBuilderBase(
            run_id="run-1",
            algorithm="ppo",
            backend="torch",
            env_id="CartPole-v1",
        ),
        sinks=[FailingSink(), recording_sink],
    )

    event = hub.record_training(
        event_type="update_finished",
        policy_version=7,
        env_steps=32,
        grad_steps=4,
        metrics={"loss": 1.25},
        status="ok",
    )

    assert recording_sink.events == [event]
    assert event["status"] == "ok"


def test_console_metric_sink_writes_json_line_to_stdout(capsys) -> None:
    event = {
        "run_id": "run-1",
        "event_type": "checkpoint_saved",
        "event_category": "checkpoint",
        "algorithm": "ppo",
        "backend": "torch",
        "env_id": "CartPole-v1",
        "policy_version": 7,
        "checkpoint_id": "ckpt-001",
        "env_steps": 64,
        "grad_steps": 8,
        "timestamp": "2026-03-21T00:00:00+00:00",
        "status": "saved",
        "metrics": {"size_bytes": 1024},
    }

    sink = ConsoleMetricSink()
    sink.write(event)

    captured = capsys.readouterr()
    assert captured.out == json.dumps(event, sort_keys=True) + "\n"
    assert captured.err == ""


def test_jsonl_metric_sink_appends_json_lines(tmp_path: Path) -> None:
    event = {
        "run_id": "run-1",
        "event_type": "evaluation_finished",
        "event_category": "evaluation",
        "algorithm": "ppo",
        "backend": "torch",
        "env_id": "CartPole-v1",
        "policy_version": 7,
        "checkpoint_id": "ckpt-001",
        "env_steps": 64,
        "grad_steps": 8,
        "timestamp": "2026-03-21T00:00:00+00:00",
        "status": "complete",
        "metrics": {"reward_mean": 10.5},
    }

    sink = JSONLMetricSink(tmp_path / "info.jsonl")
    sink.write(event)

    payload = (tmp_path / "info.jsonl").read_text(encoding="utf-8")
    assert payload == json.dumps(event, sort_keys=True) + "\n"
    assert json.loads(payload.strip()) == event
