from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from rl_training_infra.contracts import build_eval_report

from .templates import CheckpointSelectorTemplate, EvaluatorTemplate, EvalReportWriterTemplate


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(value))


def _selector_name(checkpoint_manifest: Mapping[str, Any], fallback: str = "latest") -> str:
    selector = checkpoint_manifest.get("selector")
    if isinstance(selector, str) and selector:
        return selector

    selection = checkpoint_manifest.get("selection")
    if isinstance(selection, Mapping):
        nested_selector = selection.get("selector")
        if isinstance(nested_selector, str) and nested_selector:
            return nested_selector

    return fallback


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def summarize_seed_results(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, list[float]] = {}
    for entry in per_seed:
        for key, value in entry.items():
            if key == "seed":
                continue
            numeric_value = _numeric_value(value)
            if numeric_value is None:
                continue
            summary.setdefault(key, []).append(numeric_value)

    aggregate = {key: sum(values) / len(values) for key, values in summary.items()}
    aggregate["seed_count"] = len(per_seed)
    return aggregate


def _as_checkpoint_sequence(checkpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(checkpoint) for checkpoint in checkpoints]


def _latest_checkpoint_key(checkpoint: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        checkpoint.get("policy_version", -1),
        checkpoint.get("created_at", ""),
        checkpoint.get("checkpoint_id", ""),
    )


def _best_metric_value(checkpoint: Mapping[str, Any], metric_keys: Sequence[str]) -> float:
    for key in metric_keys:
        metric_value = _numeric_value(checkpoint.get(key))
        if metric_value is not None:
            return metric_value
        metrics = checkpoint.get("metrics")
        if isinstance(metrics, Mapping):
            nested_value = _numeric_value(metrics.get(key))
            if nested_value is not None:
                return nested_value
    fallback = _numeric_value(checkpoint.get("policy_version"))
    return fallback if fallback is not None else float("-inf")


@dataclass(slots=True)
class CheckpointSelectorBase(CheckpointSelectorTemplate):
    checkpoints: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    best_metric_keys: Sequence[str] = ("selection_score", "score", "reward_mean")

    def select(self, selector: str, policy_version: int | None = None) -> dict[str, Any]:
        if selector == "latest":
            selected = max(self._checkpoint_records(), key=_latest_checkpoint_key, default=None)
        elif selector == "best":
            selected = max(
                self._checkpoint_records(),
                key=lambda checkpoint: (
                    _best_metric_value(checkpoint, self.best_metric_keys),
                    checkpoint.get("policy_version", -1),
                    checkpoint.get("created_at", ""),
                    checkpoint.get("checkpoint_id", ""),
                ),
                default=None,
            )
        elif selector == "milestone":
            if policy_version is None:
                raise ValueError("policy_version is required for milestone selection")
            matches = [checkpoint for checkpoint in self._checkpoint_records() if checkpoint.get("policy_version") == policy_version]
            selected = max(matches, key=_latest_checkpoint_key, default=None)
        else:
            raise ValueError(f"unsupported selector: {selector}")

        if selected is None:
            raise LookupError(f"no checkpoint available for selector={selector!r}")

        selected_checkpoint = _copy_mapping(selected)
        selected_checkpoint["selector"] = selector
        if policy_version is not None:
            selected_checkpoint.setdefault("selection_policy_version", policy_version)
        return selected_checkpoint

    def _checkpoint_records(self) -> list[dict[str, Any]]:
        return _as_checkpoint_sequence(self.checkpoints)


ReportSeedRunner = Callable[[Mapping[str, Any], int, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(slots=True)
class EvalReportWriterBase(EvalReportWriterTemplate):
    sink: Callable[[dict[str, Any]], Any] | None = None

    def write(self, report: dict[str, Any]) -> dict[str, Any]:
        normalized = _copy_mapping(report)
        if self.sink is not None:
            self.sink(normalized)
        return normalized


@dataclass(slots=True)
class EvaluatorBase(EvaluatorTemplate):
    seed_runner: ReportSeedRunner
    report_writer: EvalReportWriterTemplate | None = None
    selector_name: str | None = None

    def evaluate(self, checkpoint_manifest: dict[str, Any], seeds: list[int], env_spec: dict[str, Any]) -> dict[str, Any]:
        if not seeds:
            raise ValueError("seeds cannot be empty")

        checkpoint = _copy_mapping(checkpoint_manifest)
        environment = _copy_mapping(env_spec)
        per_seed: list[dict[str, Any]] = []

        for seed in seeds:
            seed_result = dict(self.seed_runner(checkpoint, seed, environment))
            seed_result.setdefault("seed", seed)
            per_seed.append(seed_result)

        report = build_eval_report(
            run_id=checkpoint["run_id"],
            checkpoint_id=checkpoint["checkpoint_id"],
            policy_version=checkpoint["policy_version"],
            selector=_selector_name(checkpoint, self.selector_name or "latest"),
            aggregate=summarize_seed_results(per_seed),
            per_seed=[dict(item) for item in per_seed],
            algorithm=checkpoint["algorithm"],
            backend=checkpoint["backend"],
            env_id=environment["id"],
        )

        if self.report_writer is not None:
            return self.report_writer.write(report)
        return report
