from __future__ import annotations

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DOCS = [
    "README.md",
    "CONTRIBUTING.md",
    "docs/index.md",
    "docs/README.md",
    "docs/external/feishu_links.md",
    "docs/architecture/system-context.md",
    "docs/architecture/execution-model.md",
    "docs/architecture/contracts.md",
    "docs/architecture/trace-schema.md",
    "docs/architecture/eval-strategy.md",
    "docs/prd/v0.1/system-prd.md",
    "docs/rfcs/RFC-template.md",
    "docs/adrs/ADR-template.md",
    "docs/runbooks/Runbook-template.md",
    "docs/releases/v0.1/index.md",
    "docs/algorithms/overview.md",
    "docs/algorithms/ppo.md",
    "docs/algorithms/sac.md",
    "docs/algorithms/td3.md",
    "docs/algorithms/extension-guide.md",
    "configs/README.md",
    "contracts/README.md",
    "scripts/README.md",
]

MODULES = ["model", "sampler", "trainer", "info", "evaluator"]
MODULE_FILES = [
    "overview.md",
    "api.md",
    "contracts.md",
    "observability.md",
    "failure-modes.md",
    "appendix.md",
    "todos.md",
]
REQUIRED_CONFIGS = [
    "configs/algo/example.yaml",
    "configs/backend/example.yaml",
    "configs/buffer/example.yaml",
    "configs/env/example.yaml",
    "configs/eval/example.yaml",
    "configs/experiment/example.yaml",
    "configs/model/example.yaml",
    "configs/sampler/example.yaml",
    "configs/trainer/example.yaml",
]


def _check_file(path_str: str, failures: list[str]) -> None:
    path = ROOT / path_str
    if not path.exists():
        failures.append(f"missing file: {path_str}")
        return
    if path.is_file() and not path.read_text(encoding="utf-8").strip():
        failures.append(f"empty file: {path_str}")


def _require_terms(path_str: str, terms: list[str], failures: list[str]) -> None:
    content = (ROOT / path_str).read_text(encoding="utf-8")
    for term in terms:
        if term not in content:
            failures.append(f"missing term {term!r}: {path_str}")


def _check_no_stale_placeholder_todos(failures: list[str]) -> None:
    todo_paths = sorted((ROOT / "docs" / "modules").glob("*/*todos.md"))
    for path in todo_paths:
        content = path.read_text(encoding="utf-8")
        if "placeholder schema" in content:
            failures.append(f"stale placeholder todo: {path.relative_to(ROOT)}")


def _check_configs(failures: list[str]) -> None:
    experiment = yaml.safe_load((ROOT / "configs/experiment/example.yaml").read_text(encoding="utf-8"))
    expected_top_level = {"run_name", "seed", "backend", "env", "model", "algo", "sampler", "trainer", "buffer", "eval"}
    missing = expected_top_level.difference(experiment.keys())
    if missing:
        failures.append(f"missing experiment keys: {sorted(missing)}")

    trainer = yaml.safe_load((ROOT / "configs/trainer/example.yaml").read_text(encoding="utf-8"))
    runtime = trainer.get("runtime", {})
    if not isinstance(runtime, dict):
        failures.append("trainer runtime must be a mapping")
    else:
        for key in ("collection_schedule", "update_schedule", "publish_schedule"):
            if key not in runtime:
                failures.append(f"missing trainer runtime key: {key}")

    eval_config = yaml.safe_load((ROOT / "configs/eval/example.yaml").read_text(encoding="utf-8"))
    if eval_config.get("selector") not in {"latest", "best", "milestone"}:
        failures.append("eval selector must be one of latest/best/milestone")


def main() -> int:
    failures: list[str] = []

    for doc in REQUIRED_DOCS:
        _check_file(doc, failures)

    for module in MODULES:
        for name in MODULE_FILES:
            _check_file(f"docs/modules/{module}/{name}", failures)

    for config in REQUIRED_CONFIGS:
        _check_file(config, failures)

    _require_terms(
        "docs/architecture/execution-model.md",
        ["RuntimeSpec", "CollectionSchedule", "UpdateSchedule", "PublishSchedule", "PolicySnapshot", "CheckpointManifest"],
        failures,
    )
    _require_terms(
        "docs/modules/sampler/api.md",
        ["ActorHandle.act", "ActionOutput"],
        failures,
    )
    _require_terms(
        "docs/modules/trainer/api.md",
        ["Learner.update", "Model.forward_train", "CheckpointManifest"],
        failures,
    )
    _require_terms(
        "docs/modules/evaluator/api.md",
        ["latest", "best", "milestone"],
        failures,
    )
    _require_terms(
        "docs/algorithms/ppo.md",
        ["sampler 原生字段", "trainer / learner 派生字段", "TrajectoryBatch.log_probs", "TrajectoryBatch.value_estimates"],
        failures,
    )
    _require_terms(
        "docs/algorithms/sac.md",
        ["ReplayBatch.truncated", "entropy temperature", "CheckpointManifest"],
        failures,
    )
    _require_terms(
        "docs/algorithms/td3.md",
        ["policy_delay", "target policy smoothing", "PublishSchedule"],
        failures,
    )
    _check_no_stale_placeholder_todos(failures)
    _check_configs(failures)

    if failures:
        print("docs validation failed")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("docs validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
