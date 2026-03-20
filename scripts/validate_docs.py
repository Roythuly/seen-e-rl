from __future__ import annotations

from pathlib import Path
import sys


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


def _check_file(path_str: str, failures: list[str]) -> None:
    path = ROOT / path_str
    if not path.exists():
        failures.append(f"missing file: {path_str}")
        return
    if path.is_file() and not path.read_text(encoding="utf-8").strip():
        failures.append(f"empty file: {path_str}")


def main() -> int:
    failures: list[str] = []

    for doc in REQUIRED_DOCS:
        _check_file(doc, failures)

    for module in MODULES:
        for name in MODULE_FILES:
            _check_file(f"docs/modules/{module}/{name}", failures)

    if failures:
        print("docs validation failed")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("docs validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
