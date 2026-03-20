from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "contracts" / "v0"
REQUIRED_SCHEMAS = [
    "backend_spec.schema.json",
    "observation_spec.schema.json",
    "observation_batch.schema.json",
    "model_spec.schema.json",
    "env_spec.schema.json",
    "trajectory_batch.schema.json",
    "replay_record.schema.json",
    "replay_batch.schema.json",
    "replay_buffer_spec.schema.json",
    "policy_snapshot.schema.json",
    "update_result.schema.json",
    "metric_event.schema.json",
    "eval_report.schema.json",
    "checkpoint_manifest.schema.json",
    "error_code.schema.json",
]
REQUIRED_FIELDS = ["$schema", "$id", "title", "description", "type", "x-status"]


def main() -> int:
    failures: list[str] = []

    for schema_name in REQUIRED_SCHEMAS:
        path = SCHEMA_DIR / schema_name
        if not path.exists():
            failures.append(f"missing schema: {schema_name}")
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid json: {schema_name}: {exc}")
            continue

        for field in REQUIRED_FIELDS:
            if field not in payload:
                failures.append(f"missing field {field}: {schema_name}")

        if payload.get("x-status") != "placeholder":
            failures.append(f"unexpected x-status in {schema_name}")

    if failures:
        print("contracts validation failed")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("contracts validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
