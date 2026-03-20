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
    "runtime_spec.schema.json",
    "collection_schedule.schema.json",
    "update_schedule.schema.json",
    "publish_schedule.schema.json",
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
MINIMUM_PROPERTIES = {
    "backend_spec.schema.json": ["name", "runtime_mode"],
    "observation_spec.schema.json": ["kind"],
    "observation_batch.schema.json": ["data"],
    "model_spec.schema.json": ["encoder", "actor_head", "critic_head", "training_interface"],
    "env_spec.schema.json": ["id"],
    "runtime_spec.schema.json": ["collection_schedule", "update_schedule", "publish_schedule"],
    "collection_schedule.schema.json": ["mode", "unit", "amount", "freeze_policy_during_collection"],
    "update_schedule.schema.json": ["trigger_unit", "updates_per_trigger"],
    "publish_schedule.schema.json": ["strategy"],
    "trajectory_batch.schema.json": [
        "observations",
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "log_probs",
        "value_estimates",
        "policy_version",
    ],
    "replay_record.schema.json": [
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminated",
        "truncated",
        "policy_version",
    ],
    "replay_batch.schema.json": [
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminated",
        "truncated",
        "policy_version",
    ],
    "replay_buffer_spec.schema.json": ["capacity", "batch_size", "min_ready_size", "sampling_mode"],
    "policy_snapshot.schema.json": ["run_id", "policy_version", "actor_ref", "published_at", "backend", "algorithm", "checkpoint_id"],
    "update_result.schema.json": ["run_id", "policy_version", "env_steps", "grad_steps", "status", "published_policy", "metrics"],
    "metric_event.schema.json": [
        "run_id",
        "event_type",
        "event_category",
        "algorithm",
        "backend",
        "env_id",
        "policy_version",
        "checkpoint_id",
        "env_steps",
        "grad_steps",
        "timestamp",
        "status",
    ],
    "eval_report.schema.json": ["run_id", "checkpoint_id", "policy_version", "selector", "aggregate", "per_seed", "algorithm", "backend", "env_id", "status"],
    "checkpoint_manifest.schema.json": ["checkpoint_id", "run_id", "policy_version", "path", "components", "backend", "algorithm", "created_at"],
}


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

        if payload.get("x-status") != "minimum":
            failures.append(f"unexpected x-status in {schema_name}")

        if schema_name in MINIMUM_PROPERTIES:
            required = set(payload.get("required", []))
            properties = set(payload.get("properties", {}).keys())
            for property_name in MINIMUM_PROPERTIES[schema_name]:
                if property_name not in required:
                    failures.append(f"missing required property {property_name}: {schema_name}")
                if property_name not in properties:
                    failures.append(f"missing property definition {property_name}: {schema_name}")

    if failures:
        print("contracts validation failed")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("contracts validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
