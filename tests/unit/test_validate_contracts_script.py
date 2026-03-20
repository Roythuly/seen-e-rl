import subprocess
import sys
import json
from pathlib import Path


def test_validate_contracts_script_passes_for_minimum_schemas():
    result = subprocess.run(
        [sys.executable, "scripts/validate_contracts.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "contracts validation passed" in result.stdout.lower()


def test_key_contracts_include_algorithm_minimum_fields():
    trajectory = json.loads(Path("contracts/v0/trajectory_batch.schema.json").read_text(encoding="utf-8"))
    replay_record = json.loads(Path("contracts/v0/replay_record.schema.json").read_text(encoding="utf-8"))
    replay_batch = json.loads(Path("contracts/v0/replay_batch.schema.json").read_text(encoding="utf-8"))

    assert {"log_probs", "value_estimates", "policy_version"}.issubset(trajectory["properties"])
    assert {"next_observations", "terminated", "truncated", "policy_version"}.issubset(replay_record["properties"])
    assert {"next_observations", "terminated", "truncated", "policy_version"}.issubset(replay_batch["properties"])


def test_runtime_schedule_contracts_exist():
    runtime = json.loads(Path("contracts/v0/runtime_spec.schema.json").read_text(encoding="utf-8"))
    assert {"collection_schedule", "update_schedule", "publish_schedule"}.issubset(runtime["properties"])
