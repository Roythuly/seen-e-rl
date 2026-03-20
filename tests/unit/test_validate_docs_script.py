import subprocess
import sys
from pathlib import Path
import yaml


def test_validate_docs_script_passes_for_repository_layout():
    result = subprocess.run(
        [sys.executable, "scripts/validate_docs.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "docs validation passed" in result.stdout.lower()


def test_execution_model_doc_makes_runtime_loop_role_explicit():
    content = Path("docs/architecture/execution-model.md").read_text(encoding="utf-8")

    assert "RuntimeLoop" in content
    assert "on-policy" in content
    assert "off-policy" in content
    assert "异步" in content
    assert "ActorHandle" in content
    assert "ReplayBuffer" in content
    assert "RuntimeSpec" in content
    assert "PublishSchedule" in content
    assert "CheckpointManifest" in content


def test_algorithm_docs_call_out_raw_and_derived_fields():
    ppo = Path("docs/algorithms/ppo.md").read_text(encoding="utf-8")
    sac = Path("docs/algorithms/sac.md").read_text(encoding="utf-8")
    td3 = Path("docs/algorithms/td3.md").read_text(encoding="utf-8")

    assert "sampler 原生字段" in ppo
    assert "trainer / learner 派生字段" in ppo
    assert "entropy temperature" in sac
    assert "policy_delay" in td3


def test_experiment_config_is_top_level_assembly_entry():
    payload = yaml.safe_load(Path("configs/experiment/example.yaml").read_text(encoding="utf-8"))
    assert {"run_name", "seed", "backend", "env", "model", "algo", "sampler", "trainer", "buffer", "eval"}.issubset(payload)
