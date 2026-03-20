import subprocess
import sys
from pathlib import Path


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
