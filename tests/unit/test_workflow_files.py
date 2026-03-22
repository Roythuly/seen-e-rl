from pathlib import Path


def test_ci_workflow_runs_real_validation_steps():
    content = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "actions/checkout" in content
    assert "actions/setup-python" in content
    assert "validate_docs.py" in content
    assert "validate_contracts.py" in content
    assert "run_tests.sh" in content
    assert "placeholder" not in content.lower()


def test_evals_workflow_runs_runtime_preflight_and_eval_script():
    content = Path(".github/workflows/evals.yml").read_text(encoding="utf-8")

    assert "actions/checkout" in content
    assert "actions/setup-python" in content
    assert "validate_runtime_env.py" in content
    assert "run_evals.sh" in content
    assert "placeholder" not in content.lower()
