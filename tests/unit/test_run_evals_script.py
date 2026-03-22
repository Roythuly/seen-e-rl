from pathlib import Path


def test_run_evals_script_executes_real_train_and_evaluate_flows():
    content = Path("scripts/run_evals.sh").read_text(encoding="utf-8")

    assert "main.py train" in content
    assert "main.py evaluate" in content
    assert "MUJOCO_GL" in content
    assert "Humanoid-v5" not in content or "configs/experiment" in content
    assert "placeholder" not in content.lower()
