import subprocess
import sys


def test_main_cli_exposes_train_and_evaluate_subcommands():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
