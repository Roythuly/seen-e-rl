import subprocess
import sys


def test_validate_runtime_env_script_passes_for_local_runtime():
    result = subprocess.run(
        [sys.executable, "scripts/validate_runtime_env.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime env validation passed" in result.stdout.lower()
