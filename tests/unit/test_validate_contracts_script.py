import subprocess
import sys


def test_validate_contracts_script_passes_for_placeholder_schemas():
    result = subprocess.run(
        [sys.executable, "scripts/validate_contracts.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "contracts validation passed" in result.stdout.lower()
