from pathlib import Path


def test_run_tests_script_disables_external_pytest_plugin_autoload():
    content = Path("scripts/run_tests.sh").read_text(encoding="utf-8")

    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in content
    assert "validate_runtime_env.py" in content
