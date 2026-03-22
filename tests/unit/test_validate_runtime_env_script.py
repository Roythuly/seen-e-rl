import importlib.util
from pathlib import Path
import subprocess
import sys


def _load_validate_runtime_env_module():
    path = Path("scripts/validate_runtime_env.py").resolve()
    spec = importlib.util.spec_from_file_location("validate_runtime_env_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_runtime_env_script_passes_for_local_runtime():
    result = subprocess.run(
        [sys.executable, "scripts/validate_runtime_env.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime env validation passed" in result.stdout.lower()


def test_validate_runtime_env_builds_single_candidate_from_explicit_backend(monkeypatch):
    module = _load_validate_runtime_env_module()
    monkeypatch.setenv("MUJOCO_GL", "glfw")

    candidates = module.build_backend_candidates()

    assert candidates == ["glfw"]


def test_validate_runtime_env_falls_back_to_next_backend(monkeypatch):
    module = _load_validate_runtime_env_module()
    calls: list[str | None] = []

    def fake_run(args, check, capture_output, text, env):
        del args, check, capture_output, text
        calls.append(env.get("MUJOCO_GL"))
        if env.get("MUJOCO_GL") == "egl":
            return subprocess.CompletedProcess([], 1, stdout="egl failed\n", stderr="")
        return subprocess.CompletedProcess([], 0, stdout="runtime env validation passed\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code, output = module.run_backend_probe_sequence(
        script_path=Path("scripts/validate_runtime_env.py").resolve(),
        python_executable=sys.executable,
        candidates=["egl", "auto"],
    )

    assert exit_code == 0
    assert "runtime env validation passed" in output
    assert calls == ["egl", None]
