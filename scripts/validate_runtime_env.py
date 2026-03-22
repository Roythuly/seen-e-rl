from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

def configure_probe_backend(backend: str) -> None:
    if backend == "auto":
        os.environ.pop("MUJOCO_GL", None)
        return
    os.environ["MUJOCO_GL"] = backend


def build_backend_candidates() -> list[str]:
    explicit_backend = os.environ.get("MUJOCO_GL")
    if explicit_backend:
        return [explicit_backend]
    return ["egl", "osmesa", "glfw", "auto"]


def run_single_probe(backend: str) -> int:
    configure_probe_backend(backend)
    try:
        import gymnasium as gym
        import torch
    except Exception as exc:  # pragma: no cover - import failures are the behavior under test
        print(f"runtime env validation failed: import error: {exc}")
        return 1

    try:
        env = gym.make("Humanoid-v5")
        observations, _ = env.reset(seed=1)
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        env.close()
    except Exception as exc:  # pragma: no cover - runtime failures are the behavior under test
        print(f"runtime env validation failed: env bootstrap error: {exc}")
        return 1

    print("runtime env validation passed")
    print(f"torch={torch.__version__}")
    print(f"mujoco_gl={os.environ.get('MUJOCO_GL', 'auto')}")
    print(f"obs_shape={getattr(observations, 'shape', None)}")
    print(f"terminated={terminated} truncated={truncated}")
    print(f"info_keys={sorted(info.keys())[:5]}")
    return 0


def run_backend_probe_sequence(
    *,
    script_path: Path,
    python_executable: str,
    candidates: list[str],
) -> tuple[int, str]:
    failures: list[str] = []
    for candidate in candidates:
        env = os.environ.copy()
        if candidate == "auto":
            env.pop("MUJOCO_GL", None)
        else:
            env["MUJOCO_GL"] = candidate
        result = subprocess.run(
            [python_executable, str(script_path), "--probe-backend", candidate],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0:
            return 0, output
        failures.append(f"[{candidate}] {output or 'probe failed'}")
    return 1, "\n".join(failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the local RL runtime stack.")
    parser.add_argument("--probe-backend", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.probe_backend is not None:
        return run_single_probe(args.probe_backend)

    exit_code, output = run_backend_probe_sequence(
        script_path=Path(__file__).resolve(),
        python_executable=sys.executable,
        candidates=build_backend_candidates(),
    )
    if output:
        print(output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
