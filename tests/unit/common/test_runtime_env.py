from __future__ import annotations

import os

from rl_training_infra.sampler import GymEnvFactory
from rl_training_infra.common import ensure_headless_mujoco_backend


def test_ensure_headless_mujoco_backend_sets_egl_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    backend = ensure_headless_mujoco_backend()

    assert backend == "egl"
    assert os.environ["MUJOCO_GL"] == "egl"


def test_ensure_headless_mujoco_backend_preserves_existing_value(monkeypatch) -> None:
    monkeypatch.setenv("MUJOCO_GL", "glfw")

    backend = ensure_headless_mujoco_backend()

    assert backend == "glfw"
    assert os.environ["MUJOCO_GL"] == "glfw"


def test_gym_env_factory_initializes_headless_backend(monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    env = GymEnvFactory().create({"id": "Pendulum-v1"}, seed=7)

    try:
        assert os.environ["MUJOCO_GL"] == "egl"
    finally:
        env.close()
