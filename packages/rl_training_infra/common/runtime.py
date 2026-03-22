from __future__ import annotations

import os


def ensure_headless_mujoco_backend(default_backend: str = "egl") -> str:
    existing = os.environ.get("MUJOCO_GL")
    if existing:
        return existing
    os.environ["MUJOCO_GL"] = default_backend
    return default_backend
