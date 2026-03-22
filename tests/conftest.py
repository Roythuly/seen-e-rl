from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def restore_mujoco_gl_environment():
    sentinel = object()
    original = os.environ.get("MUJOCO_GL", sentinel)

    yield

    if original is sentinel:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ["MUJOCO_GL"] = str(original)
