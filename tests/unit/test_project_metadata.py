from pathlib import Path


def test_pyproject_declares_imageio_for_mujoco_renderer_support():
    content = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"imageio>=' in content or '"imageio"' in content
