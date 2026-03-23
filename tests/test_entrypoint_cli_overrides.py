"""CLI entry-point override tests."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import parse_eval_args_and_load_config
from render.renderer import parse_render_args_and_load_config


def test_evaluate_entry_point_accepts_config_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config", "configs/sac.yaml",
            "--checkpoint", "/tmp/mock.pt",
            "--env_name", "Ant-v5",
        ],
    )

    args, config = parse_eval_args_and_load_config()

    assert args.checkpoint == "/tmp/mock.pt"
    assert config.env.id == "Ant-v5"


def test_render_entry_point_accepts_config_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "renderer.py",
            "--config", "configs/sac.yaml",
            "--checkpoint", "/tmp/mock.pt",
            "--env.id", "Ant-v5",
        ],
    )

    args, config = parse_render_args_and_load_config()

    assert args.checkpoint == "/tmp/mock.pt"
    assert config.env.id == "Ant-v5"
    assert config.device == "cpu"
