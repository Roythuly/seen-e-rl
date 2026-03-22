from pathlib import Path
import subprocess
import sys

import yaml


def _write_cli_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "ppo_cli.yaml"
    payload = {
        "run_name": "ppo-cli-test",
        "seed": 3,
        "backend": {"name": "torch", "runtime_mode": "train", "device": "cpu"},
        "env": {"id": "Pendulum-v1"},
        "model": {
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [16, 16]},
            "actor_head": {"kind": "gaussian_policy", "action_dim": 1},
            "critic_head": {"kind": "value_head"},
        },
        "algo": {"name": "ppo"},
        "sampler": {"mode": "trajectory"},
        "trainer": {
            "runtime": {
                "collection_schedule": {
                    "mode": "rollout",
                    "unit": "env_step",
                    "amount": 8,
                    "freeze_policy_during_collection": True,
                },
                "update_schedule": {
                    "trigger_unit": "collection",
                    "updates_per_trigger": 1,
                    "epochs": 2,
                    "minibatch_size": 4,
                },
                "publish_schedule": {"strategy": "after_update"},
                "checkpoint": {"enabled": True},
                "max_env_steps": 8,
            }
        },
        "buffer": {"capacity": 8, "batch_size": 4, "sampling_mode": "fifo", "min_ready_size": 8},
        "eval": {"selector": "latest", "seeds": [5], "episodes_per_seed": 1},
        "artifacts": {"root": str(tmp_path / "artifacts")},
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_main_cli_exposes_train_and_evaluate_subcommands():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "train" in result.stdout
    assert "evaluate" in result.stdout


def test_main_cli_can_train_and_evaluate_small_config(tmp_path: Path):
    config_path = _write_cli_config(tmp_path)
    artifacts_root = tmp_path / "artifacts"

    train = subprocess.run(
        [sys.executable, "main.py", "train", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    assert (artifacts_root / "run_state.json").exists()

    evaluate = subprocess.run(
        [sys.executable, "main.py", "evaluate", "--config", str(config_path), "--selector", "latest"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert evaluate.returncode == 0, evaluate.stdout + evaluate.stderr
    assert (artifacts_root / "eval_latest.json").exists()


def test_main_cli_reports_missing_algo_name_as_config_error(tmp_path: Path):
    config_path = _write_cli_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload.pop("algo")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "main.py", "train", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "algo.name" in result.stderr
    assert "KeyError" not in result.stderr


def test_main_cli_reports_unknown_algorithm_as_config_error(tmp_path: Path):
    config_path = _write_cli_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["algo"]["name"] = "unknown"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "main.py", "train", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unsupported algorithm" in result.stderr
    assert "unknown" in result.stderr
