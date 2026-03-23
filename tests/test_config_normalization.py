"""Configuration normalization tests for the new env/model schema."""

from seenerl.config import load_config


def test_legacy_env_name_maps_to_nested_env_id():
    config = load_config("configs/sac.yaml", ["--env_name", "Pendulum-v1"])
    assert config.env.id == "Pendulum-v1"
    assert config.env_name == "Pendulum-v1"
    assert config.env.backend == "gymnasium"
    assert config.env.num_envs == 1


def test_nested_env_id_stays_available_on_top_level():
    config = load_config(
        "configs/sac.yaml",
        ["--env.id", "Pendulum-v1", "--env.backend", "gymnasium"],
    )
    assert config.env.id == "Pendulum-v1"
    assert config.env_name == "Pendulum-v1"


def test_isaaclab_pickplace_defaults_include_explicit_task_import():
    config = load_config("configs/isaaclab_pickplace_sac.yaml")
    assert config.env.backend == "isaaclab"
    assert config.env.id == "Isaac-PickPlace-GR1T2-Abs-v0"
    assert config.env.isaaclab.task_imports == [
        "isaaclab_tasks.manager_based.manipulation.pick_place"
    ]
