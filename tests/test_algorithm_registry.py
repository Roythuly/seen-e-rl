"""Algorithm factory smoke tests."""

import gymnasium as gym

from seenerl.algorithms import build_algorithm, get_algorithm_spec
from seenerl.config import load_config


def test_algorithm_registry_exposes_expected_trainer_kinds():
    assert get_algorithm_spec("SAC").trainer_kind == "off_policy"
    assert get_algorithm_spec("TD3").trainer_kind == "off_policy"
    assert get_algorithm_spec("OBAC").trainer_kind == "off_policy"
    assert get_algorithm_spec("PPO").trainer_kind == "on_policy"


def test_build_algorithm_from_registry():
    env = gym.make("Pendulum-v1")
    config = load_config("configs/sac.yaml", ["--env_name", "Pendulum-v1", "--device", "cpu"])
    agent = build_algorithm(config, env.observation_space, env.action_space)
    assert agent.__class__.__name__ == "SAC"
    env.close()
