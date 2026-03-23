"""PPO action mapping tests."""

import numpy as np

from seenerl.algorithms import build_algorithm
from seenerl.config import load_config
from seenerl.envs import create_env


def test_ppo_maps_env_actions_into_box_space():
    config = load_config(
        "configs/ppo.yaml",
        [
            "--env_name", "Pendulum-v1",
            "--env.num_envs", "2",
            "--device", "cpu",
            "--eval", "false",
        ],
    )

    env = create_env(config)
    agent = build_algorithm(config, env.observation_space, env.action_space)
    state, _ = env.reset(seed=42)

    for _ in range(128):
        env_action, log_prob, value, raw_action = agent.select_action(state)
        assert env_action.shape == (2, 1)
        assert raw_action.shape == (2, 1)
        assert np.all(env_action >= env.action_space.low - 1e-6)
        assert np.all(env_action <= env.action_space.high + 1e-6)
        assert log_prob.shape == (2,)
        assert value.shape == (2,)

    eval_action = agent.select_action(state, evaluate=True)
    assert eval_action.shape == (2, 1)
    assert np.all(eval_action >= env.action_space.low - 1e-6)
    assert np.all(eval_action <= env.action_space.high + 1e-6)

    env.close()
