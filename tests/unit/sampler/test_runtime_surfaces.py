from __future__ import annotations

import numpy as np
import torch

from rl_training_infra.model import ModelFactory
from rl_training_infra.sampler import GymEnvFactory, TorchActorHandle


def test_gym_env_factory_and_actor_handle_work_together() -> None:
    env = GymEnvFactory().create({"id": "Pendulum-v1"}, seed=7)
    observation, _ = env.reset(seed=7)

    model = ModelFactory.build(
        {
            "algorithm": "td3",
            "encoder": {"kind": "mlp", "input_dim": int(np.asarray(observation).shape[0]), "hidden_sizes": [16]},
            "actor_head": {"kind": "deterministic_policy", "action_dim": int(np.prod(env.action_space.shape))},
            "critic_head": {"kind": "twin_q", "action_dim": int(np.prod(env.action_space.shape))},
        },
        {"name": "torch"},
    )
    actor_handle = TorchActorHandle(model)

    action_output = actor_handle.act({"observations": observation})

    assert action_output["policy_version"] == model.policy_version
    assert action_output["action"].shape[-1] == env.action_space.shape[0]
    env.close()


def test_torch_actor_handle_accepts_nested_observation_batch() -> None:
    model = ModelFactory.build(
        {
            "algorithm": "ppo",
            "encoder": {"kind": "mlp", "input_dim": 4, "hidden_sizes": [8]},
            "actor_head": {"kind": "gaussian_policy", "action_dim": 2},
            "critic_head": {"kind": "value_head"},
        },
        {"name": "torch"},
    )
    actor_handle = TorchActorHandle(model)

    action_output = actor_handle.act({"observations": torch.randn(3, 4)})

    assert set(action_output).issuperset({"action", "policy_version"})
