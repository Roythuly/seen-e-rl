"""Batched replay and rollout buffer tests."""

import numpy as np
import torch

from seenerl.buffers.replay_buffer import ReplayBuffer
from seenerl.buffers.rollout_buffer import RolloutBuffer


def test_replay_buffer_add_batch():
    buffer = ReplayBuffer(capacity=8, obs_dim=3, action_dim=2, seed=0)
    buffer.add_batch(
        states=np.arange(9, dtype=np.float32).reshape(3, 3),
        actions=np.arange(6, dtype=np.float32).reshape(3, 2),
        rewards=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        next_states=np.arange(9, 18, dtype=np.float32).reshape(3, 3),
        dones=np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )

    assert len(buffer) == 3
    batch = buffer.sample(batch_size=2)
    assert batch[0].shape == (2, 3)
    assert batch[1].shape == (2, 2)
    assert batch[2].shape == (2, 1)


def test_rollout_buffer_handles_batched_gae_and_minibatches():
    buffer = RolloutBuffer(rollout_steps=4, num_envs=2, obs_dim=3, action_dim=2)

    for step in range(4):
        buffer.add(
            state=np.full((2, 3), step, dtype=np.float32),
            action=np.full((2, 2), step, dtype=np.float32),
            reward=np.array([1.0, 0.5], dtype=np.float32),
            done=np.array([step == 3, False], dtype=np.float32),
            log_prob=np.array([0.1, 0.2], dtype=np.float32),
            value=np.array([0.3, 0.4], dtype=np.float32),
        )

    buffer.compute_returns_and_advantages(
        last_value=np.array([0.0, 0.1], dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert buffer.advantages.shape == (4, 2)
    assert buffer.returns.shape == (4, 2)

    mini_batches = list(buffer.get_mini_batches(num_mini_batch=2, device=torch.device("cpu")))
    assert len(mini_batches) == 2
    states, actions, log_probs, advantages, returns = mini_batches[0]
    assert states.shape[-1] == 3
    assert actions.shape[-1] == 2
    assert log_probs.shape[-1] == 1
    assert advantages.shape[-1] == 1
    assert returns.shape[-1] == 1
