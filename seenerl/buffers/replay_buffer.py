"""
Replay buffer for off-policy algorithms (SAC, TD3).

Uses numpy arrays for memory-efficient storage.
Supports save/load for training resumption.
"""

import os
import pickle
import random
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Fixed-size replay buffer storing (s, a, r, s', done) transitions.

    Uses pre-allocated numpy arrays for O(1) insertion and efficient sampling.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 42):
        """
        Args:
            capacity: Maximum number of transitions to store.
            obs_dim: Dimension of observation space.
            action_dim: Dimension of action space.
            seed: Random seed for sampling.
        """
        random.seed(seed)
        np.random.seed(seed)

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.position = 0
        self.size = 0

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: float) -> None:
        """Add a transition to the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Randomly sample a batch of transitions.

        Returns:
            (states, actions, rewards, next_states, dones) numpy arrays.
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size

    def save(self, path: str) -> None:
        """Save buffer state to disk for training resumption."""
        data = {
            "states": self.states[:self.size],
            "actions": self.actions[:self.size],
            "rewards": self.rewards[:self.size],
            "next_states": self.next_states[:self.size],
            "dones": self.dones[:self.size],
            "position": self.position,
            "size": self.size,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load buffer state from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        size = data["size"]
        self.states[:size] = data["states"]
        self.actions[:size] = data["actions"]
        self.rewards[:size] = data["rewards"]
        self.next_states[:size] = data["next_states"]
        self.dones[:size] = data["dones"]
        self.position = data["position"]
        self.size = size
