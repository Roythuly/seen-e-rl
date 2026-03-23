"""
Rollout buffer for on-policy algorithms (PPO).

Collects a fixed number of timesteps, then computes GAE advantages
and returns for training. Data is discarded after each training cycle.
"""

from typing import Generator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """
    Stores rollout data for on-policy training.

    Workflow:
        1. Collect `rollout_steps` transitions via `add()`
        2. Call `compute_returns_and_advantages()`
        3. Iterate mini-batches via `get_mini_batches()`
        4. Call `reset()` to prepare for next rollout
    """

    def __init__(self, rollout_steps: int, num_envs: int, obs_dim: int, action_dim: int):
        """
        Args:
            rollout_steps: Number of transitions to collect per rollout.
            obs_dim: Dimension of observation space.
            action_dim: Dimension of action space.
        """
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        """Clear all stored data for a new rollout."""
        self.states = np.zeros(
            (self.rollout_steps, self.num_envs, self.obs_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.rollout_steps, self.num_envs, self.action_dim),
            dtype=np.float32,
        )
        self.rewards = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.terminated = np.zeros((self.rollout_steps, self.num_envs), dtype=np.bool_)
        self.truncated = np.zeros((self.rollout_steps, self.num_envs), dtype=np.bool_)
        self.log_probs = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.next_values = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)

        self.advantages = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)

        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        next_value: np.ndarray,
    ) -> None:
        """Add a batched transition to the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.terminated[self.position] = terminated
        self.truncated[self.position] = truncated
        self.log_probs[self.position] = log_prob
        self.values[self.position] = value
        self.next_values[self.position] = next_value
        self.position += 1

    @property
    def is_full(self) -> bool:
        """Check if buffer has collected the full rollout."""
        return self.position >= self.rollout_steps

    def compute_returns_and_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> None:
        """
        Compute GAE advantages and discounted returns.

        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        """
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        done = np.logical_or(self.terminated, self.truncated)
        for t in reversed(range(self.rollout_steps)):
            next_value = self.next_values[t]
            next_non_terminal = 1.0 - done[t].astype(np.float32)
            next_non_terminal_value = 1.0 - self.terminated[t].astype(np.float32)

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal_value
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_mini_batches(
        self, num_mini_batch: int, device: torch.device
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Yield shuffled mini-batches from the rollout data.

        Args:
            num_mini_batch: Number of mini-batches to split data into.
            device: Torch device to move tensors to.

        Yields:
            (states, actions, old_log_probs, advantages, returns) tensors.
        """
        batch_size = self.rollout_steps * self.num_envs
        mini_batch_size = max(batch_size // num_mini_batch, 1)
        indices = np.random.permutation(batch_size)

        # Normalize advantages
        adv = self.advantages.reshape(-1).copy()
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-8
        adv = (adv - adv_mean) / adv_std

        states = self.states.reshape(batch_size, self.obs_dim)
        actions = self.actions.reshape(batch_size, self.action_dim)
        log_probs = self.log_probs.reshape(batch_size, 1)
        returns = self.returns.reshape(batch_size, 1)
        advantages = adv.reshape(batch_size, 1)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            idx = indices[start:end]
            yield (
                torch.as_tensor(states[idx], dtype=torch.float32, device=device),
                torch.as_tensor(actions[idx], dtype=torch.float32, device=device),
                torch.as_tensor(log_probs[idx], dtype=torch.float32, device=device),
                torch.as_tensor(advantages[idx], dtype=torch.float32, device=device),
                torch.as_tensor(returns[idx], dtype=torch.float32, device=device),
            )
