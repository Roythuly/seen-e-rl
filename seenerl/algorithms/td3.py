"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

Reference: Fujimoto et al., "Addressing Function Approximation Error in
Actor-Critic Methods", 2018.
"""

import copy
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from seenerl.algorithms.base import BaseAlgorithm
from seenerl.networks.mlp import DeterministicActor, MLPCritic
from seenerl.utils import soft_update


class TD3(BaseAlgorithm):
    """Twin Delayed DDPG with target policy smoothing and delayed updates."""

    def __init__(self, obs_dim: int, action_space, config):
        device = torch.device(config.device) if config.device != "auto" else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        super().__init__(device)

        action_dim = action_space.shape[0]
        hidden_size = config.hidden_size

        self.tau = config.tau
        self.discount = config.gamma
        self.policy_freq = config.get("policy_freq", 2)
        self.max_action = float(action_space.high[0])
        self.noise_clip = config.get("noise_clip", 0.5) * self.max_action
        self.policy_noise = config.get("policy_noise", 0.2) * self.max_action
        self.exploration_noise = config.get("exploration_noise", 0.1)

        # Networks
        self.policy = DeterministicActor(
            obs_dim, action_dim, hidden_size, action_space
        ).to(self.device)
        self.policy_target = copy.deepcopy(self.policy)
        self.policy_optim = Adam(self.policy.parameters(), lr=config.lr)

        self.critic = MLPCritic(obs_dim, action_dim, hidden_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.lr)

        self._last_policy_loss = 0.0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.policy(state_t).detach().cpu().numpy()[0]
        
        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise * self.max_action, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            
        return action

    def update_parameters(self, memory, batch_size: int, updates: int) -> Dict[str, float]:
        """
        Update TD3 parameters from replay buffer.

        Returns:
            Dict with keys: critic_loss, policy_loss
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = \
            memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        with torch.no_grad():
            # Target policy smoothing
            noise = (
                torch.randn_like(action_batch) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.policy_target(next_state_batch) + noise
            ).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + mask_batch * self.discount * target_q

        # Critic update
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Delayed policy update
        if updates % self.policy_freq == 0:
            policy_loss_q1, _ = self.critic(state_batch, self.policy(state_batch))
            policy_loss = -policy_loss_q1.mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Soft update targets
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.policy_target, self.policy, self.tau)

            self._last_policy_loss = policy_loss.item()

        return {
            "critic_loss": critic_loss.item(),
            "policy_loss": self._last_policy_loss,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "policy_target": self.policy_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any], evaluate: bool = False) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.policy_target.load_state_dict(state_dict["policy_target"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.policy_optim.load_state_dict(state_dict["policy_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])

        if evaluate:
            self.policy.eval()
            self.critic.eval()
        else:
            self.policy.train()
            self.critic.train()
