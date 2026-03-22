"""
Soft Actor-Critic (SAC) algorithm implementation.

Reference: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2018.
"""

import copy
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from seenerl.algorithms.base import BaseAlgorithm
from seenerl.networks.mlp import GaussianActor, DeterministicActor, MLPCritic
from seenerl.utils import soft_update, hard_update


class SAC(BaseAlgorithm):
    """
    Soft Actor-Critic with automatic entropy tuning.

    Supports both Gaussian (stochastic) and Deterministic policy types.
    """

    def __init__(self, obs_dim: int, action_space, config):
        """
        Args:
            obs_dim: Observation space dimension.
            action_space: Gymnasium action space.
            config: Configuration object with SAC parameters.
        """
        device = torch.device(config.device) if config.device != "auto" else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        super().__init__(device)

        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.policy_type = config.get("policy_type", "Gaussian")
        self.target_update_interval = config.get("target_update_interval", 1)
        self.automatic_entropy_tuning = config.get("automatic_entropy_tuning", False)

        action_dim = action_space.shape[0]
        hidden_size = config.hidden_size

        # Critic
        self.critic = MLPCritic(obs_dim, action_dim, hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.lr)
        self.critic_target = MLPCritic(obs_dim, action_dim, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Policy
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)
                ).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=config.lr)

            self.policy = GaussianActor(
                obs_dim, action_dim, hidden_size, action_space
            ).to(self.device)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicActor(
                obs_dim, action_dim, hidden_size, action_space
            ).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=config.lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.policy.sample(state_t)
        else:
            action, _, _ = self.policy.sample(state_t)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size: int, updates: int) -> Dict[str, float]:
        """
        Update SAC parameters from replay buffer.

        Returns:
            Dict with keys: critic_1_loss, critic_2_loss, policy_loss, entropy_loss, alpha
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = \
            memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        # Compute target Q
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next, qf2_next = self.critic_target(next_state_batch, next_action)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next

        # Critic loss
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Policy loss
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Entropy tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()
        else:
            alpha_loss_val = 0.0

        # Target update
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_1_loss": qf1_loss.item(),
            "critic_2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": alpha_loss_val,
            "alpha": self.alpha if isinstance(self.alpha, float) else self.alpha,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        state = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }
        if self.automatic_entropy_tuning:
            state["log_alpha"] = self.log_alpha.detach().cpu()
            state["alpha_optim"] = self.alpha_optim.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], evaluate: bool = False) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.policy_optim.load_state_dict(state_dict["policy_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        if self.automatic_entropy_tuning and "log_alpha" in state_dict:
            self.log_alpha = state_dict["log_alpha"].to(self.device).requires_grad_(True)
            self.alpha_optim.load_state_dict(state_dict["alpha_optim"])
            self.alpha = self.log_alpha.exp().item()

        if evaluate:
            self.policy.eval()
            self.critic.eval()
        else:
            self.policy.train()
            self.critic.train()
