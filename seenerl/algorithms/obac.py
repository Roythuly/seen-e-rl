"""
Offline-Boosted Actor-Critic (OBAC) algorithm.

Reference: Luo et al., "Offline-Boosted Actor-Critic: Adaptively Blending Optimal Historical Behaviors in Deep Off-Policy RL", 2024.
"""

import copy
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from seenerl.algorithms.base import BaseAlgorithm
from seenerl.utils import soft_update


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    """Xavier uniform initialization for Linear layers."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class OBACValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(OBACValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = torch.tanh(self.LayerNorm((self.linear1(state))))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)
        return x


class OBACQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(OBACQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm1 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm2 = nn.LayerNorm(hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = torch.tanh(self.LayerNorm1(self.linear1(xu)))
        x1 = F.elu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = torch.tanh(self.LayerNorm2(self.linear4(xu)))
        x2 = F.elu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class OBACGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(OBACGaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = torch.tanh(self.LayerNorm((self.linear1(state))))
        x = F.elu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def get_log_density(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (action - self.action_bias) / self.action_scale
        log_density = normal.log_prob(y_t)
        return log_density

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(OBACGaussianPolicy, self).to(device)


class OBAC(BaseAlgorithm):
    """Offline-Boosted Actor-Critic (OBAC) algorithm."""

    def __init__(self, obs_dim: int, action_space, config):
        device = torch.device(config.device) if config.device != "auto" else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        super().__init__(device)

        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.get("alpha", 0.2)
        self.quantile = config.get("quantile", 0.9)
        self.bc_weight = config.get("bc_weight", 0.1)

        self.target_update_interval = config.get("target_update_interval", 1)
        self.automatic_entropy_tuning = config.get("automatic_entropy_tuning", True)

        action_dim = action_space.shape[0]
        hidden_size = config.hidden_size

        self.critic = OBACQNetwork(obs_dim, action_dim, hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.lr)

        self.critic_target = OBACQNetwork(obs_dim, action_dim, hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target Entropy
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=config.lr)

        self.policy = OBACGaussianPolicy(obs_dim, action_dim, hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=config.lr)
        
        # Behavior policy critics
        self.critic_buffer = OBACQNetwork(obs_dim, action_dim, hidden_size).to(self.device)
        self.critic_buffer_optim = Adam(self.critic_buffer.parameters(), lr=config.lr)
        self.critic_buffer.load_state_dict(self.critic.state_dict())

        self.critic_target_buffer = OBACQNetwork(obs_dim, action_dim, hidden_size).to(self.device)
        self.critic_target_buffer.load_state_dict(self.critic_buffer.state_dict())

        self.V_critic_buffer = OBACValueNetwork(obs_dim, hidden_size).to(self.device)
        self.V_critic_buffer_optim = Adam(self.V_critic_buffer.parameters(), lr=config.lr)

        self._last_policy_loss = 0.0
        self._last_alpha_loss = 0.0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(state_t)
            else:
                action, _, _ = self.policy.sample(state_t)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size: int, updates: int) -> Dict[str, float]:
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # compute the Q loss for current policy
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        # Compute the target Q value for behavior policy
        vf_pred = self.V_critic_buffer(state_batch)
        target_Vf_pred = self.V_critic_buffer(next_state_batch)
        next_q_value_buffer = reward_batch + mask_batch * self.gamma * target_Vf_pred
        
        # compute the Q loss for behavior policy
        qf1_buffer, qf2_buffer = self.critic_buffer(state_batch, action_batch)
        qf_buffer = torch.min(qf1_buffer, qf2_buffer).mean()
        qf1_buffer_loss = F.mse_loss(qf1_buffer, next_q_value_buffer)  
        qf2_buffer_loss = F.mse_loss(qf2_buffer, next_q_value_buffer)
        qf_buffer_loss = qf1_buffer_loss + qf2_buffer_loss
        
        # compute the V loss for behavior policy
        with torch.no_grad():
            q_pred_1, q_pred_2 = self.critic_target_buffer(state_batch, action_batch)
            q_pred = torch.min(q_pred_1, q_pred_2)
            
        vf_err = q_pred - vf_pred
        vf_sign = (vf_err < 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()
        
        # compute action by current policy
        with torch.no_grad():
            pi_no_grad, _, _ = self.policy.sample(state_batch)
            qf1_pi_no_grad, qf2_pi_no_grad = self.critic(state_batch, pi_no_grad)
            qf_pi = torch.min(qf1_pi_no_grad, qf2_pi_no_grad).mean().item()
            qf_buffer_item = qf_buffer.item()

        # update Q value of current policy
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        # update Q value of behavior policy
        self.critic_buffer_optim.zero_grad()
        qf_buffer_loss.backward()
        self.critic_buffer_optim.step()
        
        # update V value of behavior policy
        self.V_critic_buffer_optim.zero_grad()
        vf_loss.backward()
        self.V_critic_buffer_optim.step()
        
        if updates % self.target_update_interval == 0:
            pi, log_pi, _ = self.policy.sample(state_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            
            if qf_pi >= qf_buffer_item:
                policy_loss = (self.alpha * log_pi - min_qf_pi).mean()
            else:
                log_density = self.policy.get_log_density(state_batch, action_batch)
                log_density = torch.clamp(log_density, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
                policy_loss = (self.alpha * log_pi - self.bc_weight * log_density - min_qf_pi).mean()
            
            # update policy
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp().item()
                self._last_alpha_loss = alpha_loss.item()
            
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.critic_target_buffer, self.critic_buffer, self.tau)
            
            self._last_policy_loss = policy_loss.item()
            
        return {
            "critic_loss": qf_loss.item(),
            "critic_buffer_loss": qf_buffer_loss.item(),
            "v_buffer_loss": vf_loss.item(),
            "policy_loss": self._last_policy_loss,
            "alpha_loss": self._last_alpha_loss,
            "alpha": self.alpha,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        state_dict = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "critic_buffer": self.critic_buffer.state_dict(),
            "critic_target_buffer": self.critic_target_buffer.state_dict(),
            "V_critic_buffer": self.V_critic_buffer.state_dict(),
        }
        if self.automatic_entropy_tuning:
            state_dict["log_alpha"] = self.log_alpha
            state_dict["alpha_optim"] = self.alpha_optim.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], evaluate: bool = False) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.policy_optim.load_state_dict(state_dict["policy_optim"])
        self.critic_buffer.load_state_dict(state_dict["critic_buffer"])
        self.critic_target_buffer.load_state_dict(state_dict["critic_target_buffer"])
        self.V_critic_buffer.load_state_dict(state_dict["V_critic_buffer"])

        if self.automatic_entropy_tuning and "log_alpha" in state_dict:
            self.log_alpha = state_dict["log_alpha"]
            self.alpha_optim.load_state_dict(state_dict["alpha_optim"])
            self.alpha = self.log_alpha.exp().item()

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_buffer.eval()
            self.V_critic_buffer.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_buffer.train()
            self.V_critic_buffer.train()
