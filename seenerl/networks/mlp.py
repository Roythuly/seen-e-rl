"""MLP-based actor and critic network implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from seenerl.networks.base import BaseActor, BaseCritic, weights_init_
from seenerl.networks.registry import register_actor, register_critic

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


@register_actor("gaussian")
class GaussianActor(BaseActor):
    """
    Gaussian policy network for continuous action spaces.

    Supports squashed Gaussian distribution (tanh transform) for SAC,
    and unsquashed distribution for PPO.
    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int = 256,
                 action_space=None, squash: bool = True):
        super().__init__()
        self.squash = squash
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        
        # PPO usually prefers state-independent log_std, but state-dependent is okay.
        # We keep state-dependent log_std for compatibility with SAC.
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state: torch.Tensor):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        
        if self.squash:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = x_t
            log_prob = normal.log_prob(x_t).sum(1, keepdim=True)
            mean_action = mean

        return action, log_prob, mean_action

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        entropy = normal.entropy().sum(-1, keepdim=True)
        return log_prob, entropy

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


@register_actor("deterministic")
class DeterministicActor(BaseActor):
    """
    Deterministic policy network for continuous action spaces.

    Used by TD3 and deterministic SAC variant.
    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int = 256,
                 action_space=None):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)
        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state: torch.Tensor):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super().to(device)


@register_critic("q_network")
class MLPCritic(BaseCritic):
    """
    Twin Q-network for off-policy algorithms (SAC, TD3).

    Contains two independent Q-networks for the clipped double-Q trick.
    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        # Q1
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


@register_critic("value_network")
class MLPValue(BaseCritic):
    """
    State-value network V(s) for on-policy algorithms (PPO).
    """

    def __init__(self, num_inputs: int, hidden_dim: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
