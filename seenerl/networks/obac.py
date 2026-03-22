"""OBAC-specific network implementations registered for model factory use."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from seenerl.networks.base import BaseActor, BaseCritic, weights_init_
from seenerl.networks.registry import register_actor, register_critic


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


@register_critic("obac_value_network")
class OBACValueNetwork(BaseCritic):
    """Value network used by OBAC's behavior policy branch."""

    def __init__(self, num_inputs: int, hidden_dim: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        x = torch.tanh(self.layer_norm(self.linear1(state)))
        x = F.elu(self.linear2(x))
        return self.linear3(x)


@register_critic("obac_q_network")
class OBACQNetwork(BaseCritic):
    """Twin Q-network used by OBAC."""

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        xu = torch.cat([state, action], dim=1)

        x1 = torch.tanh(self.layer_norm1(self.linear1(xu)))
        x1 = F.elu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = torch.tanh(self.layer_norm2(self.linear4(xu)))
        x2 = F.elu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


@register_actor("obac_gaussian")
class OBACGaussianActor(BaseActor):
    """Gaussian policy used by OBAC."""

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int = 256,
                 action_space=None):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.as_tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            )
            self.action_bias = torch.as_tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            )

    def forward(self, state: torch.Tensor):
        x = torch.tanh(self.layer_norm(self.linear1(state)))
        x = F.elu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def get_log_density(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (action - self.action_bias) / self.action_scale
        return normal.log_prob(y_t)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
