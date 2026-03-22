"""Abstract base classes for actor and critic networks."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def weights_init_(m: nn.Module) -> None:
    """Xavier uniform initialization for Linear layers."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class BaseActor(nn.Module, ABC):
    """
    Abstract base class for actor (policy) networks.

    Subclasses must implement `forward` and `sample`.
    This provides the extension point for MLP, CNN, Transformer, VLA, etc.
    """

    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, state: torch.Tensor):
        """
        Sample an action from the policy.

        Returns:
            Tuple of (action, log_prob, mean) for stochastic policies,
            or (action, 0, mean) for deterministic policies.
        """
        raise NotImplementedError


class BaseCritic(nn.Module, ABC):
    """
    Abstract base class for critic (value/Q) networks.

    Subclasses must implement `forward`.
    Supports both single-Q and twin-Q architectures.
    """

    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor = None):
        """
        Forward pass.

        For Q-networks: takes (state, action), returns Q-value(s).
        For V-networks: takes (state), returns V-value.
        """
        raise NotImplementedError
