"""Utility functions for RL training."""

import torch
import numpy as np
import random


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Soft update target network: θ_target = τ*θ_source + (1-τ)*θ_target."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Hard update target network: θ_target = θ_source."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    """
    Resolve device string to torch.device.

    Args:
        device_str: "auto", "cpu", "cuda", "cuda:0", etc.

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
