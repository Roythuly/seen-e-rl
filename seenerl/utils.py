"""Utility functions for RL training."""

import random

import numpy as np
import torch


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


def to_numpy(value) -> np.ndarray:
    """Convert torch / numpy / scalar inputs to numpy arrays."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sample_batched_actions(action_space, num_envs: int) -> np.ndarray:
    """Sample a batch of actions from a Gym action space."""
    return np.stack([action_space.sample() for _ in range(num_envs)], axis=0).astype(np.float32)


def resolve_buffer_next_states(next_states: np.ndarray, info: dict) -> np.ndarray:
    """Use final observations when an autoreset wrapper exposes them."""
    next_states = np.asarray(next_states, dtype=np.float32).copy()
    final_mask = info.get("final_mask")
    final_observation = info.get("final_observation")
    if final_mask is None or final_observation is None:
        return next_states

    final_mask = np.asarray(final_mask, dtype=np.bool_)
    final_observation = np.asarray(final_observation, dtype=np.float32)
    next_states[final_mask] = final_observation[final_mask]
    return next_states
