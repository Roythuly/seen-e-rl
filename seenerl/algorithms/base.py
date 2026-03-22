"""
Base class for all RL algorithms.

Provides a unified interface for action selection, parameter updates,
and checkpoint save/load. All algorithms (SAC, TD3, PPO) inherit from this.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch


class BaseAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.

    Subclasses must implement:
        - select_action: Choose an action given a state.
        - update_parameters: Update network parameters, return loss dict.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def _prepare_state_tensor(self, state) -> tuple[torch.Tensor, bool]:
        """Convert a single or batched observation into a batched tensor."""
        if isinstance(state, torch.Tensor):
            state_t = state.to(self.device, dtype=torch.float32)
        else:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        single = state_t.ndim == 1
        if single:
            state_t = state_t.unsqueeze(0)
        return state_t, single

    @staticmethod
    def _format_action_output(action: torch.Tensor, single: bool) -> np.ndarray:
        action_np = action.detach().cpu().numpy()
        return action_np[0] if single else action_np

    @staticmethod
    def _format_scalar_output(value: torch.Tensor, single: bool):
        value_np = value.detach().cpu().numpy().reshape(-1)
        return float(value_np[0]) if single else value_np

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select an action given the current state.

        Args:
            state: Current observation.
            evaluate: If True, use deterministic action (no exploration).

        Returns:
            Action as numpy array.
        """
        raise NotImplementedError

    @abstractmethod
    def update_parameters(self, *args, **kwargs) -> Dict[str, float]:
        """
        Update network parameters.

        Returns:
            Dictionary of loss values for logging.
        """
        raise NotImplementedError

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get all saveable state dictionaries.

        Returns:
            Dict containing model and optimizer state dicts.
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any], evaluate: bool = False) -> None:
        """
        Load state dictionaries for resumption.

        Args:
            state_dict: Previously saved state dict.
            evaluate: If True, set networks to eval mode.
        """
        raise NotImplementedError

    def save_checkpoint(self, path: str, tag: str) -> None:
        """Save model checkpoint."""
        ckpt_path = os.path.join(path, f"{tag}.pt")
        os.makedirs(path, exist_ok=True)
        torch.save(self.get_state_dict(), ckpt_path)

    def load_checkpoint(self, path: str, tag: str = None,
                        evaluate: bool = False) -> None:
        """Load model checkpoint."""
        if tag is not None:
            ckpt_path = os.path.join(path, f"{tag}.pt")
        else:
            # path is the full checkpoint file
            ckpt_path = path
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(state_dict, evaluate=evaluate)
