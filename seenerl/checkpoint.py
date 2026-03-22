"""
Checkpoint manager with multiple save strategies.

Strategies (composable):
  - latest: Always save the latest checkpoint (overwrites).
  - best: Save when evaluation reward is the best so far.
  - interval_steps: Save every N training steps.
  - interval_epochs: Save every N epochs/rollout cycles.

Saves model parameters, optimizer states, buffer data (optional),
and training progress for full resumption.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import torch

from seenerl.algorithms.base import BaseAlgorithm

logger = logging.getLogger("seenerl")


class CheckpointManager:
    """
    Multi-strategy checkpoint save/load manager.

    Checkpoint contents:
        - agent state dict (model + optimizer params)
        - training state (step, epoch, best_reward)
        - buffer path (saved separately for large buffers)
    """

    def __init__(
        self,
        save_dir: str,
        strategies: List[str] = None,
        interval_steps: int = None,
        interval_epochs: int = None,
        save_buffer: bool = True,
    ):
        """
        Args:
            save_dir: Base directory for checkpoints.
            strategies: List of strategies: "latest", "best", "interval_steps", "interval_epochs".
            interval_steps: Save every N steps (if strategy includes "interval_steps").
            interval_epochs: Save every N epochs (if strategy includes "interval_epochs").
            save_buffer: Whether to save replay buffer with checkpoints.
        """
        self.save_dir = save_dir
        self.strategies = strategies or ["latest", "best"]
        self.interval_steps = interval_steps
        self.interval_epochs = interval_epochs
        self.save_buffer = save_buffer
        self.best_reward = -float("inf")

        os.makedirs(save_dir, exist_ok=True)

    def should_save(
        self, step: int, epoch: int, eval_reward: float = None
    ) -> Dict[str, bool]:
        """
        Determine which checkpoint tags should be saved.

        Returns:
            Dict mapping tag name -> should_save boolean.
        """
        result = {}

        if "latest" in self.strategies:
            result["latest"] = True

        if "best" in self.strategies and eval_reward is not None:
            if eval_reward > self.best_reward:
                result["best"] = True
            else:
                result["best"] = False

        if "interval_steps" in self.strategies and self.interval_steps:
            result[f"step_{step}"] = (step % self.interval_steps == 0)

        if "interval_epochs" in self.strategies and self.interval_epochs:
            result[f"epoch_{epoch}"] = (epoch % self.interval_epochs == 0)

        return result

    def save(
        self,
        agent: BaseAlgorithm,
        step: int,
        epoch: int,
        eval_reward: float = None,
        buffer=None,
        tag: str = "latest",
    ) -> str:
        """
        Save a checkpoint with the given tag.

        Args:
            agent: The RL algorithm to save.
            step: Current training step.
            epoch: Current epoch/episode number.
            eval_reward: Latest evaluation reward.
            buffer: Optional replay buffer to save.
            tag: Checkpoint filename tag.

        Returns:
            Path to saved checkpoint.
        """
        ckpt = {
            "agent_state_dict": agent.get_state_dict(),
            "step": step,
            "epoch": epoch,
            "best_reward": self.best_reward,
        }

        if eval_reward is not None:
            ckpt["eval_reward"] = eval_reward
            if eval_reward > self.best_reward:
                self.best_reward = eval_reward

        ckpt_path = os.path.join(self.save_dir, f"{tag}.pt")
        torch.save(ckpt, ckpt_path)
        logger.info(f"[INFO]  Checkpoint saved: {ckpt_path}")

        # Save buffer separately (can be large)
        if self.save_buffer and buffer is not None and hasattr(buffer, "save"):
            buffer_path = os.path.join(self.save_dir, f"{tag}_buffer.pkl")
            buffer.save(buffer_path)

        return ckpt_path

    def save_if_needed(
        self,
        agent: BaseAlgorithm,
        step: int,
        epoch: int,
        eval_reward: float = None,
        buffer=None,
    ) -> None:
        """Check all strategies and save accordingly."""
        tags = self.should_save(step, epoch, eval_reward)
        for tag, should in tags.items():
            if should:
                self.save(agent, step, epoch, eval_reward, buffer, tag)

    @staticmethod
    def load(
        path: str,
        agent: BaseAlgorithm,
        buffer=None,
        evaluate: bool = False,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore agent (and optionally buffer).

        Args:
            path: Path to checkpoint .pt file.
            agent: Agent to restore state into.
            buffer: Optional buffer to restore.
            evaluate: If True, set agent to eval mode.

        Returns:
            Dict with training state: step, epoch, best_reward.
        """
        ckpt = torch.load(path, map_location=agent.device)
        agent.load_state_dict(ckpt["agent_state_dict"], evaluate=evaluate)

        # Try to load buffer
        if buffer is not None and hasattr(buffer, "load"):
            buffer_path = path.replace(".pt", "_buffer.pkl")
            if os.path.exists(buffer_path):
                buffer.load(buffer_path)
                logger.info(f"[INFO]  Buffer loaded from: {buffer_path}")

        return {
            "step": ckpt.get("step", 0),
            "epoch": ckpt.get("epoch", 0),
            "best_reward": ckpt.get("best_reward", -float("inf")),
        }
