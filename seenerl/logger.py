"""
Unified logging module with TensorBoard and wandb support.

Provides standardized output with [TRAIN], [EVAL], [INFO] prefixes
and consistent formatting across all training stages.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

# Optional imports
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None


class TrainingLogger:
    """
    Unified logger for RL training.

    Supports:
        - Python logging to console with formatted output
        - TensorBoard scalar logging
        - wandb scalar / config logging
        - Standardized [TRAIN] / [EVAL] / [INFO] prefixes
    """

    def __init__(self, log_dir: str, config: dict = None,
                 use_tensorboard: bool = True, use_wandb: bool = False,
                 wandb_project: str = "seen-e-rl", wandb_entity: str = None,
                 wandb_name: str = None):
        """
        Args:
            log_dir: Directory for log files and TensorBoard events.
            config: Configuration dict to log as hyperparameters.
            use_tensorboard: Enable TensorBoard logging.
            use_wandb: Enable wandb logging.
            wandb_project: wandb project name.
            wandb_entity: wandb entity (username/team).
            wandb_name: wandb run name.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Console logger
        self._logger = logging.getLogger("seenerl")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # TensorBoard
        self._tb_writer = None
        if use_tensorboard:
            if SummaryWriter is None:
                self._logger.warning("[INFO] tensorboard not installed, skipping")
            else:
                self._tb_writer = SummaryWriter(log_dir)

        # wandb
        self._use_wandb = use_wandb
        if use_wandb:
            if wandb is None:
                self._logger.warning("[INFO] wandb not installed, skipping")
                self._use_wandb = False
            else:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_name,
                    config=config or {},
                    dir=log_dir,
                    reinit=True,
                )

        # Log config as hyperparameters
        if config and self._tb_writer:
            # Flatten nested config for TensorBoard hparams
            flat_config = self._flatten_dict(config)
            self._tb_writer.add_text("config", str(config))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        if self._tb_writer:
            self._tb_writer.add_scalar(tag, value, step)
        if self._use_wandb:
            wandb.log({tag: value}, step=step)

    def log_dict(self, tag_dict: Dict[str, float], step: int,
                 prefix: str = "") -> None:
        """Log multiple scalars at once."""
        for tag, value in tag_dict.items():
            full_tag = f"{prefix}/{tag}" if prefix else tag
            self.log_scalar(full_tag, value, step)

    def log_train(self, step: int, episode: int, episode_steps: int,
                  reward: float, **kwargs) -> None:
        """Log training metrics with [TRAIN] prefix."""
        msg = (
            f"[TRAIN] step: {step:>8d} | episode: {episode:>5d} | "
            f"ep_steps: {episode_steps:>4d} | reward: {reward:>10.2f}"
        )
        for k, v in kwargs.items():
            if isinstance(v, float):
                msg += f" | {k}: {v:.4f}"
            else:
                msg += f" | {k}: {v}"
        self._logger.info(msg)

    def log_eval(self, step: int, avg_reward: float, std_reward: float = 0.0,
                 num_episodes: int = 0, **kwargs) -> None:
        """Log evaluation metrics with [EVAL] prefix."""
        msg = (
            f"[EVAL]  step: {step:>8d} | avg_reward: {avg_reward:>10.2f} | "
            f"std: {std_reward:>8.2f} | episodes: {num_episodes}"
        )
        for k, v in kwargs.items():
            if isinstance(v, float):
                msg += f" | {k}: {v:.4f}"
            else:
                msg += f" | {k}: {v}"
        self._logger.info(msg)

    def log_info(self, message: str) -> None:
        """Log general information with [INFO] prefix."""
        self._logger.info(f"[INFO]  {message}")

    def close(self) -> None:
        """Close all logging backends."""
        if self._tb_writer:
            self._tb_writer.close()
        if self._use_wandb and wandb is not None:
            wandb.finish()

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten a nested dict for TensorBoard hparams."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(TrainingLogger._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
