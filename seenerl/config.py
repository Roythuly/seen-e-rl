"""
YAML-based configuration management.

Supports:
  - Base config inheritance via `_base_` key
  - CLI argument overrides
  - Nested dict access via dot notation
"""

import argparse
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence

import yaml


class Config(dict):
    """Dictionary-like config with attribute access and dot-notation support."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, Config):
                value = Config(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo: dict) -> "Config":
        return Config(deepcopy(dict(self), memo))

    def __repr__(self) -> str:
        items = []
        for k, v in sorted(self.items()):
            if not k.startswith("_"):
                items.append(f"  {k}: {repr(v)}")
        return "Config(\n" + "\n".join(items) + "\n)"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml_with_base(yaml_path: str) -> dict:
    """Load a YAML file, resolving `_base_` inheritance chain."""
    yaml_path = os.path.abspath(yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.pop("_base_", None)
    if base_path is not None:
        # Resolve relative path from the directory of the current YAML
        base_dir = os.path.dirname(yaml_path)
        base_full_path = os.path.join(base_dir, base_path)
        base_cfg = _load_yaml_with_base(base_full_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def _parse_override_value(value_str: str) -> Any:
    """Parse a CLI override string into the appropriate Python type."""
    # Boolean
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null" or value_str.lower() == "none":
        return None
    # Int
    try:
        return int(value_str)
    except ValueError:
        pass
    # Float
    try:
        return float(value_str)
    except ValueError:
        pass
    # String
    return value_str


def _apply_overrides(cfg: dict, overrides: Sequence[str]) -> dict:
    """Apply CLI overrides in the form --key value or --nested.key value."""
    i = 0
    while i < len(overrides):
        arg = overrides[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
                value = _parse_override_value(overrides[i + 1])
                i += 2
            else:
                value = True
                i += 1
            # Support nested keys via dot notation: --checkpoint.save_buffer false
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
            if key == "env_name":
                cfg.setdefault("env", {})
                if not isinstance(cfg["env"], dict):
                    cfg["env"] = {}
                cfg["env"]["id"] = value
            elif key == "env_backend":
                cfg.setdefault("env", {})
                if not isinstance(cfg["env"], dict):
                    cfg["env"] = {}
                cfg["env"]["backend"] = value
        else:
            i += 1
    return cfg


def _default_isaaclab_task_imports(env_id: str) -> list[str]:
    """Infer Isaac Lab task imports for blacklisted task packages."""
    if "Locomanipulation" in env_id or "FixedBaseUpperBodyIK" in env_id:
        return ["isaaclab_tasks.manager_based.locomanipulation.pick_place"]
    if any(token in env_id for token in ("PickPlace", "NutPour", "ExhaustPipe")):
        return ["isaaclab_tasks.manager_based.manipulation.pick_place"]
    return []


def _normalize_model_config(cfg: dict) -> None:
    """Normalize optional nested model configuration."""
    model_cfg = deepcopy(cfg.get("model", {}))
    for key in ("actor", "critic", "value"):
        section = deepcopy(model_cfg.get(key, {}))
        if not isinstance(section, dict):
            section = {}
        if "hidden_size" in section and "hidden_dim" not in section:
            section["hidden_dim"] = section.pop("hidden_size")
        kwargs = deepcopy(section.get("kwargs", {}))
        if not isinstance(kwargs, dict):
            kwargs = {}
        section["kwargs"] = kwargs
        if section:
            model_cfg[key] = section
    cfg["model"] = model_cfg


def _normalize_env_config(cfg: dict) -> None:
    """Normalize environment configuration while keeping legacy keys working."""
    env_cfg = deepcopy(cfg.get("env", {}))
    if not isinstance(env_cfg, dict):
        env_cfg = {}

    legacy_env_name = cfg.get("env_name")
    legacy_backend = cfg.get("env_backend")

    env_cfg.setdefault("backend", legacy_backend or "gymnasium")
    env_cfg.setdefault("id", legacy_env_name)
    env_cfg.setdefault("num_envs", 1)

    kwargs = deepcopy(env_cfg.get("kwargs", {}))
    if not isinstance(kwargs, dict):
        kwargs = {}
    env_cfg["kwargs"] = kwargs

    isaaclab_cfg = deepcopy(env_cfg.get("isaaclab", {}))
    if not isinstance(isaaclab_cfg, dict):
        isaaclab_cfg = {}
    isaaclab_cfg.setdefault("headless", True)
    isaaclab_cfg.setdefault("use_fabric", None)

    task_imports = isaaclab_cfg.get("task_imports")
    if task_imports is None:
        task_imports = []
    elif isinstance(task_imports, str):
        task_imports = [task_imports]
    else:
        task_imports = list(task_imports)

    if env_cfg["backend"] == "isaaclab" and not task_imports and env_cfg["id"]:
        task_imports = _default_isaaclab_task_imports(env_cfg["id"])

    isaaclab_cfg["task_imports"] = task_imports
    env_cfg["isaaclab"] = isaaclab_cfg

    cfg["env"] = env_cfg
    if env_cfg["id"] is not None:
        cfg["env_name"] = env_cfg["id"]


def _normalize_config(cfg: dict) -> dict:
    """Apply compatibility normalization for config consumers."""
    _normalize_env_config(cfg)
    _normalize_model_config(cfg)
    return cfg


def load_config(
    yaml_path: str,
    cli_args: Optional[Sequence[str]] = None,
) -> Config:
    """
    Load configuration from a YAML file with optional CLI overrides.

    Args:
        yaml_path: Path to the YAML config file.
        cli_args: Optional list of CLI arguments for overrides.

    Returns:
        Config object with all settings merged.

    Example:
        config = load_config("configs/sac.yaml", ["--env_name", "Humanoid-v5"])
    """
    cfg = _load_yaml_with_base(yaml_path)

    if cli_args:
        cfg = _apply_overrides(cfg, list(cli_args))

    cfg = _normalize_config(cfg)

    return Config(cfg)


def parse_args_and_load_config() -> Config:
    """
    Parse command line arguments and load config.

    Expected usage:
        python train.py --config configs/sac.yaml --env_name Humanoid-v5 --seed 42
        python train.py --config configs/sac.yaml --resume path/to/checkpoint.pt
    """
    parser = argparse.ArgumentParser(description="seenerl training", add_help=False)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming")

    known_args, remaining = parser.parse_known_args()
    config = load_config(known_args.config, remaining)

    if known_args.resume:
        config["resume"] = known_args.resume

    return config
