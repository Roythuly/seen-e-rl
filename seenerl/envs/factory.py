"""Environment factory and batched environment adapters."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from seenerl.envs.runtime import (
    ensure_isaaclab_app,
    maybe_patch_pink_configuration_limit,
    maybe_patch_usd_to_urdf_cache,
    release_isaaclab_app,
)


def _ensure_float32_box(space: gym.Space, clip: float = 1.0) -> gym.spaces.Box:
    """Return a finite Box action space for policy scaling and env clipping."""
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"Unsupported action space type: {type(space)!r}")

    low = np.asarray(space.low, dtype=np.float32)
    high = np.asarray(space.high, dtype=np.float32)
    if np.isfinite(low).all() and np.isfinite(high).all():
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    bounded = np.full(space.shape, clip, dtype=np.float32)
    return gym.spaces.Box(low=-bounded, high=bounded, dtype=np.float32)


def _flatten_batch(space: gym.Space, obs: Any) -> np.ndarray:
    """Flatten a batch of observations into a 2-D float32 numpy array."""
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    obs_array = np.asarray(obs, dtype=np.float32)
    single_shape = getattr(space, "shape", None)
    if single_shape is None:
        raise TypeError(f"Unsupported observation space type: {type(space)!r}")
    if obs_array.shape == single_shape:
        obs_array = obs_array.reshape(1, -1)
    else:
        obs_array = obs_array.reshape(obs_array.shape[0], -1)
    return obs_array


def _extract_policy_obs(obs: Any) -> Any:
    """Select the policy observation from a raw env observation."""
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        if len(obs) == 1:
            return next(iter(obs.values()))
        raise KeyError("Expected a 'policy' observation entry in the environment output.")
    return obs


def _clip_actions(actions: np.ndarray, action_space: gym.spaces.Box) -> np.ndarray:
    return np.clip(actions, action_space.low, action_space.high).astype(np.float32)


def _sample_batched_action_space(action_space: gym.spaces.Box, num_envs: int) -> np.ndarray:
    return np.stack([action_space.sample() for _ in range(num_envs)], axis=0).astype(np.float32)


class SingleGymEnv:
    """Wrap a single Gymnasium env with a batched interface."""

    def __init__(self, env_id: str, env_kwargs: Dict[str, Any] | None = None,
                 render_mode: str | None = None):
        self.env = gym.make(env_id, render_mode=render_mode, **(env_kwargs or {}))
        self.num_envs = 1
        self.observation_space = gym.spaces.flatten_space(self.env.observation_space)
        self.action_space = _ensure_float32_box(self.env.action_space)
        self.device = torch.device("cpu")

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return _flatten_batch(self.env.observation_space, obs), info

    def step(self, actions: np.ndarray):
        clipped_actions = _clip_actions(np.asarray(actions, dtype=np.float32), self.action_space)
        obs, reward, terminated, truncated, info = self.env.step(clipped_actions[0])
        next_obs = _flatten_batch(self.env.observation_space, obs)

        if terminated or truncated:
            final_obs = next_obs.copy()
            reset_obs, reset_info = self.env.reset()
            next_obs = _flatten_batch(self.env.observation_space, reset_obs)
            info = dict(info)
            info["final_observation"] = final_obs
            info["final_mask"] = np.array([True], dtype=np.bool_)
            info["reset_info"] = reset_info

        return (
            next_obs,
            np.asarray([reward], dtype=np.float32),
            np.asarray([terminated], dtype=np.bool_),
            np.asarray([truncated], dtype=np.bool_),
            info,
        )

    def close(self):
        self.env.close()

    def sample_random_actions(self) -> np.ndarray:
        return _sample_batched_action_space(self.action_space, self.num_envs)


class VectorGymEnv:
    """Wrap a vectorized Gymnasium env with a consistent batched interface."""

    def __init__(self, env_id: str, num_envs: int, env_kwargs: Dict[str, Any] | None = None,
                 render_mode: str | None = None):
        env_kwargs = deepcopy(env_kwargs or {})

        def make_env():
            return gym.make(env_id, render_mode=render_mode, **env_kwargs)

        self.env = SyncVectorEnv(
            [make_env for _ in range(num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        self.num_envs = num_envs
        self.observation_space = gym.spaces.flatten_space(self.env.single_observation_space)
        self.action_space = _ensure_float32_box(self.env.single_action_space)
        self.device = torch.device("cpu")

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return _flatten_batch(self.env.single_observation_space, obs), info

    def step(self, actions: np.ndarray):
        clipped_actions = _clip_actions(np.asarray(actions, dtype=np.float32), self.action_space)
        obs, reward, terminated, truncated, info = self.env.step(clipped_actions)
        next_obs = _flatten_batch(self.env.single_observation_space, obs)

        final_mask = info.get("_final_obs")
        if final_mask is not None and np.any(final_mask):
            final_obs = np.zeros_like(next_obs)
            for index, is_final in enumerate(final_mask):
                if is_final:
                    final_obs[index] = _flatten_batch(
                        self.env.single_observation_space,
                        info["final_obs"][index],
                    )[0]
            info = dict(info)
            info["final_observation"] = final_obs
            info["final_mask"] = final_mask.astype(np.bool_)

        return (
            next_obs,
            np.asarray(reward, dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
            info,
        )

    def close(self):
        self.env.close()

    def sample_random_actions(self) -> np.ndarray:
        return _sample_batched_action_space(self.action_space, self.num_envs)


class IsaacLabEnv:
    """Wrap an Isaac Lab vector env with the same batched API as Gym."""

    def __init__(self, env_id: str, num_envs: int, env_cfg: Dict[str, Any],
                 render_mode: str | None = None):
        isaac_cfg = deepcopy(env_cfg.get("isaaclab", {}))
        env_kwargs = deepcopy(env_cfg.get("kwargs", {}))
        headless = False if render_mode == "human" else isaac_cfg.get("headless", True)
        enable_cameras = render_mode == "rgb_array"

        ensure_isaaclab_app(headless=headless, enable_cameras=enable_cameras)

        import gymnasium as gymnasium
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        import isaaclab_tasks  # noqa: F401

        task_imports = list(isaac_cfg.get("task_imports", []))
        for module_name in task_imports:
            importlib.import_module(module_name)
        maybe_patch_usd_to_urdf_cache(env_id, task_imports)
        maybe_patch_pink_configuration_limit(env_id, task_imports)

        parsed_cfg = parse_env_cfg(
            env_id,
            device=env_cfg.get("device", "cuda:0"),
            num_envs=num_envs,
            use_fabric=isaac_cfg.get("use_fabric"),
        )

        self.env = gymnasium.make(env_id, cfg=parsed_cfg, render_mode=render_mode)
        self.num_envs = num_envs
        single_obs_space = self.env.unwrapped.single_observation_space
        if isinstance(single_obs_space, gym.spaces.Dict):
            single_obs_space = single_obs_space["policy"]
        self.observation_space = gym.spaces.flatten_space(single_obs_space)
        self.action_space = _ensure_float32_box(self.env.unwrapped.single_action_space)
        self.device = self.env.unwrapped.device
        self._single_obs_space = single_obs_space

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return _flatten_batch(self._single_obs_space, _extract_policy_obs(obs)), info

    def step(self, actions: np.ndarray):
        clipped_actions = _clip_actions(np.asarray(actions, dtype=np.float32), self.action_space)
        action_tensor = torch.as_tensor(clipped_actions, dtype=torch.float32, device=self.device)
        obs, reward, terminated, truncated, info = self.env.step(action_tensor)
        return (
            _flatten_batch(self._single_obs_space, _extract_policy_obs(obs)),
            np.asarray(reward.detach().cpu().numpy(), dtype=np.float32),
            np.asarray(terminated.detach().cpu().numpy(), dtype=np.bool_),
            np.asarray(truncated.detach().cpu().numpy(), dtype=np.bool_),
            info,
        )

    def close(self):
        self.env.close()
        release_isaaclab_app()

    def sample_random_actions(self) -> np.ndarray:
        return _sample_batched_action_space(self.action_space, self.num_envs)


def create_env(config, num_envs: int | None = None, render_mode: str | None = None):
    """Create a batched environment adapter from config."""
    env_cfg = deepcopy(config.env)
    backend = env_cfg.get("backend", "gymnasium")
    env_id = env_cfg["id"]
    actual_num_envs = int(num_envs if num_envs is not None else env_cfg.get("num_envs", 1))
    env_cfg["device"] = config.device

    if backend == "gymnasium":
        if actual_num_envs == 1:
            return SingleGymEnv(env_id, env_cfg.get("kwargs"), render_mode)
        return VectorGymEnv(env_id, actual_num_envs, env_cfg.get("kwargs"), render_mode)

    if backend == "isaaclab":
        return IsaacLabEnv(env_id, actual_num_envs, env_cfg, render_mode)

    raise ValueError(f"Unknown environment backend: {backend}")
