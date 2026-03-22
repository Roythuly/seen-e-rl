from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import inspect
from typing import Any

import numpy as np


def _copy_payload(value: Any) -> Any:
    return deepcopy(value)


def _copy_info(info: Any) -> dict[str, Any]:
    if info is None:
        return {}
    return dict(info)


def unpack_reset_result(result: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, tuple):
        if len(result) == 2:
            observation, info = result
            return _copy_payload(observation), _copy_info(info)
        if len(result) == 1:
            return _copy_payload(result[0]), {}
    return _copy_payload(result), {}


def unpack_step_result(result: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
    if not isinstance(result, tuple):
        raise TypeError("env.step() must return a tuple")
    if len(result) == 5:
        next_observation, reward, terminated, truncated, info = result
        return _copy_payload(next_observation), reward, bool(terminated), bool(truncated), _copy_info(info)
    if len(result) == 4:
        next_observation, reward, done, info = result
        return _copy_payload(next_observation), reward, bool(done), False, _copy_info(info)
    raise ValueError("env.step() must return 4 or 5 values")


class EnvAdapter:
    """Small helper that normalizes common env reset/step conventions."""

    def reset(self, env: Any, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        reset_fn = env.reset
        if seed is None:
            result = reset_fn()
            return unpack_reset_result(result)

        try:
            signature = inspect.signature(reset_fn)
        except (TypeError, ValueError):
            signature = None

        accepts_seed = False
        if signature is not None:
            accepts_seed = "seed" in signature.parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
            )

        if accepts_seed:
            result = reset_fn(seed=seed)
        else:
            result = reset_fn()
        return unpack_reset_result(result)

    def step(self, env: Any, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        action_space = None
        try:
            action_space = env.action_space
        except Exception:
            action_space = None

        if action_space is not None and hasattr(action_space, "low") and hasattr(action_space, "high"):
            action = np.asarray(action, dtype=getattr(action_space, "dtype", None))
            action = np.clip(action, action_space.low, action_space.high)
        result = env.step(action)
        return unpack_step_result(result)


class BaseCollector(ABC):
    required_action_fields: tuple[str, ...] = ("action", "policy_version")

    def __init__(self, env: Any, actor: Any, adapter: EnvAdapter | None = None) -> None:
        self.env = env
        self.actor = actor
        self.adapter = adapter or EnvAdapter()

    def collect_step_records(self, amount: int, seed: int | None = None) -> list[dict[str, Any]]:
        if amount <= 0:
            raise ValueError("amount must be positive")
        observation, _ = self.adapter.reset(self.env, seed=seed)
        records: list[dict[str, Any]] = []
        current_policy_version: int | None = None

        for env_step in range(1, amount + 1):
            action_output = self.actor.act({"observations": observation}, policy_version=current_policy_version)
            self._validate_action_output(action_output)
            current_policy_version = action_output["policy_version"]
            next_observation, reward, terminated, truncated, step_info = self.adapter.step(
                self.env, action_output["action"]
            )
            records.append(
                self.build_record(
                    observation=observation,
                    action_output=action_output,
                    next_observation=next_observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    step_info=step_info,
                    env_step=env_step,
                )
            )
            if terminated or truncated:
                if env_step < amount:
                    observation, _ = self.adapter.reset(self.env)
                continue
            observation = next_observation
        return records

    def _validate_action_output(self, action_output: dict[str, Any]) -> None:
        missing = [field for field in self.required_action_fields if field not in action_output]
        if missing:
            raise KeyError(f"action output missing required fields: {', '.join(missing)}")

    @abstractmethod
    def build_record(
        self,
        *,
        observation: Any,
        action_output: dict[str, Any],
        next_observation: Any,
        reward: Any,
        terminated: bool,
        truncated: bool,
        step_info: dict[str, Any],
        env_step: int,
    ) -> dict[str, Any]:
        raise NotImplementedError
