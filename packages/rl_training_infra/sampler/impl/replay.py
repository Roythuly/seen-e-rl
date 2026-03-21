from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..base import BaseCollector


class ReplayAssembler:
    def build(self, record: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(record)


class ReplayCollector(BaseCollector):
    required_action_fields = ("action", "policy_version")

    def __init__(self, env: Any, actor: Any, adapter=None, assembler: ReplayAssembler | None = None) -> None:
        super().__init__(env, actor, adapter=adapter)
        self.assembler = assembler or ReplayAssembler()

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
        return {
            "observations": deepcopy(observation),
            "actions": deepcopy(action_output["action"]),
            "rewards": reward,
            "next_observations": deepcopy(next_observation),
            "terminated": terminated,
            "truncated": truncated,
            "policy_version": action_output["policy_version"],
            "env_step": env_step,
        }

    def collect(self, amount: int, seed: int | None = None) -> list[dict[str, Any]]:
        return [self.assembler.build(record) for record in self.collect_step_records(amount, seed=seed)]
