from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..base import BaseCollector


class TrajectoryAssembler:
    def build(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("trajectory records cannot be empty")
        policy_versions = {record["policy_version"] for record in records}
        if len(policy_versions) != 1:
            raise ValueError("trajectory records must use a single policy_version")
        return {
            "observations": [deepcopy(record["observations"]) for record in records],
            "actions": [deepcopy(record["actions"]) for record in records],
            "rewards": [record["rewards"] for record in records],
            "terminated": [bool(record["terminated"]) for record in records],
            "truncated": [bool(record["truncated"]) for record in records],
            "log_probs": [record["log_probs"] for record in records],
            "value_estimates": [record["value_estimates"] for record in records],
            "policy_version": records[0]["policy_version"],
        }


class TrajectoryCollector(BaseCollector):
    required_action_fields = ("action", "log_prob", "value_estimate", "policy_version")

    def __init__(self, env: Any, actor: Any, adapter=None, assembler: TrajectoryAssembler | None = None) -> None:
        super().__init__(env, actor, adapter=adapter)
        self.assembler = assembler or TrajectoryAssembler()

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
            "terminated": terminated,
            "truncated": truncated,
            "log_probs": action_output["log_prob"],
            "value_estimates": action_output["value_estimate"],
            "policy_version": action_output["policy_version"],
        }

    def collect(self, amount: int, seed: int | None = None) -> dict[str, Any]:
        return self.assembler.build(self.collect_step_records(amount, seed=seed))
