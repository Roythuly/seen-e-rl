from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_training_infra.common import load_json, save_json
from rl_training_infra.evaluator import CheckpointSelector, Evaluator
from rl_training_infra.info import ConsoleMetricSink, InfoHub, MetricEventBuilderBase
from rl_training_infra.model import ModelFactory
from rl_training_infra.sampler import GymEnvFactory, TorchActorHandle


@dataclass(slots=True)
class AlgorithmAssemblyTemplate:
    algorithm_name: str
    config: dict[str, Any]
    module_registry: dict[str, Any]
    runtime_loop: Any | None = None
    evaluator: Any | None = None
    artifacts_root: Path | None = None

    def __post_init__(self) -> None:
        default_run_name = self.config.get("run_name") or self.algorithm_name or "template-run"
        root = self.config.get("artifacts", {}).get("root", f"artifacts/{default_run_name}")
        self.artifacts_root = Path(root) if self.artifacts_root is None else Path(self.artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict[str, Any]:
        if self.runtime_loop is None:
            raise RuntimeError("runtime_loop is not configured")
        runtime_result = self.runtime_loop.run(self.config["trainer"]["runtime"])
        state = {
            "algorithm": self.algorithm_name,
            "run_name": self.config["run_name"],
            "artifacts_root": str(self.artifacts_root),
            "checkpoints": runtime_result.get("checkpoints", []),
            "published_policy": runtime_result.get("published_policy"),
            "env_steps": runtime_result.get("env_steps", 0),
            "updates": runtime_result.get("updates", 0),
        }
        save_json(self.artifacts_root / "run_state.json", state)
        return runtime_result

    def evaluate(self, selector: str | None = None, policy_version: int | None = None) -> dict[str, Any]:
        if self.evaluator is None:
            raise RuntimeError("evaluator is not configured")

        run_state = load_json(self.artifacts_root / "run_state.json")
        checkpoints = run_state.get("checkpoints", [])
        checkpoint_selector = CheckpointSelector(checkpoints=checkpoints)
        selection = checkpoint_selector.select(selector or self.config.get("eval", {}).get("selector", "latest"), policy_version=policy_version)
        report = self.evaluator.evaluate(selection, self.config.get("eval", {}).get("seeds", [1]), self.config["env"])
        save_json(self.artifacts_root / f"eval_{selection['selector']}.json", report)
        return report


AlgorithmTemplate = AlgorithmAssemblyTemplate
AlgorithmConfigTemplate = dict[str, Any]


def build_default_model_spec() -> dict[str, Any]:
    return {
        "encoder": {"kind": "mlp", "hidden_sizes": [256, 256]},
        "actor_head": {"kind": "gaussian_policy"},
        "critic_head": {"kind": "twin_q"},
        "feature_sharing": {"actor_critic_encoder": False},
        "training_interface": {"forward_train": "structured"},
    }


def build_default_runtime_spec() -> dict[str, Any]:
    return {
        "collection_schedule": {
            "mode": "step",
            "unit": "env_step",
            "amount": 1,
            "freeze_policy_during_collection": False,
        },
        "update_schedule": {"trigger_unit": "env_step", "updates_per_trigger": 1},
        "publish_schedule": {"strategy": "every_n_updates", "every_n_updates": 1},
    }


def build_default_eval_spec() -> dict[str, Any]:
    return {"selector": "latest", "seeds": [1], "episodes_per_seed": 1}


def build_artifacts_root(config: dict[str, Any]) -> Path:
    return Path(config.get("artifacts", {}).get("root", f"artifacts/{config['run_name']}"))


def build_info_hub(config: dict[str, Any]) -> InfoHub:
    builder = MetricEventBuilderBase(
        run_id=config["run_name"],
        algorithm=config["algo"]["name"],
        backend=config["backend"]["name"],
        env_id=config["env"]["id"],
    )
    return InfoHub(builder=builder, sinks=[ConsoleMetricSink()])


def build_seed_runner(config: dict[str, Any]):
    env_factory = GymEnvFactory()
    env_spec = config["env"]
    backend = config["backend"]
    model_spec = dict(config["model"])
    model_spec["algorithm"] = config["algo"]["name"]
    episodes_per_seed = int(config.get("eval", {}).get("episodes_per_seed", 1))

    def seed_runner(checkpoint_manifest: dict[str, Any], seed: int, _: dict[str, Any]) -> dict[str, Any]:
        checkpoint_payload = torch.load(checkpoint_manifest["path"], map_location="cpu")
        model = ModelFactory.build(model_spec, backend)
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        runtime_state = checkpoint_payload.get("runtime_state", {})
        if "policy_version" in runtime_state:
            model.set_policy_version(int(runtime_state["policy_version"]))

        actor_handle = TorchActorHandle(model, deterministic=True)
        env = env_factory.create(env_spec, seed=seed)
        action_space = env.action_space
        episode_returns: list[float] = []
        for episode_offset in range(episodes_per_seed):
            observation, _ = env.reset(seed=seed + episode_offset)
            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action_output = actor_handle.act({"observations": observation})
                action = np.asarray(action_output["action"], dtype=getattr(action_space, "dtype", None))
                action = np.clip(action, action_space.low, action_space.high)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
            episode_returns.append(total_reward)
        env.close()
        return {
            "reward_mean": sum(episode_returns) / len(episode_returns),
            "episode_count": len(episode_returns),
        }

    return seed_runner


def build_evaluator(config: dict[str, Any]) -> Evaluator:
    return Evaluator(seed_runner=build_seed_runner(config))


def build_algorithm(
    config: dict[str, Any],
    module_registry: dict[str, Any],
    *,
    runtime_loop: Any | None = None,
    evaluator: Any | None = None,
) -> AlgorithmAssemblyTemplate:
    return AlgorithmAssemblyTemplate(
        algorithm_name=config.get("name") or config.get("algo", {}).get("name", "template"),
        config=config,
        module_registry=module_registry,
        runtime_loop=runtime_loop,
        evaluator=evaluator,
    )


__all__ = [
    "AlgorithmAssemblyTemplate",
    "AlgorithmConfigTemplate",
    "AlgorithmTemplate",
    "build_algorithm",
    "build_artifacts_root",
    "build_default_eval_spec",
    "build_default_model_spec",
    "build_default_runtime_spec",
    "build_evaluator",
    "build_info_hub",
    "build_seed_runner",
]
