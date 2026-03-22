from __future__ import annotations

from algorithms.template import build_algorithm, build_artifacts_root, build_evaluator, build_info_hub
from rl_training_infra.model import ModelFactory
from rl_training_infra.sampler import GymEnvFactory, TorchActorHandle, TrajectoryCollector
from rl_training_infra.trainer import OnPolicyRuntimeLoop

from .learner import PPOLearner

from .defaults import build_default_ppo_experiment


def build_ppo_algorithm(config: dict | None = None, module_registry: dict | None = None):
    experiment = config or build_default_ppo_experiment()
    artifacts_root = build_artifacts_root(experiment)

    model_spec = dict(experiment["model"])
    model_spec["algorithm"] = "ppo"
    model = ModelFactory.build(model_spec, experiment["backend"])
    actor_handle = TorchActorHandle(model)
    env = GymEnvFactory().create(experiment["env"], seed=experiment.get("seed"))
    collector = TrajectoryCollector(env, actor_handle)
    learner = PPOLearner(
        model=model,
        run_id=experiment["run_name"],
        backend=experiment["backend"]["name"],
        algorithm="ppo",
        artifacts_dir=artifacts_root,
        config=dict(experiment["trainer"]["runtime"]["update_schedule"]),
    )
    runtime_loop = OnPolicyRuntimeLoop(
        collector=collector,
        learner=learner,
        info=build_info_hub(experiment),
    )
    registry = module_registry or {
        "model_factory": ModelFactory,
        "env_factory": GymEnvFactory,
        "actor_handle": actor_handle,
        "collector": collector,
        "learner": learner,
        "runtime_loop": runtime_loop,
    }
    return build_algorithm(experiment, registry, runtime_loop=runtime_loop, evaluator=build_evaluator(experiment))
