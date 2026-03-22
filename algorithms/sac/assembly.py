from __future__ import annotations

from algorithms.template import build_algorithm, build_artifacts_root, build_evaluator, build_info_hub
from rl_training_infra.model import ModelFactory
from rl_training_infra.sampler import GymEnvFactory, ReplayCollector, TorchActorHandle
from rl_training_infra.trainer import OffPolicyRuntimeLoop, ReplayBuffer

from .learner import SACLearner

from .defaults import build_default_sac_experiment


def build_sac_algorithm(config: dict | None = None, module_registry: dict | None = None):
    experiment = config or build_default_sac_experiment()
    artifacts_root = build_artifacts_root(experiment)

    model_spec = dict(experiment["model"])
    model_spec["algorithm"] = "sac"
    model = ModelFactory.build(model_spec, experiment["backend"])
    actor_handle = TorchActorHandle(model)
    env = GymEnvFactory().create(experiment["env"], seed=experiment.get("seed"))
    collector = ReplayCollector(env, actor_handle)
    replay_buffer = ReplayBuffer(
        capacity=int(experiment["buffer"]["capacity"]),
        batch_size=int(experiment["buffer"]["batch_size"]),
        sampling_mode=experiment["buffer"]["sampling_mode"],
        seed=experiment.get("seed"),
    )
    learner_config = dict(experiment["trainer"]["runtime"]["update_schedule"])
    learner_config["batch_size"] = int(experiment["buffer"]["batch_size"])
    learner = SACLearner(
        model=model,
        run_id=experiment["run_name"],
        backend=experiment["backend"]["name"],
        algorithm="sac",
        artifacts_dir=artifacts_root,
        config=learner_config,
    )
    runtime_loop = OffPolicyRuntimeLoop(
        replay_buffer=replay_buffer,
        collector=collector,
        learner=learner,
        info=build_info_hub(experiment),
    )
    registry = module_registry or {
        "model_factory": ModelFactory,
        "env_factory": GymEnvFactory,
        "actor_handle": actor_handle,
        "collector": collector,
        "replay_buffer": replay_buffer,
        "learner": learner,
        "runtime_loop": runtime_loop,
    }
    return build_algorithm(experiment, registry, runtime_loop=runtime_loop, evaluator=build_evaluator(experiment))
