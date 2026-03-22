from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..templates import RuntimeLoopTemplate


def _should_publish(publish_schedule: dict[str, Any], update_count: int, update_result: dict[str, Any]) -> bool:
    strategy = publish_schedule.get("strategy", "manual")
    if strategy == "after_update":
        return True
    if strategy == "every_n_updates":
        every_n_updates = int(publish_schedule.get("every_n_updates", 1))
        return every_n_updates > 0 and update_count % every_n_updates == 0
    if strategy == "on_actor_update":
        return bool(update_result.get("metrics", {}).get("actor_updated"))
    if strategy == "manual":
        return False
    raise ValueError(f"unsupported publish strategy: {strategy}")


def _should_checkpoint(checkpoint_spec: dict[str, Any], publish_happened: bool, update_count: int) -> bool:
    if not checkpoint_spec.get("enabled", False):
        return False
    if checkpoint_spec.get("on_publish_only", False):
        return publish_happened
    every_n_updates = int(checkpoint_spec.get("every_n_updates", 1))
    return every_n_updates > 0 and update_count % every_n_updates == 0


@dataclass(slots=True)
class OnPolicyRuntimeLoop(RuntimeLoopTemplate):
    collector: Any
    learner: Any
    info: Any | None = None

    def run(
        self,
        runtime_spec: dict[str, Any],
        actor_handle: Any | None = None,
        sampler: Any | None = None,
        learner: Any | None = None,
        info: Any | None = None,
    ) -> dict[str, Any]:
        del actor_handle
        collector = sampler or self.collector
        learner = learner or self.learner
        info_hub = info or self.info

        collection_schedule = runtime_spec["collection_schedule"]
        publish_schedule = runtime_spec["publish_schedule"]
        checkpoint_spec = runtime_spec.get("checkpoint", {})
        max_env_steps = int(runtime_spec.get("max_env_steps", collection_schedule["amount"]))

        env_steps = 0
        update_count = 0
        checkpoints: list[dict[str, Any]] = []
        published_snapshot = None
        last_update_result = None
        while env_steps < max_env_steps:
            collect_amount = min(int(collection_schedule["amount"]), max_env_steps - env_steps)
            batch = collector.collect(collect_amount)
            env_steps += collect_amount
            last_update_result = learner.update(batch, {"env_steps": env_steps})
            update_count += 1
            if info_hub is not None:
                info_hub.record_training(
                    policy_version=last_update_result["policy_version"],
                    env_steps=last_update_result["env_steps"],
                    grad_steps=last_update_result["grad_steps"],
                    metrics=last_update_result["metrics"],
                )

            published = False
            if _should_publish(publish_schedule, update_count, last_update_result):
                published_snapshot = learner.publish_policy()
                published = True
            if _should_checkpoint(checkpoint_spec, published, update_count):
                checkpoint = learner.save_checkpoint()
                checkpoints.append(checkpoint)
                if info_hub is not None:
                    info_hub.record_checkpoint(
                        checkpoint_id=checkpoint["checkpoint_id"],
                        path=checkpoint["path"],
                        policy_version=checkpoint["policy_version"],
                        env_steps=last_update_result["env_steps"],
                        grad_steps=last_update_result["grad_steps"],
                        metrics=last_update_result["metrics"],
                    )

        return {
            "env_steps": env_steps,
            "updates": update_count,
            "published_policy": published_snapshot,
            "checkpoints": checkpoints,
            "last_update": last_update_result,
        }


@dataclass(slots=True)
class OffPolicyRuntimeLoop(RuntimeLoopTemplate):
    replay_buffer: Any
    collector: Any
    learner: Any
    info: Any | None = None

    def run(
        self,
        runtime_spec: dict[str, Any],
        actor_handle: Any | None = None,
        sampler: Any | None = None,
        learner: Any | None = None,
        info: Any | None = None,
    ) -> dict[str, Any]:
        del actor_handle
        collector = sampler or self.collector
        learner = learner or self.learner
        info_hub = info or self.info

        collection_schedule = runtime_spec["collection_schedule"]
        update_schedule = runtime_spec["update_schedule"]
        publish_schedule = runtime_spec["publish_schedule"]
        checkpoint_spec = runtime_spec.get("checkpoint", {})

        amount = int(collection_schedule["amount"])
        warmup_env_steps = int(collection_schedule.get("warmup_env_steps", 0))
        min_ready_size = int(update_schedule.get("min_ready_size", 0))
        updates_per_trigger = int(update_schedule.get("updates_per_trigger", 1))
        batch_size = int(update_schedule.get("batch_size", getattr(self.replay_buffer, "batch_size", 1)))
        max_env_steps = int(runtime_spec.get("max_env_steps", amount))

        env_steps = 0
        update_count = 0
        checkpoints: list[dict[str, Any]] = []
        published_snapshot = None
        last_update_result = None

        while env_steps < max_env_steps:
            collect_amount = min(amount, max_env_steps - env_steps)
            collected_records = collector.collect(collect_amount)
            records = collected_records if isinstance(collected_records, list) else [collected_records]
            for record in records:
                self.replay_buffer.write(record)
                env_steps += 1

            if env_steps <= warmup_env_steps or len(self.replay_buffer) < max(min_ready_size, batch_size):
                continue

            for _ in range(updates_per_trigger):
                batch = self.replay_buffer.sample({"batch_size": batch_size})
                last_update_result = learner.update(batch, {"env_steps": env_steps})
                update_count += 1
                if info_hub is not None:
                    info_hub.record_training(
                        policy_version=last_update_result["policy_version"],
                        env_steps=last_update_result["env_steps"],
                        grad_steps=last_update_result["grad_steps"],
                        metrics=last_update_result["metrics"],
                    )

                published = False
                if _should_publish(publish_schedule, update_count, last_update_result):
                    published_snapshot = learner.publish_policy()
                    published = True
                if _should_checkpoint(checkpoint_spec, published, update_count):
                    checkpoint = learner.save_checkpoint()
                    checkpoints.append(checkpoint)
                    if info_hub is not None:
                        info_hub.record_checkpoint(
                            checkpoint_id=checkpoint["checkpoint_id"],
                            path=checkpoint["path"],
                            policy_version=checkpoint["policy_version"],
                            env_steps=last_update_result["env_steps"],
                            grad_steps=last_update_result["grad_steps"],
                            metrics=last_update_result["metrics"],
                        )

        return {
            "env_steps": env_steps,
            "updates": update_count,
            "published_policy": published_snapshot,
            "checkpoints": checkpoints,
            "last_update": last_update_result,
        }
