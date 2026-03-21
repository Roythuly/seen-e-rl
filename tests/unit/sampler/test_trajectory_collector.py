from rl_training_infra.sampler.impl.trajectory import TrajectoryCollector


class _FakeTrajectoryEnv:
    def __init__(self):
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, seed=None):
        self.reset_calls += 1
        return {"agent": {"obs": 0}}, {"seed": seed}

    def step(self, action):
        self.step_calls += 1
        if self.step_calls == 1:
            return {"agent": {"obs": 1}}, 1.0, False, False, {"step": 1}
        return {"agent": {"obs": 2}}, 2.0, True, False, {"step": 2}


class _FakeTrajectoryActor:
    def act(self, observation_batch, policy_version=None):
        obs_value = observation_batch["observations"]["agent"]["obs"]
        return {
            "action": obs_value + 10,
            "log_prob": -0.5 * obs_value,
            "value_estimate": obs_value + 0.25,
            "policy_version": 7,
        }


def test_trajectory_collector_builds_ppo_style_batch():
    collector = TrajectoryCollector(_FakeTrajectoryEnv(), _FakeTrajectoryActor())

    batch = collector.collect(2)

    assert batch["observations"] == [{"agent": {"obs": 0}}, {"agent": {"obs": 1}}]
    assert batch["actions"] == [10, 11]
    assert batch["rewards"] == [1.0, 2.0]
    assert batch["terminated"] == [False, True]
    assert batch["truncated"] == [False, False]
    assert batch["log_probs"] == [0.0, -0.5]
    assert batch["value_estimates"] == [0.25, 1.25]
    assert batch["policy_version"] == 7
