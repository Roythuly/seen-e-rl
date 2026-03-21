from rl_training_infra.sampler.impl.replay import ReplayCollector


class _FakeReplayEnv:
    def __init__(self):
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, seed=None):
        self.reset_calls += 1
        return {"agent": {"obs": 0}}, {"seed": seed}

    def step(self, action):
        self.step_calls += 1
        if self.step_calls == 1:
            return {"agent": {"obs": 1}}, 1.5, False, False, {"step": 1}
        return {"agent": {"obs": 2}}, 2.5, True, False, {"step": 2}


class _FakeReplayActor:
    def act(self, observation_batch, policy_version=None):
        obs_value = observation_batch["observations"]["agent"]["obs"]
        return {
            "action": obs_value + 5,
            "policy_version": 11,
        }


def test_replay_collector_builds_sac_style_records():
    collector = ReplayCollector(_FakeReplayEnv(), _FakeReplayActor())

    records = collector.collect(2)

    assert records == [
        {
            "observations": {"agent": {"obs": 0}},
            "actions": 5,
            "rewards": 1.5,
            "next_observations": {"agent": {"obs": 1}},
            "terminated": False,
            "truncated": False,
            "policy_version": 11,
            "env_step": 1,
        },
        {
            "observations": {"agent": {"obs": 1}},
            "actions": 6,
            "rewards": 2.5,
            "next_observations": {"agent": {"obs": 2}},
            "terminated": True,
            "truncated": False,
            "policy_version": 11,
            "env_step": 2,
        },
    ]
