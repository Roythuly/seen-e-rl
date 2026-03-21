import pytest

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


class _NoSeedReplayEnv(_FakeReplayEnv):
    def reset(self):
        self.reset_calls += 1
        return {"agent": {"obs": 0}}, {}


class _EpisodeResetReplayEnv(_FakeReplayEnv):
    def step(self, action):
        self.step_calls += 1
        next_obs = {"agent": {"obs": self.step_calls}}
        return next_obs, float(self.step_calls), True, False, {"step": 1}


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


def test_replay_collector_handles_envs_without_seed_reset_parameter():
    collector = ReplayCollector(_NoSeedReplayEnv(), _FakeReplayActor())

    records = collector.collect(1, seed=123)

    assert records[0]["observations"] == {"agent": {"obs": 0}}


def test_replay_collector_rejects_zero_amount():
    collector = ReplayCollector(_FakeReplayEnv(), _FakeReplayActor())

    with pytest.raises(ValueError, match="amount must be positive"):
        collector.collect(0)


def test_replay_collector_keeps_monotonic_env_step_across_episode_rollover():
    collector = ReplayCollector(_EpisodeResetReplayEnv(), _FakeReplayActor())

    records = collector.collect(3)

    assert [record["env_step"] for record in records] == [1, 2, 3]
