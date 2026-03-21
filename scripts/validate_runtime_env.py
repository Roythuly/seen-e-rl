from __future__ import annotations

import sys


def main() -> int:
    try:
        import gymnasium as gym
        import mujoco  # noqa: F401
        import torch
    except Exception as exc:  # pragma: no cover - import failures are the behavior under test
        print(f"runtime env validation failed: import error: {exc}")
        return 1

    try:
        env = gym.make("Humanoid-v5")
        observations, _ = env.reset(seed=1)
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        env.close()
    except Exception as exc:  # pragma: no cover - runtime failures are the behavior under test
        print(f"runtime env validation failed: env bootstrap error: {exc}")
        return 1

    print("runtime env validation passed")
    print(f"torch={torch.__version__}")
    print(f"obs_shape={getattr(observations, 'shape', None)}")
    print(f"terminated={terminated} truncated={truncated}")
    print(f"info_keys={sorted(info.keys())[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
