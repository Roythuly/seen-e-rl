from pathlib import Path

import yaml


def test_humanoid_experiment_configs_exist_for_all_three_algorithms():
    expected = {
        "ppo_humanoid_v5.yaml": "ppo",
        "sac_humanoid_v5.yaml": "sac",
        "td3_humanoid_v5.yaml": "td3",
    }

    for filename, algo_name in expected.items():
        path = Path("configs/experiment") / filename
        assert path.exists(), filename
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["env"]["id"] == "Humanoid-v5"
        assert payload["algo"]["name"] == algo_name
