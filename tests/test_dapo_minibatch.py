from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from train.dapo_data import batch_schedule, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["dapo_minibatch.yaml", "dapo_llama32_3b_minibatch.yaml", "dapo_qwen25_3b.yaml"])
def test_minibatch_schedule_and_smoke(filename, tmp_path):
    config = load_config(ROOT / "configs" / filename)
    smoke = load_config(ROOT / "configs" / filename, smoke=True)
    assert config["algorithm"]["num_iterations"] == smoke["algorithm"]["num_iterations"] == 1
    assert batch_schedule(config) == {"rollout_batch_size": 128, "mini_batch_size": 64,
                                      "minibatches_per_rollout": 2, "updates_per_rollout": 2}
    assert batch_schedule(smoke) == {"rollout_batch_size": 16, "mini_batch_size": 8,
                                     "minibatches_per_rollout": 2, "updates_per_rollout": 2}
    assert config["training"]["max_steps"] == 300 and config["training"]["save_steps"] == 20
    assert smoke["training"]["max_steps"] == smoke["training"]["save_steps"] == 2
    for section, key, value in [("algorithm", "mini_batch_size", 0),
                                ("algorithm", "mini_batch_size", True),
                                ("algorithm", "mini_batch_size", 63),
                                ("algorithm", "mini_batch_size", 256),
                                ("training", "max_steps", 301), ("training", "save_steps", 21)]:
        bad = deepcopy(config)
        bad[section][key] = value
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError, match="mini_batch_size|rollout cycle"):
            load_config(path)


def test_both_models_use_same_minibatch_training_settings():
    exaone = load_config(ROOT / "configs/dapo_minibatch.yaml")
    llama = load_config(ROOT / "configs/dapo_llama32_3b_minibatch.yaml")
    for section in ["algorithm", "training", "smoke", "data", "rollout"]:
        assert exaone[section] == llama[section]
