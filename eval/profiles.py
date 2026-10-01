"""Named evaluation profiles with isolated baseline/SFT comparison protocols."""

import json
import math
from copy import deepcopy
from pathlib import Path

import yaml

from common.io import digest, write_json
from eval.run_english_eval import ROOT, code_fingerprint

PROFILE_NAMES = ("greedy", "sample8")


def load_profile(name, config, execution):
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown evaluation profile: {name}")
    profile = yaml.safe_load((ROOT / "configs/eval_profiles.yaml").read_text())[name]
    if set(profile) != {"temperature", "top_p", "pass_k", "samples", "execution"}:
        raise ValueError("Unexpected profile settings")
    config, execution = deepcopy(config), deepcopy(execution)
    if set(profile["samples"]) != set(config["datasets"]):
        raise ValueError("Profile must specify samples for every benchmark")
    for key in ["temperature", "top_p", "pass_k"]:
        config[key] = profile[key]
    for dataset, n in profile["samples"].items():
        config["datasets"][dataset]["n"] = n
    execution.update(profile["execution"])
    validate_profile_config(config)
    return config, execution


def validate_profile_config(config):
    if config["engine"] != "vllm" or config["tensor_parallel_size"] != 1:
        raise ValueError("Evaluation requires vLLM and one GPU")
    if not 0 < config["max_new_tokens"] < config["max_model_len"]:
        raise ValueError("Invalid generation/context budget")
    if not math.isfinite(config["temperature"]) or config["temperature"] < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    if not 0 < config["top_p"] <= 1:
        raise ValueError("Invalid top_p")
    ns = [spec["n"] for spec in config["datasets"].values()]
    if not ns or any(type(n) is not int or n < 1 for n in ns):
        raise ValueError("Sample counts must be positive integers")
    if config["temperature"] == 0 and any(n != 1 for n in ns):
        raise ValueError("Greedy evaluation requires one answer per problem")
    ks = config["pass_k"]
    if not ks or any(type(k) is not int or not 1 <= k <= max(ns) for k in ks):
        raise ValueError("Invalid pass@k values")


def profile_root(output_root, name):
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown evaluation profile: {name}")
    return Path(output_root) / "profiles" / name


def bind_profile_protocol(root, name, config, execution, suite, stage, selected, smoke,
                          implementation_digest, *, create=True):
    protocol = {
        "profile": name, "config": config, "execution": execution, "suite": suite,
        "code_digest": code_fingerprint(), "runner_digest": implementation_digest,
        "mode": "smoke" if smoke else "full", "selected_ids": selected,
    }
    path = root / "results/eval_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Evaluation profile protocol changed; use a separate output root")
    elif stage != "stage0":
        raise ValueError(f"Run the {name} baseline first with the same profile")
    elif create:
        write_json(path, protocol)
    return digest(protocol)
