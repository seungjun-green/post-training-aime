"""Validate a completed DAPO run before continuing its optimizer into a new run."""

import json
import struct
from copy import deepcopy
from pathlib import Path


def check_checkpoint(path, *, require_training_state=True):
    """Check completion/state files and shard sizes without loading model weights."""
    path = Path(path).resolve()
    required = ["dapo_checkpoint.json", "config.json", "tokenizer_config.json"]
    if require_training_state:
        required += ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
    for name in required:
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise FileNotFoundError(f"Incomplete resumable checkpoint: {path / name}")
    marker = json.loads((path / "dapo_checkpoint.json").read_text())
    state = json.loads((path / "trainer_state.json").read_text()) if require_training_state else None
    if (state and marker["global_step"] != state["global_step"]) or path.name != f"checkpoint-{marker['global_step']}":
        raise ValueError("Checkpoint marker, folder and Trainer step disagree")
    index = path / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if not shards:
        raise ValueError("Checkpoint has no model weights")
    for name in shards:
        shard = path / name
        if shard.parent != path or not shard.is_file():
            raise FileNotFoundError(f"Missing model shard: {shard}")
        size = shard.stat().st_size
        with shard.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Incomplete model shard: {shard}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > min(size - 8, 100_000_000):
                raise ValueError(f"Incomplete model shard header: {shard}")
            header = json.loads(f.read(header_size))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        if size != 8 + header_size + end:
            raise ValueError(f"Incomplete model shard data: {shard}")
    return marker, state


def extension_source(config, checkpoint, output_root):
    """Permit only a larger total step budget; keep the original artifacts intact."""
    path = Path(checkpoint).resolve()
    marker, state = check_checkpoint(path)
    original_root = path.parents[2]
    if Path(output_root).resolve() == original_root:
        raise ValueError("Use a separate OUTPUT_ROOT for continuation; preserve the original run")
    manifest_path = original_root / "logs" / path.parent.name / "run_manifest.json"
    previous = json.loads(manifest_path.read_text())
    if marker["run_identity"] != previous["identity"]:
        raise ValueError("Parent checkpoint does not match its run manifest")
    original = previous["config"]
    old_limit = original["training"]["max_steps"]
    if state["global_step"] != old_limit or state["max_steps"] != old_limit:
        raise ValueError("Continue from the final checkpoint of the completed original run")
    if config["data"]["revision"] == "main":
        config["data"]["revision"] = original["data"]["revision"]
    comparable = deepcopy(config)
    target = comparable["training"]["max_steps"]
    comparable["training"]["max_steps"] = old_limit
    if comparable != original or target <= old_limit:
        raise ValueError("Continuation may only increase training.max_steps; all other settings must match")
    if original["training"]["lr_scheduler_type"] != "constant_with_warmup":
        raise ValueError("Extending this runner requires a schedule independent of the total step budget")
    return previous, {
        "checkpoint": str(path), "parent_run_identity": previous["identity"],
        "parent_git_commit": previous["git_commit"], "from_step": old_limit, "target_step": target,
    }


def check_extension_runtime(previous, manifest):
    for key in ["source", "data_report_digest", "packages", "parameter_precision", "compute_precision"]:
        if previous[key] != manifest[key]:
            raise ValueError(f"Continuation changed {key}; use the original source, data and runtime")
