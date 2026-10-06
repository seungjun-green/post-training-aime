"""Validate the full Qwen base + Kimi-style SFT checkpoint for inference."""

import json
import re
import struct
from pathlib import Path

from eval.answer_only import BUNDLE_FILES as EVALUATOR_FILES

BUNDLE_FILES = EVALUATOR_FILES + ["eval/qwen_kimi_sft.py", "configs/qwen_kimi_sft_eval.yaml"]


def checkpoint_source(train_root, run_name, epoch, spec, expected_identity=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Unsafe TRAIN_RUN_NAME")
    if type(epoch) is not int or not 1 <= epoch <= 5:
        raise ValueError("EPOCH must be an integer from 1 through 5")
    root = Path(train_root)
    checkpoint = root / "checkpoints/stage1" / run_name / f"epoch_{epoch}"
    for name in ["stage1_checkpoint.json", "config.json", "generation_config.json",
                 "tokenizer_config.json", "tokenizer.json"]:
        path = checkpoint / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
    marker = json.loads((checkpoint / "stage1_checkpoint.json").read_text())
    manifest = json.loads((root / "logs/stage1" / run_name / "run_manifest.json").read_text())
    if marker["epoch"] != epoch or marker["run_identity"] != manifest["identity"]:
        raise ValueError("Checkpoint epoch/identity does not match the training manifest")
    if expected_identity is not None and manifest["identity"] != expected_identity:
        raise ValueError("Checkpoint does not match EXPECTED_RUN_IDENTITY")
    config = manifest["config"]
    if config["run_name"] != run_name or config["stage"] != "stage1":
        raise ValueError("Training manifest belongs to another run")
    if any(config["model"][key] != value for key, value in spec["base_model"].items()):
        raise ValueError("Expected SFT starting from the pinned Qwen BASE model")
    data = config["data"]
    if (data.get("response_format") != "answer_only"
            or data["columns"] != {"question": "question", "answer": "kimi-style-reasoning-answer"}
            or config["training"]["num_train_epochs"] != 5):
        raise ValueError("Expected five-epoch SFT on the complete Kimi-style answer column")
    if any(data["hf_source"].get(key) != value for key, value in spec["training_data"].items()):
        raise ValueError("Training dataset differs from the pinned Kimi-style source")
    model = json.loads((checkpoint / "config.json").read_text())
    generation = json.loads((checkpoint / "generation_config.json").read_text())
    tokenizer = json.loads((checkpoint / "tokenizer_config.json").read_text())
    eos = tokenizer.get("eos_token")
    eos = eos.get("content") if isinstance(eos, dict) else eos
    stops = generation.get("eos_token_id")
    if (model.get("model_type") != "qwen2" or model.get("eos_token_id") != 151645
            or eos != "<|im_end|>" or (stops if isinstance(stops, list) else [stops]) != [151645]):
        raise ValueError("Expected the trained Qwen im_end EOS in tokenizer/model/generation configs")
    index = checkpoint / "model.safetensors.index.json"
    shards = (set(json.loads(index.read_text())["weight_map"].values())
              if index.exists() else {"model.safetensors"})
    if not shards:
        raise ValueError("Checkpoint has no model weights")
    for name in shards:
        path = checkpoint / name
        if path.parent != checkpoint or not path.is_file():
            raise FileNotFoundError(f"Missing model weights: {path}")
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Incomplete model shard: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 0 < header_size <= min(size - 8, 100_000_000):
                raise ValueError(f"Incomplete model shard header: {path}")
            header = json.loads(stream.read(header_size))
        end = max(value["data_offsets"][1] for key, value in header.items() if key != "__metadata__")
        if size != 8 + header_size + end:
            raise ValueError(f"Incomplete model shard data: {path}")
    return checkpoint, marker
