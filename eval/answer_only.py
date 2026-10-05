"""Checkpoint selection and code identity for bundled answer-only SFT evaluation."""

import json
import re
from pathlib import Path

from common.io import digest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = [
    "common/__init__.py", "common/io.py", "common/process.py", "common/english_prompts.py",
    "common/prompts.py", "common/math_text.py", "eval/__init__.py", "eval/answer_only.py",
    "eval/run_amc_math_eval.py", "eval/run_batched_eval.py", "eval/run_english_eval.py",
    "eval/run_eval.py", "eval/batched_generation.py", "eval/budget_generation.py",
    "eval/profiles.py", "eval/engines.py", "eval/english_engines.py", "eval/scoring.py",
    "configs/answer_only_eval_amc_math.yaml", "configs/eval_english.yaml",
    "configs/english_eval_suite.json", "configs/eval_execution.yaml", "configs/eval_profiles.yaml",
    "scripts/setup_eval_runtime.py", "requirements-eval.lock",
]


def evaluation_code_digest():
    return digest({name: (ROOT / name).read_text() for name in BUNDLE_FILES})


def checkpoint_source(train_root, run_name, epoch, base_model):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Unsafe TRAIN_RUN_NAME")
    if type(epoch) is not int or not 1 <= epoch <= 5:
        raise ValueError("EPOCH must be an integer from 1 through 5")
    root = Path(train_root)
    checkpoint = root / "checkpoints/stage1" / run_name / f"epoch_{epoch}"
    for name in ["stage1_checkpoint.json", "config.json", "tokenizer_config.json", "tokenizer.json"]:
        path = checkpoint / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing checkpoint file: {path}. Check TRAIN_ROOT and TRAIN_RUN_NAME.")
    marker = json.loads((checkpoint / "stage1_checkpoint.json").read_text())
    manifest = json.loads((root / "logs/stage1" / run_name / "run_manifest.json").read_text())
    if marker["epoch"] != epoch or marker["run_identity"] != manifest["identity"]:
        raise ValueError("Checkpoint epoch/identity does not match its training run")
    config = manifest["config"]
    if config["run_name"] != run_name or config["stage"] != "stage1":
        raise ValueError("Training manifest belongs to another run")
    if any(config["model"][key] != base_model[key] for key in ["repo", "revision"]):
        raise ValueError("Expected training from the original EXAONE-3.5-2.4B-Instruct")
    data = config["data"]
    if (data.get("response_format") != "answer_only"
            or data["columns"] != {"question": "question", "answer": "deepseek-v4-pro_answer"}
            or config["training"]["num_train_epochs"] != 5):
        raise ValueError("Expected the five-epoch answer-only SFT run")
    index = checkpoint / "model.safetensors.index.json"
    shards = (set(json.loads(index.read_text())["weight_map"].values())
              if index.exists() else {"model.safetensors"})
    if not shards:
        raise ValueError("Checkpoint has no model weights")
    for name in shards:
        path = checkpoint / name
        if path.parent != checkpoint or not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing model weights: {path}")
    return checkpoint, marker
