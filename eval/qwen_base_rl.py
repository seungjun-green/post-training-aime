"""Validate an intermediate native-EOS Qwen base DAPO checkpoint for evaluation."""

import json
import re
from pathlib import Path

from eval.answer_only import BUNDLE_FILES as EVALUATOR_FILES
from train.dapo_resume import check_checkpoint

BUNDLE_FILES = EVALUATOR_FILES + [
    "eval/qwen_base_rl.py", "configs/qwen_base_rl_eval.yaml",
    "train/__init__.py", "train/dapo_resume.py",
]


def checkpoint_source(train_root, run_name, step, spec):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Unsafe TRAIN_RUN_NAME")
    if type(step) is not int or step <= 0:
        raise ValueError("STEP must be a positive integer")
    root = Path(train_root)
    checkpoint = root / "checkpoints" / run_name / f"checkpoint-{step}"
    marker, _ = check_checkpoint(checkpoint, require_training_state=False)
    manifest = json.loads((root / "logs" / run_name / "run_manifest.json").read_text())
    config = manifest["config"]
    if marker["run_identity"] != manifest["identity"]:
        raise ValueError("Checkpoint identity does not match the training manifest")
    if (marker["global_step"] != step or marker.get("model_kind") != "base"
            or config.get("model_kind") != "base"
            or config["run_name_prefix"] + "_base" != run_name
            or step > config["training"]["max_steps"]):
        raise ValueError("Checkpoint step or run does not match the base RL training manifest")
    if any(config["model"].get(key) != value for key, value in spec["base_model"].items()):
        raise ValueError("Expected RL starting from the pinned Qwen BASE model")
    if config.get("tokenizer") or config.get("generation"):
        raise ValueError("Expected the native-EOS run without tokenizer/generation overrides")
    model = json.loads((checkpoint / "config.json").read_text())
    generation = json.loads((checkpoint / "generation_config.json").read_text())
    tokenizer = json.loads((checkpoint / "tokenizer_config.json").read_text())
    def token_value(name):
        value = tokenizer.get(name)
        return value.get("content") if isinstance(value, dict) else value
    stops = generation.get("eos_token_id")
    expected = {
        "config.json model_type": (model.get("model_type"), "qwen2"),
        "config.json eos_token_id": (model.get("eos_token_id"), 151643),
        "generation_config.json eos_token_id":
            (stops if isinstance(stops, list) else [stops], [151643]),
        "tokenizer eos_token": (token_value("eos_token"), "<|endoftext|>"),
        "tokenizer pad_token": (token_value("pad_token"), "<|endoftext|>"),
    }
    for field, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"Expected native Qwen base {field}={wanted!r}; found {actual!r}")
    # Transformers save_pretrained writes this file and removes the JSON key.
    # Match its loading precedence: a separate template overrides the legacy key.
    template_path = checkpoint / "chat_template.jinja"
    template = (template_path.read_text() if template_path.is_file()
                else tokenizer.get("chat_template"))
    if isinstance(template, list):
        template = {item["name"]: item["template"] for item in template}
    if isinstance(template, dict):
        template = template.get("default")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Missing nonempty default chat template in chat_template.jinja "
                         "or tokenizer_config.json")
    if not (checkpoint / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Missing saved tokenizer: {checkpoint / 'tokenizer.json'}")
    return checkpoint, marker
