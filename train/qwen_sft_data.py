"""Pin the published Kimi-style targets and configure Qwen's existing chat tokens."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from common.io import digest, read_jsonl, write_jsonl
from train.answer_only import prepare_run

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = [
    "common/__init__.py", "common/io.py", "common/english_prompts.py", "common/process.py",
    "train/__init__.py", "train/sft_data.py", "train/sft_trainer.py", "train/stage1_sft.py",
    "train/answer_only.py", "train/qwen_sft_data.py", "train/qwen_sft_trainer.py",
    "train/stage1_qwen_sft.py", "train/dapo_logps.py", "train/llama_sft_data.py",
    "scripts/setup_stage1_runtime.py", "requirements-stage1.lock", "configs/sft_qwen25_3b_s1_kimi.yaml",
]


def training_code_digest():
    return digest({name: (ROOT / name).read_text() for name in BUNDLE_FILES})


def configure_tokenizer(tokenizer, config):
    # Qwen base ships a ChatML template but has endoftext as its original EOS.
    # Supervise/stop on that template's existing im_end; add no vocabulary tokens.
    vocabulary, template = tokenizer.get_vocab(), tokenizer.chat_template
    for key, value in config["tokenizer"].items():
        if value not in vocabulary:
            raise ValueError(f"Missing native Qwen token: {value}")
        setattr(tokenizer, key, value)
    tokenizer.padding_side = "right"
    if not template or tokenizer.eos_token != "<|im_end|>" or tokenizer.pad_token != "<|endoftext|>":
        raise ValueError("Use Qwen's native template, im_end EOS, and endoftext padding")
    if tokenizer.get_vocab() != vocabulary or tokenizer.chat_template != template:
        raise ValueError("Qwen vocabulary and chat template must remain unchanged")
    return tokenizer


def prepare_hf_run(template_path, output_root, run_name, *, source_rows=None, token=None):
    """Freeze the two training columns on Drive; reuse the verified snapshot on rerun."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Unsafe run_name")
    config = yaml.safe_load(Path(template_path).read_text())
    source = config["data"]["hf_source"]
    if not re.fullmatch(r"[a-f0-9]{40}", source["revision"]):
        raise ValueError("Pin the HF dataset revision")
    columns = list(config["data"]["columns"].values())
    snapshot = Path(output_root) / "inputs" / run_name / "dataset.jsonl"
    if source_rows is None:
        if snapshot.exists():
            source_rows = read_jsonl(snapshot)
        else:
            from datasets import load_dataset

            source_rows = load_dataset(source["repo"], name=source["config"], split=source["split"],
                                       revision=source["revision"], token=token).select_columns(columns)
    rows = [{column: row[column] for column in columns} for row in source_rows]
    if len(rows) != source["expected_rows"] or digest(rows) != source["selected_content_digest"]:
        raise ValueError("Question/answer rows differ from the pinned Hugging Face dataset")
    with tempfile.TemporaryDirectory(prefix="qwen-kimi-snapshot-") as directory:
        path = Path(directory) / "source.jsonl"
        write_jsonl(path, rows)
        return prepare_run(path, template_path, output_root, run_name)


def latest_checkpoint(checkpoint_root, epochs):
    """Select only an atomically completed epoch; never silently restart a partial save."""
    root = Path(checkpoint_root)
    completed = sorted(root.glob("epoch_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not completed:
        return None
    path = completed[-1]
    marker = json.loads((path / "stage1_checkpoint.json").read_text())
    state = json.loads((path / "trainer_state.json").read_text())
    if (path.name != f"epoch_{marker['epoch']}" or not 1 <= marker["epoch"] <= epochs
            or marker["global_step"] != state["global_step"]):
        raise ValueError("Epoch checkpoint marker and Trainer state disagree")
    for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json", "tokenizer_config.json"]:
        if not (path / name).is_file() or not (path / name).stat().st_size:
            raise FileNotFoundError(f"Incomplete checkpoint: {path / name}")
    index = path / "model.safetensors.index.json"
    shards = (set(json.loads(index.read_text())["weight_map"].values())
              if index.exists() else {"model.safetensors"})
    if not shards:
        raise ValueError("Checkpoint has no model shards")
    for name in shards:
        shard = path / name
        if shard.parent != path or not shard.is_file() or not shard.stat().st_size:
            raise FileNotFoundError(f"Incomplete checkpoint shard: {shard}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    print("Pinned training config:", prepare_hf_run(args.config, args.output_root, args.run_name,
                                                   token=os.getenv("HF_TOKEN")))


if __name__ == "__main__":
    main()
