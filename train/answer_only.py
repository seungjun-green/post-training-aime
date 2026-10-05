"""Freeze the generated answer dataset and identify self-contained SFT code."""

import argparse
import re
from pathlib import Path

import yaml

from common.io import digest, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = [
    "common/__init__.py", "common/io.py", "common/english_prompts.py", "common/process.py",
    "train/__init__.py", "train/sft_data.py", "train/sft_trainer.py", "train/stage1_sft.py",
    "train/answer_only.py", "scripts/setup_stage1_runtime.py", "requirements-stage1.lock",
    "configs/stage1_sft_deepseek_pro_answer_only.yaml",
]


def training_code_digest():
    return digest({p: (ROOT / p).read_text() for p in BUNDLE_FILES})


def prepare_run(dataset_path, template_path, output_root, run_name):
    """Save only question/answer, retaining null rows and their original indices.

    Each run pins its input on first preparation; a rerun cannot replace that
    snapshot or silently change its settings, including before an epoch is saved.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Unsafe run_name")
    config = yaml.safe_load(Path(template_path).read_text())
    if config["data"].get("response_format") != "answer_only":
        raise ValueError("Expected an answer-only config")
    source = read_jsonl(dataset_path)
    if not source:
        raise ValueError(f"Empty or missing dataset: {dataset_path}")
    columns = config["data"]["columns"]
    rows = []
    for i, row in enumerate(source):
        question = row.get(columns["question"])
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question at source row {i}")
        if columns["answer"] not in row:
            raise ValueError(f"Missing answer column at source row {i}: {columns['answer']}")
        answer = row[columns["answer"]]
        if answer is not None and not isinstance(answer, str):
            raise ValueError(f"Expected answer text or null at source row {i}")
        rows.append({columns["question"]: question, columns["answer"]: answer})
    if not any(isinstance(r[columns["answer"]], str) and r[columns["answer"]].strip() for r in rows):
        raise ValueError("No nonempty generated answers")
    root = Path(output_root).resolve()
    snapshot_dir = root / "inputs" / run_name
    snapshot = snapshot_dir / "dataset.jsonl"
    config_path = snapshot_dir / "training.yaml"
    config.update(run_name=run_name, output_root=str(root))
    config["data"].update(
        path=str(snapshot), expected_rows=len(rows), content_digest=digest(rows),
        ids_digest=digest([f"regenerated:{i}" for i in range(len(rows))]),
    )
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
        raise ValueError("Run data/settings changed; choose a new RUN_NAME")
    if snapshot.exists():
        if digest(read_jsonl(snapshot)) != config["data"]["content_digest"]:
            raise ValueError("Existing dataset snapshot differs; choose a new RUN_NAME")
    else:
        write_jsonl(snapshot, rows)
    if not config_path.exists():
        temporary = config_path.with_suffix(".tmp")
        temporary.write_text(yaml.safe_dump(config, sort_keys=False))
        temporary.replace(config_path)
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    print("Pinned training config:", prepare_run(
        args.dataset, args.config, args.output_root, args.run_name,
    ))


if __name__ == "__main__":
    main()
