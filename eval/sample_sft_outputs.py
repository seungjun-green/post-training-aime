"""Inspect the same 15 questions with original EXAONE s1 SFT or Qwen2.5-3B-Instruct."""

import argparse
import json
import os
import random
import re
from pathlib import Path

import yaml

from common.io import digest, read_jsonl, write_json, write_jsonl
from eval.profiles import load_profile
from eval.run_batched_eval import evaluate_batched, runner_digest, validate_execution
from eval.run_english_eval import ROOT, code_fingerprint, load_eval_sets
from eval.run_eval import package_versions, resolve_model
from train.sft_data import DEFAULT_COLUMNS


def select_questions(datasets, groups, seed):
    """Uniform sampling without replacement from each pooled group, not per AIME year."""
    selected = {}
    seen = set()
    for group, spec in groups.items():
        pool = sorted(
            ((name, row) for name in spec["datasets"] for row in datasets[name]),
            key=lambda item: (item[0], str(item[1]["id"])),
        )
        count = spec["count"]
        if type(count) is not int or not 0 < count <= len(pool):
            raise ValueError(f"Invalid sample count: {group}")
        rng = random.Random(f"{seed}:{group}")
        for name, row in rng.sample(pool, count):
            key = (name, str(row["id"]))
            if key in seen:
                raise ValueError("Sampling groups must not overlap")
            seen.add(key)
            selected.setdefault(name, []).append(row)
    return {name: sorted(rows, key=lambda row: str(row["id"]))
            for name, rows in sorted(selected.items())}


def split_response(text, plain_response=False):
    """Best-effort display fields; response always preserves the exact generation."""
    if "<think>" not in text:
        if plain_response:
            return {"reasoning": None, "answer_text": text, "parse_status": "plain_response"}
        return {"reasoning": None, "answer_text": None, "parse_status": "missing_think_tag"}
    _, _, tail = text.partition("<think>")
    reasoning, end, answer = tail.partition("</think>")
    return {"reasoning": reasoning, "answer_text": answer if end else None,
            "parse_status": "closed_think" if end else "unclosed_think"}


def checkpoint_source(train_root, settings):
    expected = yaml.safe_load((ROOT / settings["training_config"]).read_text())
    run = expected["run_name"]
    epoch = settings["epoch"]
    checkpoint = Path(train_root) / "checkpoints" / expected["stage"] / run / f"epoch_{epoch}"
    marker_path = checkpoint / "stage1_checkpoint.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Original s1 SFT checkpoint not found: {marker_path}. Check TRAIN_ROOT.")
    marker = json.loads(marker_path.read_text())
    manifest = json.loads((Path(train_root) / "logs" / expected["stage"] / run / "run_manifest.json").read_text())
    actual = manifest["config"]
    if (marker["epoch"] != epoch or marker["run_identity"] != manifest["identity"]
            or actual["model"] != expected["model"] or actual["run_name"] != run
            or actual["data"].get("columns", DEFAULT_COLUMNS) != DEFAULT_COLUMNS):
        raise ValueError("Expected original EXAONE s1 SFT, using original DeepSeek columns")
    if not (checkpoint / "config.json").is_file() or not list(checkpoint.glob("*.safetensors")):
        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    return checkpoint, marker


def model_source(train_root, settings):
    kind = settings.get("model_kind", "sft")
    if kind == "sft":
        checkpoint, marker = checkpoint_source(train_root, settings)
        return str(checkpoint), None, marker
    if kind == "qwen":
        model = settings["qwen_model"]
        if model["repo"] != "Qwen/Qwen2.5-3B-Instruct" or not re.fullmatch(r"[a-f0-9]{40}", model["revision"]):
            raise ValueError("Expected Qwen/Qwen2.5-3B-Instruct with a pinned revision")
        return model["repo"], model["revision"], None
    raise ValueError("model_kind must be 'sft' or 'qwen'")


def export_responses(journal, output, datasets, config, source):
    rows = {(name, str(row["id"])): row for name, values in datasets.items() for row in values}
    records = [record for entry in read_jsonl(journal, repair_tail=True) for record in entry["records"]]
    records.sort(key=lambda record: (record["dataset"], str(record["id"])))
    exported = []
    for record in records:
        spec = config["datasets"][record["dataset"]]
        row = rows[(record["dataset"], str(record["id"]))]
        entry = {**record, "question": row[spec["problem_column"]],
                 "gold_answer": record["answer"],
                 **split_response(record["response"], plain_response=source.get("model_kind") == "qwen"),
                 "truncated": record["finish_reason"] == "length", "source": source}
        del entry["answer"]  # Do not mistake the reference answer for generated text.
        exported.append(entry)
    write_jsonl(output, exported)
    return exported


def run(settings, train_root, output_root, prepare_only=False):
    token = os.getenv("HF_TOKEN")
    model, requested_revision, marker = model_source(train_root, settings)
    model_kind = settings.get("model_kind", "sft")
    config = yaml.safe_load((ROOT / settings["evaluation_config"]).read_text())
    execution = yaml.safe_load((ROOT / settings["execution_config"]).read_text())
    config, execution = load_profile("sample1", config, execution, sampling_override=settings["sampling"])
    config["seed"] = settings["seed"]
    config["max_new_tokens"] = settings["max_new_tokens"]
    if type(config["max_new_tokens"]) is not int or not 0 < config["max_new_tokens"] < config["max_model_len"]:
        raise ValueError("max_new_tokens must be positive and smaller than the context window")
    validate_execution(execution)
    suite = json.loads((ROOT / settings["suite"]).read_text())
    datasets = select_questions(load_eval_sets(config, suite, token), settings["groups"], settings["seed"])
    config["datasets"] = {name: config["datasets"][name] for name in datasets}
    selection = [{"dataset": name, "id": row["id"],
                  "question": row[config["datasets"][name]["problem_column"]],
                  "gold_answer": row[config["datasets"][name]["answer_column"]]}
                 for name, rows in datasets.items() for row in rows]
    identity = {"settings": settings, "model": model, "model_revision": requested_revision, "marker": marker,
                "config": config, "execution": execution, "suite": suite,
                "selection": selection, "code_digest": digest([
                    code_fingerprint(), runner_digest(), Path(__file__).read_text()])}
    output = Path(output_root) / model_kind / digest(identity)[:16]
    write_json(output / "selection.json", selection)
    write_json(output / "settings.json", identity)
    print(f"Model: {model}\nSelected: {len(selection)} problems\nOutput: {output}", flush=True)
    write_json(Path(output_root) / "latest_run.json", {"directory": str(output)})
    if prepare_only:
        return output

    from eval.engines import check_hardware
    from eval.english_engines import create_engine

    hardware = check_hardware(config)
    print("Verifying model identity before generation...", flush=True)
    revision, model_digest = resolve_model(model, requested_revision, token)
    metadata = {"model_digest": model_digest, "packages": package_versions(), "hardware": hardware}
    manifest_path = output / "runtime.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != metadata:
        raise ValueError("Model/runtime changed. Use a new OUTPUT_ROOT to keep saved responses separate.")
    write_json(manifest_path, metadata)
    engine, fallback = create_engine(model, config, revision)
    if engine.name != "vllm":
        raise ValueError(f"Continuous generation requires vLLM: {fallback}")
    source = {"model_kind": model_kind, "model": model, "model_revision": revision,
              "checkpoint": model if model_kind == "sft" else None,
              "marker": marker, "model_digest": model_digest,
              "seed": settings["seed"], "sampling": settings["sampling"],
              "max_new_tokens": config["max_new_tokens"], "budget_forcing": False}
    journal = output / "problems.jsonl"
    try:
        _, summary = evaluate_batched(engine, datasets, config, execution, journal)
        write_json(output / "sample_metrics.json", summary)
    finally:
        export_responses(journal, output / "generations.jsonl", datasets, config, source)
    print(f"Saved full reasoning + answers: {output / 'generations.jsonl'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sft_sample15.yaml"))
    parser.add_argument("--train-root", default=".", help="Original SFT Drive root; unused for Qwen")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.train_root, args.output_root, args.prepare_only)


if __name__ == "__main__":
    main()
