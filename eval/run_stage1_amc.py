"""AMC-only epoch curve using the frozen English baseline's generation and scoring."""

import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import yaml

from common.io import digest, write_json
from eval.run_english_eval import (
    ROOT,
    bind_protocol,
    git_identity,
    load_eval_sets,
    select_model_revision,
)
from eval.run_eval import bind_runtime, evaluate, package_versions, resolve_model, validate_config


def select_amc(config, suite):
    selected_config, selected_suite = deepcopy(config), deepcopy(suite)
    selected_config["datasets"] = {"amc23": selected_config["datasets"]["amc23"]}
    selected_suite["datasets"] = {"amc23": selected_suite["datasets"]["amc23"]}
    return selected_config, selected_suite


def validate_baseline(root, config, suite):
    """Read/validate the existing full protocol. Never establish a subset baseline."""
    ids = {name: entry["ids"] for name, entry in suite["datasets"].items()}
    return bind_protocol(root, config, suite, "stage1", ids, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/eval_english.yaml"))
    parser.add_argument("--suite", default=str(ROOT / "configs/english_eval_suite.json"))
    parser.add_argument("--output_root", required=True, help="Existing baseline output root, before /full")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("run_name must use letters, digits, underscores or hyphens")
    config = yaml.safe_load(Path(args.config).read_text())
    suite = json.loads(Path(args.suite).read_text())
    validate_config(config)
    root = Path(args.output_root) / "full"
    protocol_id = validate_baseline(root, config, suite)
    selected_config, selected_suite = select_amc(config, suite)
    token = os.getenv("HF_TOKEN")
    datasets = load_eval_sets(selected_config, selected_suite, token)
    commit = git_identity()
    result_dir = root / "results/stage1"
    run_path = result_dir / f"{args.run_name}_manifest.json"
    saved = json.loads(run_path.read_text()) if run_path.exists() else None
    revision, model_digest = resolve_model(
        args.model, select_model_revision(args.model, args.revision, saved), token,
    )
    from eval.engines import check_hardware
    from eval.english_engines import create_engine

    metadata = {
        "model": args.model, "model_revision": revision, "model_digest": model_digest,
        "git_commit": commit, "stage": "stage1", "run_name": args.run_name,
        "mode": "subset", "subset": ["amc23"], "config": selected_config,
        "protocol_digest": protocol_id, "dataset_suite": selected_suite,
        "selected_ids": {"amc23": [r["id"] for r in datasets["amc23"]]},
        "subset_runner_digest": digest(Path(__file__).read_text()),
        "packages": package_versions(), "hardware": check_hardware(config),
    }
    generations = result_dir / f"{args.run_name}_generations.jsonl"
    if saved is not None:
        if saved != metadata:
            raise ValueError("AMC run identity changed; use a new run_name")
    elif generations.exists():
        raise ValueError("AMC generations exist without their manifest")
    else:
        write_json(run_path, metadata)
    engine, fallback = create_engine(args.model, selected_config, revision)
    bind_runtime(root, engine.tokenizer, "stage1")
    engine_info = {
        "engine": engine.name, "fallback_reason": fallback,
        "chat_template_digest": digest(engine.tokenizer.chat_template),
    }
    engine_path = result_dir / f"{args.run_name}_engine.json"
    if engine_path.exists() and json.loads(engine_path.read_text()) != engine_info:
        raise ValueError("AMC engine changed during resume")
    write_json(engine_path, engine_info)
    results = evaluate(engine, datasets, selected_config, generations)
    destination = result_dir / f"{args.run_name}.json"
    write_json(destination, {**metadata, **engine_info, "metrics": results})
    print(json.dumps(results, indent=2), flush=True)
    print("Saved:", destination, flush=True)


if __name__ == "__main__":
    main()
