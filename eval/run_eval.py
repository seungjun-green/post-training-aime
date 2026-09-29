"""Frozen Spec 1 evaluation protocol; usable unchanged by every subsequent stage."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from common.io import append_jsonl, digest, read_jsonl, write_json
from eval.scoring import korean_ratio, metrics, score_response


def code_fingerprint():
    files = [
        ROOT / p
        for p in [
            "common/prompts.py",
            "common/math_text.py",
            "common/io.py",
            "eval/run_eval.py",
            "eval/scoring.py",
            "eval/engines.py",
        ]
    ]
    return digest({str(p.relative_to(ROOT)): p.read_text() for p in files})


def package_versions():
    result = {}
    for name in [
        "vllm",
        "torch",
        "transformers",
        "datasets",
        "math-verify",
        "latex2sympy2-extended",
    ]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def validate_config(config):
    if config["engine"] != "vllm" or config["tensor_parallel_size"] != 1:
        raise ValueError("Spec 1 requires vLLM first and one GPU")
    if config["max_new_tokens"] >= config["max_model_len"]:
        raise ValueError("No context budget remains for the prompt")
    if config["temperature"] <= 0 or not 0 < config["top_p"] <= 1:
        raise ValueError("Invalid sampling configuration")


def bind_protocol(root, config, suite, stage):
    protocol = {"config": config, "suite": suite, "code_digest": code_fingerprint()}
    path = root / "results" / "eval_protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != protocol:
            raise ValueError("Evaluation protocol, code, or dataset suite changed since baseline")
    elif stage != "stage0":
        raise ValueError("Run stage0 baseline first to freeze the evaluation protocol")
    else:
        write_json(path, protocol)
    return digest(protocol)


def bind_runtime(root, tokenizer, stage):
    runtime = {
        "chat_template_digest": digest(tokenizer.chat_template),
        "tokenizer_vocab_digest": digest(tokenizer.get_vocab()),
        "special_tokens": tokenizer.special_tokens_map,
        "packages": package_versions(),
    }
    path = root / "results" / "eval_runtime.json"
    if path.exists():
        if json.loads(path.read_text()) != runtime:
            raise ValueError(
                "Tokenizer, chat template, or evaluation dependencies changed since baseline"
            )
    elif stage != "stage0":
        raise ValueError("Baseline runtime has not been recorded")
    else:
        write_json(path, runtime)


def resolve_model(model, revision, token):
    from huggingface_hub import HfApi

    path = Path(model)
    if path.is_dir():
        if revision:
            raise ValueError("--revision applies only to Hugging Face models")
        # Include weights and tokenizer/config files so same-directory replacements cannot resume.
        hashes = {}
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.suffix in {".json", ".safetensors", ".bin", ".model", ".py"}:
                h = hashlib.sha256()
                with file.open("rb") as f:
                    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
                hashes[str(file.relative_to(path))] = h.hexdigest()
        if not hashes:
            raise ValueError("Local model directory has no model files")
        return None, digest(hashes)
    resolved = HfApi(token=token).model_info(model, revision=revision).sha
    return resolved, resolved


def load_eval_sets(config, suite, token):
    from datasets import load_dataset

    if set(config["datasets"]) != set(suite["datasets"]):
        raise ValueError("Evaluation suite must contain exactly the five configured datasets")
    result = {}
    for name, spec in config["datasets"].items():
        locked = suite["datasets"][name]
        if locked["repo"] != spec["repo"] or locked["split"] != spec["split"]:
            raise ValueError(f"Upload manifest/config mismatch for {name}")
        if locked["source_rows"] != spec["source_rows"]:
            raise ValueError(f"Source benchmark size mismatch for {name}")
        dataset = load_dataset(
            spec["repo"], split=spec["split"], revision=locked["revision"], token=token
        )
        rows = list(dataset)
        if not rows or len(rows) != locked["rows"] or [r["id"] for r in rows] != locked["ids"]:
            raise ValueError(f"Evaluation IDs or size changed for {name}")
        if len({str(r["id"]) for r in rows}) != len(rows):
            raise ValueError(f"Duplicate evaluation IDs: {name}")
        for row in rows:
            if (
                not isinstance(row[spec["problem_column"]], str)
                or not row[spec["problem_column"]].strip()
            ):
                raise ValueError(f"Missing Korean problem: {name}/{row['id']}")
            if row[spec["answer_column"]] is None:
                raise ValueError(f"Missing answer: {name}/{row['id']}")
        result[name] = rows
    return result


def problem_seed(seed, dataset, identifier):
    return int(digest([seed, dataset, str(identifier)])[:8], 16)


def evaluate(engine, datasets, config, path):
    records = read_jsonl(path, repair_tail=True)
    known = {(name, str(row["id"])) for name, rows in datasets.items() for row in rows}
    grouped = defaultdict(dict)
    for record in records:
        key = record["dataset"], str(record["id"])
        if key not in known:
            raise ValueError(f"Unexpected saved generation: {key}")
        n = config["datasets"][key[0]]["n"]
        index = record["sample_index"]
        if index in grouped[key] or not 0 <= index < n:
            raise ValueError(f"Duplicate/invalid sample index: {key}/{index}")
        grouped[key][index] = record
    for name, rows in datasets.items():
        spec = config["datasets"][name]
        for i, row in enumerate(rows):
            key = name, str(row["id"])
            existing = grouped[key]
            if len(existing) == spec["n"]:
                continue
            seed = problem_seed(config["seed"], name, row["id"])
            prompt, responses = engine.generate(row[spec["problem_column"]], spec["n"], seed)
            if len(responses) != spec["n"]:
                raise ValueError("Engine returned an incorrect number of samples")
            for sample_index, response in enumerate(responses):
                if sample_index in existing:
                    if response["text"] != existing[sample_index]["response"]:
                        raise ValueError(
                            "Partial-problem resume is not deterministic; use a new run_name"
                        )
                    continue
                extracted, correct = score_response(
                    response["text"], row[spec["answer_column"]], config["verify_timeout_seconds"]
                )
                record = {
                    "dataset": name,
                    "id": row["id"],
                    "sample_index": sample_index,
                    "problem_seed": seed,
                    "prompt": prompt,
                    "response": response["text"],
                    "answer": row[spec["answer_column"]],
                    "extracted_answer": extracted,
                    "correct": correct,
                    "token_count": response["token_count"],
                    "finish_reason": response["finish_reason"],
                    "korean_response_ratio": korean_ratio(response["text"]),
                }
                append_jsonl(path, record)
                records.append(record)
            print(f"{name}: {i + 1}/{len(rows)} problems", flush=True)
    return {
        name: metrics([r for r in records if r["dataset"] == name], spec["n"], config["pass_k"])
        for name, spec in config["datasets"].items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/eval.yaml"))
    parser.add_argument("--output_root", default=".")
    parser.add_argument(
        "--suite", help="Defaults to <output_root>/eval_suite.json from translation upload"
    )
    parser.add_argument(
        "--revision", help="Optional Hugging Face model revision; resolved to a commit"
    )
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in [args.stage, args.run_name]):
        parser.error("stage and run_name must contain only letters, digits, underscores or hyphens")
    config = yaml.safe_load(Path(args.config).read_text())
    validate_config(config)
    root = Path(args.output_root)
    suite_path = Path(args.suite) if args.suite else root / "eval_suite.json"
    if not suite_path.exists():
        raise ValueError(
            "Missing eval_suite.json: complete the reviewed full translation and upload first"
        )
    suite = json.loads(suite_path.read_text())
    protocol_id = bind_protocol(root, config, suite, args.stage)
    token = os.getenv("HF_TOKEN")
    revision, model_digest = resolve_model(args.model, args.revision, token)
    datasets = load_eval_sets(config, suite, token)
    from eval.engines import check_hardware, create_engine

    hardware = check_hardware(config)
    metadata = {
        "model": args.model,
        "model_revision": revision,
        "model_digest": model_digest,
        "stage": args.stage,
        "run_name": args.run_name,
        "config": config,
        "seed": config["seed"],
        "protocol_digest": protocol_id,
        "dataset_suite": suite,
        "packages": package_versions(),
        "hardware": hardware,
    }
    result_dir = root / "results" / args.stage
    run_path = result_dir / f"{args.run_name}_manifest.json"
    generations = result_dir / f"{args.run_name}_generations.jsonl"
    if run_path.exists():
        if json.loads(run_path.read_text()) != metadata:
            raise ValueError("Run identity changed; choose a new run_name")
    elif generations.exists():
        raise ValueError("Generations exist without their run manifest; refusing to reuse them")
    else:
        write_json(run_path, metadata)
    engine, fallback_reason = create_engine(args.model, config, revision)
    bind_runtime(root, engine.tokenizer, args.stage)
    # Keep engine choice stable when resuming a partially completed run.
    engine_path = result_dir / f"{args.run_name}_engine.json"
    engine_info = {
        "engine": engine.name,
        "fallback_reason": fallback_reason,
        "chat_template_digest": digest(engine.tokenizer.chat_template),
    }
    if engine_path.exists() and json.loads(engine_path.read_text()) != engine_info:
        raise ValueError("Engine or chat template changed during resume")
    write_json(engine_path, engine_info)
    results = evaluate(engine, datasets, config, generations)
    write_json(
        result_dir / f"{args.run_name}.json", {**metadata, **engine_info, "metrics": results}
    )
    print(f"Saved {result_dir / (args.run_name + '.json')}")


if __name__ == "__main__":
    main()
