"""English EXAONE evaluation. Run with python -m eval.run_english_eval --help."""

import argparse
import json
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

from common.io import digest, write_json
from eval.run_eval import bind_runtime, evaluate, package_versions, resolve_model, validate_config

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
# LG's February 2026 loader requires Transformers 5. This earlier revision has
# identical weights/tokenizer/config and works with requirements-eval.lock.
BASE_MODEL_REVISION = "e949c91dec92095908d34e6b560af77dd0c993f8"


def select_model_revision(model, requested=None, saved=None):
    if requested is not None:
        return requested
    if saved and saved["model"] == model:
        return saved["model_revision"]
    return BASE_MODEL_REVISION if model == BASE_MODEL else None


def load_eval_sets(config, suite, token=None, loader=None):
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    if config.get("language") != "en" or suite.get("language") != "en":
        raise ValueError("English configuration and suite are required")
    if set(config["datasets"]) != set(suite["datasets"]):
        raise ValueError("Expected exactly the configured evaluation sets")
    result = {}
    for name, spec in config["datasets"].items():
        locked = suite["datasets"][name]
        for key in ["repo", "config", "split", "problem_column", "answer_column", "source_rows"]:
            if spec[key] != locked[key]:
                raise ValueError(f"Suite/config mismatch: {name}/{key}")
        if locked["role"] != "eval" or locked["removed_rows"] != 0:
            raise ValueError(f"Expected unchanged evaluation data: {name}")
        if not re.fullmatch("[0-9a-f]{40}", locked["revision"]):
            raise ValueError(f"Unpinned dataset: {name}")
        original = list(
            loader(
                locked["repo"],
                name=locked["config"],
                split=locked["split"],
                revision=locked["revision"],
                token=token,
            )
        )
        if len(original) != locked["rows"] or len(original) != spec["source_rows"]:
            raise ValueError(f"Benchmark size mismatch: {name}")
        if digest(original) != locked["content_digest"]:
            raise ValueError(f"Dataset contents differ from original English source: {name}")
        rows = []
        for i, source in enumerate(original):
            identifier = source
            if locked["id_column"]:
                for part in locked["id_column"].split("."):
                    identifier = identifier[part]
            else:
                identifier = f"{name}:{locked['split']}:{i}"
            row = dict(source, id=identifier)
            problem, answer = row[spec["problem_column"]], row[spec["answer_column"]]
            if (
                not isinstance(problem, str)
                or not problem.strip()
                or answer is None
                or not str(answer).strip()
            ):
                raise ValueError(f"Missing English problem or answer: {name}/{identifier}")
            rows.append(row)
        ids = [r["id"] for r in rows]
        if ids != locked["ids"] or len({str(i) for i in ids}) != len(ids):
            raise ValueError(f"Evaluation IDs changed or duplicated: {name}")
        result[name] = rows
    return result


def code_fingerprint():
    paths = [
        "common/english_prompts.py",
        "common/prompts.py",
        "common/io.py",
        "common/math_text.py",
        "eval/run_english_eval.py",
        "eval/english_engines.py",
        "eval/run_eval.py",
        "eval/engines.py",
        "eval/scoring.py",
        "requirements-eval.lock",
    ]
    return digest({p: (ROOT / p).read_text() for p in paths})


def bind_protocol(root, config, suite, stage, selected_ids, smoke):
    protocol = {
        "config": config,
        "suite": suite,
        "code_digest": code_fingerprint(),
        "mode": "smoke" if smoke else "full",
        "selected_ids": selected_ids,
    }
    path = root / "results/eval_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("English evaluation protocol changed; use a separate output root")
    elif stage != "stage0":
        raise ValueError("Run the English stage0 baseline first")
    else:
        write_json(path, protocol)
    return digest(protocol)


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    try:
        commit = git("rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        raise ValueError(
            "Commit the project before evaluation so runs have a Git revision"
        ) from None
    if git("status", "--porcelain"):
        raise ValueError("Project has uncommitted files; commit or use a clean checkout")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--stage", default="stage0")
    parser.add_argument("--run_name", default="baseline_english")
    parser.add_argument("--config", default=str(ROOT / "configs/eval_english.yaml"))
    parser.add_argument("--suite", default=str(ROOT / "configs/english_eval_suite.json"))
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--smoke", action="store_true", help="One problem/sample per set; separate smoke results"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check all dataset pins, contents and IDs without a GPU",
    )
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in [args.stage, args.run_name]):
        parser.error("stage/run_name must use letters, digits, underscores or hyphens")
    config = yaml.safe_load(Path(args.config).read_text())
    validate_config(config)
    suite = json.loads(Path(args.suite).read_text())
    token = os.getenv("HF_TOKEN")
    datasets = load_eval_sets(config, suite, token)
    for name, rows in datasets.items():
        print(f"Validated {name}: {len(rows)} unchanged English problems", flush=True)
    if args.validate_only:
        return
    commit = git_identity()
    root = Path(args.output_root) / ("smoke" if args.smoke else "full")
    if args.smoke:
        config = deepcopy(config)
        for name in datasets:
            datasets[name] = datasets[name][:1]
            config["datasets"][name]["n"] = 1
    selected = {name: [r["id"] for r in rows] for name, rows in datasets.items()}
    protocol_id = bind_protocol(root, config, suite, args.stage, selected, args.smoke)
    result_dir = root / "results" / args.stage
    run_path = result_dir / f"{args.run_name}_manifest.json"
    saved = json.loads(run_path.read_text()) if run_path.exists() else None
    # Resume the same immutable model even if its upstream main branch has moved.
    requested_revision = select_model_revision(args.model, args.revision, saved)
    revision, model_digest = resolve_model(args.model, requested_revision, token)
    from eval.engines import check_hardware
    from eval.english_engines import create_engine

    hardware = check_hardware(config)
    metadata = {
        "model": args.model,
        "model_revision": revision,
        "model_digest": model_digest,
        "git_commit": commit,
        "stage": args.stage,
        "run_name": args.run_name,
        "mode": "smoke" if args.smoke else "full",
        "config": config,
        "protocol_digest": protocol_id,
        "dataset_suite": suite,
        "selected_ids": selected,
        "packages": package_versions(),
        "hardware": hardware,
    }
    generations = result_dir / f"{args.run_name}_generations.jsonl"
    if saved is not None:
        if saved != metadata:
            raise ValueError("Run identity changed; use a new run_name")
    elif generations.exists():
        raise ValueError("Generations exist without a run manifest")
    else:
        write_json(run_path, metadata)
    count = sum(len(rows) * config["datasets"][name]["n"] for name, rows in datasets.items())
    print(f"{'SMOKE' if args.smoke else 'FULL'} English evaluation: {count} responses", flush=True)
    engine, fallback = create_engine(args.model, config, revision)
    bind_runtime(root, engine.tokenizer, args.stage)
    engine_info = {
        "engine": engine.name,
        "fallback_reason": fallback,
        "chat_template_digest": digest(engine.tokenizer.chat_template),
    }
    engine_path = result_dir / f"{args.run_name}_engine.json"
    if engine_path.exists() and json.loads(engine_path.read_text()) != engine_info:
        raise ValueError("Engine changed during resume")
    write_json(engine_path, engine_info)
    results = evaluate(engine, datasets, config, generations)
    destination = result_dir / f"{args.run_name}.json"
    write_json(destination, {**metadata, **engine_info, "metrics": results})
    print(json.dumps(results, indent=2), flush=True)
    print("Saved:", destination, flush=True)
    if args.smoke:
        from eval.smoke_report import write_smoke_archive

        print(
            "Download for review:", write_smoke_archive(root, args.stage, args.run_name), flush=True
        )


if __name__ == "__main__":
    main()
