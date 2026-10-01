"""Full English evaluation with continuous batching and explicit legacy-run reuse."""

import argparse
import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import monotonic

import yaml

from common.english_prompts import render_prompt
from common.io import append_jsonl, digest, read_jsonl, write_json, write_jsonl
from eval.batched_generation import generate_problems
from eval.profiles import (
    PROFILE_NAMES,
    bind_profile_protocol,
    bind_profile_runtime,
    load_profile,
    profile_root,
)
from eval.run_english_eval import (
    ROOT,
    bind_protocol,
    git_identity,
    load_eval_sets,
    select_model_revision,
)
from eval.run_eval import (
    bind_runtime,
    package_versions,
    problem_seed,
    resolve_model,
    validate_config,
)
from eval.scoring import korean_ratio, metrics, score_response


def validate_execution(execution):
    if set(execution) != {"version", "max_pending_problems", "status_interval_seconds"}:
        raise ValueError("Unexpected execution configuration keys")
    if execution["version"] != "continuous-v1":
        raise ValueError("Unsupported execution version")
    for key in ["max_pending_problems", "status_interval_seconds"]:
        if type(execution[key]) is not int or execution[key] < 1:
            raise ValueError(f"{key} must be a positive integer")


def runner_digest():
    return digest({name: (ROOT / name).read_text() for name in [
        "eval/run_batched_eval.py", "eval/batched_generation.py", "eval/profiles.py",
    ]})


def read_source_records(path):
    """Read an interrupted legacy journal without modifying even its incomplete tail."""
    if not path.exists():
        return []
    records = []
    with path.open("rb") as stream:
        while line := stream.readline():
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not line.endswith(b"\n") and not stream.read(1):
                    break
                raise ValueError(f"Corrupt legacy journal: {path}") from None
    return records


def complete_groups(records, datasets, config, *, allow_partial=False):
    expected = {(name, str(row["id"])): row
                for name, rows in datasets.items() for row in rows}
    groups = defaultdict(dict)
    for record in records:
        key = (record["dataset"], str(record["id"]))
        if key not in expected:
            raise ValueError(f"Unknown saved problem: {key}")
        spec = config["datasets"][key[0]]
        index = record["sample_index"]
        if type(index) is not int or not 0 <= index < spec["n"] or index in groups[key]:
            raise ValueError(f"Duplicate or invalid sample: {key}/{index}")
        if record["problem_seed"] != problem_seed(config["seed"], *key):
            raise ValueError(f"Saved seed mismatch: {key}")
        if record["answer"] != expected[key][spec["answer_column"]]:
            raise ValueError(f"Saved answer mismatch: {key}")
        if type(record["correct"]) is not bool:
            raise ValueError(f"Invalid saved score: {key}")
        groups[key][index] = record
    result = {}
    for key, samples in groups.items():
        n = config["datasets"][key[0]]["n"]
        if len(samples) != n:
            if allow_partial:
                continue
            raise ValueError(f"Incomplete committed problem: {key}")
        result[key] = [samples[i] for i in range(n)]
    return result


def legacy_source(result_dir, name, metadata, datasets, config):
    """Accept completed problems only from the same model and scientific protocol."""
    if name is None:
        return {}, None
    manifest_path = result_dir / f"{name}_manifest.json"
    journal = result_dir / f"{name}_generations.jsonl"
    if not manifest_path.exists():
        if journal.exists():
            raise ValueError("Legacy generations exist without their manifest")
        return {}, None
    manifest = json.loads(manifest_path.read_text())
    for field in ["model", "model_revision", "model_digest", "stage", "mode", "config",
                  "protocol_digest", "dataset_suite", "selected_ids", "packages", "hardware"]:
        if manifest.get(field) != metadata[field]:
            raise ValueError(f"Legacy run mismatch: {field}")
    if manifest.get("run_name") != name or "execution" in manifest:
        raise ValueError("Reuse requires an original sequential full-evaluation manifest")
    records = read_source_records(journal)
    groups = complete_groups(records, datasets, config, allow_partial=True)
    engine_path = result_dir / f"{name}_engine.json"
    engine_info = json.loads(engine_path.read_text()) if engine_path.exists() else None
    if records and (engine_info is None or engine_info.get("engine") != "vllm"):
        raise ValueError("Legacy generations must have a recorded vLLM engine")
    return groups, {
        "run_name": name, "manifest": manifest, "engine": engine_info,
        "completed_problems": len(groups),
        "records_digest": digest([r for group in groups.values() for r in group]),
        "ignored_incomplete_responses": len(records) - sum(map(len, groups.values())),
    }


def evaluate_batched(engine, datasets, config, execution, journal, imported=None):
    # Each line is one whole scored problem. A kill during append loses at most
    # that problem; it never forces a stochastic replay to match partial samples.
    entries = read_jsonl(journal, repair_tail=True)
    records = [record for entry in entries for record in entry["records"]]
    completed = complete_groups(records, datasets, config)
    expected = {(name, str(row["id"])): row
                for name, rows in datasets.items() for row in rows}
    imported = imported or {}
    for key, group in {**completed, **imported}.items():
        spec = config["datasets"][key[0]]
        prompt = render_prompt(engine.tokenizer, expected[key][spec["problem_column"]])
        if any(record["prompt"] != prompt for record in group):
            raise ValueError(f"Saved prompt mismatch: {key}")
    for key, group in imported.items():
        if key in completed:
            if completed[key] != group:
                raise ValueError(f"Imported problem differs from saved copy: {key}")
            continue
        append_jsonl(journal, {"origin": "legacy", "records": group})
        completed[key] = group
        records.extend(group)
    for name, rows in datasets.items():
        spec = config["datasets"][name]
        done = sum((name, str(row["id"])) in completed for row in rows)
        print(f"{name}: {done}/{len(rows)} problems", flush=True)
        jobs = [
            {"key": (name, str(row["id"])), "problem": row[spec["problem_column"]],
             "n": spec["n"], "seed": problem_seed(config["seed"], name, row["id"])}
            for row in rows if (name, str(row["id"])) not in completed
        ]
        started, new_tokens = monotonic(), 0
        stream = generate_problems(engine, jobs, execution)
        try:
            for job, prompt, responses in stream:
                key = job["key"]
                if key in completed or key not in expected or len(responses) != spec["n"]:
                    raise ValueError("Unexpected generated problem or response count")
                row = expected[key]
                group = []
                for index, response in enumerate(responses):
                    extracted, correct = score_response(
                        response["text"], row[spec["answer_column"]], config["verify_timeout_seconds"]
                    )
                    group.append({
                        "dataset": name, "id": row["id"], "sample_index": index,
                        "problem_seed": job["seed"], "prompt": prompt,
                        "response": response["text"], "answer": row[spec["answer_column"]],
                        "extracted_answer": extracted, "correct": correct,
                        "token_count": response["token_count"],
                        "finish_reason": response["finish_reason"],
                        "korean_response_ratio": korean_ratio(response["text"]),
                    })
                append_jsonl(journal, {"origin": "continuous-v1", "records": group})
                completed[key] = group
                records.extend(group)
                done += 1
                new_tokens += sum(r["token_count"] for r in group)
                print(f"{name}: {done}/{len(rows)} problems", flush=True)
                print(f"Saved {name}/{row['id']}; completed-response throughput "
                      f"{new_tokens / max(monotonic() - started, 1e-9):.1f} tokens/s "
                      "(includes scoring and saving)", flush=True)
        finally:
            stream.close()
    # Stable export order, independent of scheduler completion order.
    ordered = [record for key in expected for record in completed[key]]
    return ordered, {
        name: metrics([r for r in ordered if r["dataset"] == name], spec["n"], config["pass_k"])
        for name, spec in config["datasets"].items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--reuse_run_name", help="Copy only complete problems from this legacy run")
    parser.add_argument("--config", default=str(ROOT / "configs/eval_english.yaml"))
    parser.add_argument("--suite", default=str(ROOT / "configs/english_eval_suite.json"))
    parser.add_argument("--execution_config", default=str(ROOT / "configs/eval_execution.yaml"))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--profile", choices=PROFILE_NAMES,
                        help="Named profile; stores runs under output_root/profiles/PROFILE")
    parser.add_argument("--smoke", action="store_true",
                        help="With --profile: one problem/answer per benchmark, isolated from full runs")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for name in [args.stage, args.run_name, args.reuse_run_name]:
        if name is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            parser.error("stage/run names must use letters, digits, underscores or hyphens")
    if args.run_name == args.reuse_run_name:
        parser.error("Use a separate batched run name; legacy files are never overwritten")
    if args.profile and args.reuse_run_name:
        parser.error("New profiles cannot reuse legacy responses with different sampling settings")
    if args.smoke and not args.profile:
        parser.error("--smoke requires a named --profile")
    config = yaml.safe_load(Path(args.config).read_text())
    execution = yaml.safe_load(Path(args.execution_config).read_text())
    suite = json.loads(Path(args.suite).read_text())
    if args.profile:
        config, execution = load_profile(args.profile, config, execution)
    else:
        validate_config(config)
    validate_execution(execution)
    token = os.getenv("HF_TOKEN")
    datasets = load_eval_sets(config, suite, token)
    if args.smoke:
        config = deepcopy(config)
        config["pass_k"] = [1]
        datasets = {name: rows[:1] for name, rows in datasets.items()}
        for spec in config["datasets"].values():
            spec["n"] = 1
    selected = {name: [r["id"] for r in rows] for name, rows in datasets.items()}
    parent = profile_root(args.output_root, args.profile) if args.profile else Path(args.output_root)
    root = parent / ("smoke" if args.smoke else "full")
    if args.profile:
        protocol_id = bind_profile_protocol(
            root, args.profile, config, execution, suite, args.stage, selected, args.smoke,
            runner_digest(), create=not args.validate_only,
        )
    else:
        # Legacy path retains the original scientific protocol and separate runner identity.
        protocol_id = bind_protocol(root, config, suite, "stage1", selected, False)
    if args.validate_only:
        print("Validated datasets, evaluation protocol, and execution settings", flush=True)
        return
    from eval.engines import check_hardware
    from eval.english_engines import create_engine

    commit = git_identity()
    result_dir = root / "results" / args.stage
    manifest_path = result_dir / f"{args.run_name}_manifest.json"
    saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    revision, model_digest = resolve_model(
        args.model, select_model_revision(args.model, args.revision, saved), token,
    )
    metadata = {
        "model": args.model, "model_revision": revision, "model_digest": model_digest,
        "git_commit": commit, "stage": args.stage, "run_name": args.run_name,
        "mode": "smoke" if args.smoke else "full",
        "config": config, "protocol_digest": protocol_id, "dataset_suite": suite,
        "selected_ids": selected, "packages": package_versions(), "hardware": check_hardware(config),
        "execution": execution, "runner_digest": runner_digest(),
        "reuse_run_name": args.reuse_run_name,
    }
    if args.profile:
        metadata["profile"] = args.profile
    imported, source = legacy_source(result_dir, args.reuse_run_name, metadata, datasets, config)
    metadata["legacy_source"] = source
    journal = result_dir / f"{args.run_name}_problems.jsonl"
    if saved is not None:
        if saved != metadata:
            raise ValueError("Batched run identity/source changed; use a new run_name")
    elif any(result_dir.glob(f"{args.run_name}_*.json*")) or (result_dir / f"{args.run_name}.json").exists():
        raise ValueError("Batched run artifacts exist without a manifest")
    else:
        write_json(manifest_path, metadata)
    print(f"Continuous evaluation: {sum(map(len, datasets.values()))} problems, "
          f"{sum(len(rows) * config['datasets'][name]['n'] for name, rows in datasets.items())} responses; "
          f"up to {execution['max_pending_problems']} queued problems, "
          f"{config['max_num_seqs']} GPU responses", flush=True)
    print(f"Reusing {len(imported)} complete legacy problems; progress journal: {journal}", flush=True)
    engine, fallback = create_engine(args.model, config, revision)
    if engine.name != "vllm":
        raise ValueError(f"Continuous evaluation needs vLLM; unsupported model: {fallback}")
    if args.profile:
        bind_profile_runtime(root, engine.tokenizer)
    else:
        bind_runtime(root, engine.tokenizer, "stage1")
    engine_info = {
        "engine": engine.name, "fallback_reason": fallback,
        "chat_template_digest": digest(engine.tokenizer.chat_template),
    }
    if source and source["engine"] is not None and source["engine"] != engine_info:
        raise ValueError("Legacy engine/tokenizer mismatch")
    engine_path = result_dir / f"{args.run_name}_engine.json"
    if engine_path.exists() and json.loads(engine_path.read_text()) != engine_info:
        raise ValueError("Engine changed during batched resume")
    write_json(engine_path, engine_info)
    records, results = evaluate_batched(engine, datasets, config, execution, journal, imported)
    write_jsonl(result_dir / f"{args.run_name}_generations.jsonl", records)
    destination = result_dir / f"{args.run_name}.json"
    write_json(destination, {**metadata, **engine_info, "metrics": results})
    print(json.dumps(results, indent=2), flush=True)
    print("Saved:", destination, flush=True)
    if args.smoke:
        from eval.smoke_report import write_smoke_archive

        print("Smoke archive:", write_smoke_archive(root, args.stage, args.run_name), flush=True)


if __name__ == "__main__":
    main()
