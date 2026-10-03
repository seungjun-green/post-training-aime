"""AMC/MATH-only comparison entry point; reuse the existing batched evaluator."""

import argparse
import json
import os
import re
from pathlib import Path

import yaml

from common.io import digest, write_json, write_jsonl
from eval.profiles import bind_profile_protocol, bind_profile_runtime, load_profile
from eval.run_batched_eval import evaluate_batched, runner_digest, validate_execution
from eval.run_english_eval import ROOT, git_identity, load_eval_sets, select_model_revision
from eval.run_eval import package_versions, resolve_model


def comparison_settings(path):
    spec = yaml.safe_load(Path(path).read_text())
    config = yaml.safe_load((ROOT / spec["evaluation_config"]).read_text())
    execution = yaml.safe_load((ROOT / spec["execution_config"]).read_text())
    suite = json.loads((ROOT / spec["suite"]).read_text())
    config, execution = load_profile(spec["profile"], config, execution)
    names = spec["benchmarks"]
    if names != ["amc23", "math_500"] or spec["profile"] != "greedy":
        raise ValueError("This comparison requires AMC 2023 and MATH-500 with the greedy profile")
    # Filter before loading datasets: no AIME download, generation or scoring.
    config["datasets"] = {name: config["datasets"][name] for name in names}
    suite["datasets"] = {name: suite["datasets"][name] for name in names}
    if config["temperature"] != 0 or config["top_p"] != 1 or "budget_forcing" in config:
        raise ValueError("Comparison requires temperature 0, top-p 1 and no budget forcing")
    if config["pass_k"] != [1] or any(d["n"] != 1 for d in config["datasets"].values()):
        raise ValueError("Comparison requires one answer per problem")
    validate_execution(execution)
    return spec, config, execution, suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/dapo_compare_eval.yaml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("Invalid run name")
    _, config, execution, suite = comparison_settings(args.config)
    datasets = load_eval_sets(config, suite, os.getenv("HF_TOKEN"))
    if args.smoke:
        datasets = {name: rows[:1] for name, rows in datasets.items()}
    selected = {name: [row["id"] for row in rows] for name, rows in datasets.items()}
    root = Path(args.output_root) / ("smoke" if args.smoke else "full")
    implementation = digest({"shared": runner_digest(), "subset": Path(__file__).read_text()})
    protocol = bind_profile_protocol(root, "greedy", config, execution, suite,
                                     "comparison", selected, args.smoke, implementation)
    from eval.engines import check_hardware
    from eval.english_engines import create_engine

    result_dir = root / "results"
    manifest_path = result_dir / f"{args.run_name}_manifest.json"
    saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    revision, model_digest = resolve_model(args.model, select_model_revision(args.model, args.revision, saved),
                                           os.getenv("HF_TOKEN"))
    metadata = {
        "model": args.model, "model_revision": revision, "model_digest": model_digest,
        "git_commit": git_identity(), "run_name": args.run_name,
        "mode": "smoke" if args.smoke else "full", "config": config,
        "execution": execution, "protocol_digest": protocol, "dataset_suite": suite,
        "selected_ids": selected, "packages": package_versions(), "hardware": check_hardware(config),
        "runner_digest": implementation,
    }
    if saved is not None:
        if saved != metadata:
            raise ValueError("Comparison run changed; use a new output root or run name")
    elif any(result_dir.glob(f"{args.run_name}*.json*")):
        raise ValueError("Run artifacts exist without a manifest")
    else:
        write_json(manifest_path, metadata)
    print(f"Evaluating {args.run_name}: {sum(map(len, datasets.values()))} problems; "
          "temperature 0, one answer per problem, AMC 2023 and MATH-500 only", flush=True)
    engine, fallback = create_engine(args.model, config, revision)
    if engine.name != "vllm":
        raise ValueError(f"Continuous evaluation requires vLLM: {fallback}")
    bind_profile_runtime(root, engine.tokenizer)
    engine_info = {"engine": engine.name, "chat_template_digest": digest(engine.tokenizer.chat_template)}
    engine_path = result_dir / f"{args.run_name}_engine.json"
    if engine_path.exists() and json.loads(engine_path.read_text()) != engine_info:
        raise ValueError("Engine changed during comparison resume")
    write_json(engine_path, engine_info)
    records, results = evaluate_batched(engine, datasets, config, execution,
                                        result_dir / f"{args.run_name}_problems.jsonl")
    write_jsonl(result_dir / f"{args.run_name}_generations.jsonl", records)
    destination = result_dir / f"{args.run_name}.json"
    write_json(destination, {**metadata, **engine_info, "metrics": results})
    print(json.dumps(results, indent=2), flush=True)
    print("Saved:", destination, flush=True)


if __name__ == "__main__":
    main()
