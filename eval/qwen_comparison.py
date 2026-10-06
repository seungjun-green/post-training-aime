"""Checkpoint validation and result tables for the three Qwen baselines."""

import json
import re
from pathlib import Path

from common.io import write_json
from eval.answer_only import BUNDLE_FILES as EVALUATOR_FILES
from train.dapo_resume import check_checkpoint

BUNDLE_FILES = EVALUATOR_FILES + [
    "eval/qwen_comparison.py", "configs/qwen_compare_eval.yaml",
    "train/__init__.py", "train/dapo_resume.py",
]
LABELS = {"base": "Qwen base", "instruct": "Qwen instruct", "instruct_rl": "Qwen instruct + RL"}
TOTALS = {"amc23": 40, "math_500": 500}


def checkpoint_source(root, run_name, step, instruct):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Invalid RL_RUN_NAME")
    if type(step) is not int or step <= 0:
        raise ValueError("RL_STEP must be a positive integer")
    root = Path(root)
    checkpoint = root / "checkpoints" / run_name / f"checkpoint-{step}"
    marker, _ = check_checkpoint(checkpoint, require_training_state=False)
    manifest = json.loads((root / "logs" / run_name / "run_manifest.json").read_text())
    config = manifest["config"]
    if marker["run_identity"] != manifest["identity"]:
        raise ValueError("RL checkpoint identity does not match the training manifest")
    if (marker["global_step"] != step or config["training"]["max_steps"] != step
            or config["run_name_prefix"] + "_" + config["model_kind"] != run_name):
        raise ValueError("Select the completed final checkpoint of this RL run")
    if any(config["model"][key] != instruct[key] for key in ["repo", "revision"]):
        raise ValueError("RL must start from the pinned Qwen instruct model")
    if json.loads((checkpoint / "config.json").read_text()).get("model_type") != "qwen2":
        raise ValueError("Expected a Qwen2 checkpoint")
    return checkpoint, marker


def prepare_runs(spec, rl_root, rl_run_name, rl_step, output_root):
    checkpoint, marker = checkpoint_source(rl_root, rl_run_name, rl_step, spec["models"]["instruct"])
    runs = [
        {"key": key, "model": entry["repo"], "revision": entry["revision"]}
        for key, entry in spec["models"].items()
    ]
    runs.append({"key": "instruct_rl", "model": str(checkpoint), "revision": None,
                 "checkpoint_marker": marker})
    for run in runs:
        run["run_name"] = "qwen_" + run["key"] + (f"_step{rl_step}" if run["key"] == "instruct_rl" else "")
        run["output_root"] = str(Path(output_root) / run["key"])
    # Never silently replace a comparison's selected checkpoints on resume.
    path = Path(output_root) / "model_selection.json"
    if path.exists() and json.loads(path.read_text()) != runs:
        raise ValueError("Selected models changed; use a new OUTPUT_ROOT")
    write_json(path, runs)
    return runs


def result_path(run, smoke=False):
    return Path(run["output_root"]) / ("smoke" if smoke else "full") / "results" / (run["run_name"] + ".json")


def comparison_rows(runs, config):
    results = []
    for run in runs:
        result = json.loads(result_path(run).read_text())
        if (result["mode"] != "full" or result["model"] != run["model"]
                or result["config"] != config or result["run_name"] != run["run_name"]
                or result["model_revision"] != run["revision"]):
            raise ValueError("Saved result does not match the selected model/settings")
        if results:
            for key in ["execution", "protocol_digest", "dataset_suite", "selected_ids",
                        "packages", "hardware", "runner_digest", "bundle_digest"]:
                if result[key] != results[0][key]:
                    raise ValueError(f"Comparison changed {key}")
        results.append(result)
    if results[1]["chat_template_digest"] != results[2]["chat_template_digest"]:
        raise ValueError("Instruct and instruct + RL must use the same native chat template")
    rows = []
    for run, result in zip(runs, results, strict=True):
        for benchmark, total in TOTALS.items():
            metric = result["metrics"][benchmark]
            if (metric["problems"] != total or metric["responses"] != total
                    or metric["samples_per_problem"] != 1):
                raise ValueError(f"Incomplete full evaluation: {run['key']} / {benchmark}")
            rows.append({"Model": LABELS[run["key"]],
                         "Benchmark": "AMC 2023" if benchmark == "amc23" else "MATH-500",
                         "Problems": total, "Correct": round(metric["avg@1"] * total),
                         "Accuracy (%)": 100 * metric["avg@1"],
                         "Mean tokens": metric["response_length_tokens"]["all"]})
    return rows
