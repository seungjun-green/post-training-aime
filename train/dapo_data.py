"""Checked inputs and deterministic sampling for the EXAONE DAPO experiment."""

import json
import random
import re
from copy import deepcopy
from pathlib import Path

import yaml

from common.io import digest
from pipeline.datasets import core_dapo_problem
from train.dapo_model import training_prompt


def batch_schedule(config):
    a = config["algorithm"]
    count = a["group_size"] * a["retained_groups"]
    mini = a.get("mini_batch_size", count)
    iterations = a.get("num_iterations", 1)
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("algorithm.num_iterations must be a positive integer")
    if type(mini) is not int or mini <= 0 or count % mini:
        raise ValueError("mini_batch_size must be a positive divisor of the rollout response count")
    return {"rollout_batch_size": count, "mini_batch_size": mini,
            "minibatches_per_rollout": count // mini,
            "updates_per_rollout": count // mini * iterations}


def load_config(path, smoke=False):
    config = yaml.safe_load(Path(path).read_text())
    a, t, r = config["algorithm"], config["training"], config["rollout"]
    if smoke:
        for key in ["retained_groups", "candidate_groups_per_batch", "max_generation_batches"]:
            a[key] = config["smoke"][key]
        for key in ["max_steps", "save_steps", "warmup_steps"]:
            t[key] = config["smoke"][key]
        if "mini_batch_size" in a:
            a["mini_batch_size"] = config["smoke"]["mini_batch_size"]
    for value in [a["group_size"], a["retained_groups"], a["candidate_groups_per_batch"],
                  a["max_generation_batches"], t["max_steps"], t["save_steps"], a["logprob_chunk_tokens"]]:
        if type(value) is not int or value <= 0:
            raise ValueError("Batch sizes, retry limits and step counts must be positive integers")
    cycle = batch_schedule(config)["updates_per_rollout"]
    if t["max_steps"] % cycle or t["save_steps"] % cycle:
        raise ValueError("max_steps and save_steps must end on a complete rollout cycle")
    if a["group_size"] < 2 or not 0 <= a["soft_length_limit"] < a["max_completion_length"]:
        raise ValueError("Invalid group size or length penalty interval")
    if config["data"]["max_prompt_tokens"] + a["max_completion_length"] > r["max_model_len"]:
        raise ValueError("Prompt and completion budgets exceed the model context")
    if (a["loss_type"] != "dapo" or a["scale_rewards"] != "group" or a["beta"] != 0
            or a["mask_truncated_completions"] or t["per_device_train_batch_size"] != 1
            or a["temperature"] != 1 or a["top_p"] != 1):
        raise ValueError("This tested DAPO path requires token loss, group scaling, no KL, unmasked shaping, microbatch 1, T=top_p=1")
    if not re.fullmatch(r"[a-f0-9]{40}", config["model"]["revision"]):
        raise ValueError("Pin the model revision to a full commit hash")
    if not t["bf16"] or t["fp16"] or r["tensor_parallel_size"] != 1:
        raise ValueError("This experiment uses one GPU and BF16 compute")
    return config


def select_model(config, kind, sft_root):
    if kind == "base":
        return config["model"]["repo"], {"kind": "base", **config["model"]}
    if kind != "sft":
        raise ValueError("MODEL_KIND must be 'base' or 'sft'")
    if "sft" not in config:
        raise ValueError("This config starts from the instruction model; no SFT source is configured")
    path = Path(sft_root) / config["sft"]["relative_checkpoint"]
    marker = json.loads((path / "stage1_checkpoint.json").read_text())
    if marker["epoch"] != config["sft"]["epoch"]:
        raise ValueError("Expected the original SFT epoch 5")
    manifest_path = Path(sft_root) / "logs/stage1" / config["sft"]["run_name"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    previous = manifest["config"]
    from train.sft_data import DEFAULT_COLUMNS
    if (manifest["identity"] != marker["run_identity"]
            or previous["run_name"] != config["sft"]["run_name"]
            or previous["model"] != config["model"]
            or previous["data"].get("columns", DEFAULT_COLUMNS) != DEFAULT_COLUMNS):
        raise ValueError("Checkpoint must be the original EXAONE SFT with original DeepSeek columns")
    if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
        raise ValueError(f"Incomplete SFT model: {path}")
    return str(path), {"kind": "sft", "path": str(path), "marker": marker}


def verify_rows(rows, ids, spec):
    if len(rows) != spec["expected_rows"] or digest(rows) != spec["content_digest"]:
        raise ValueError("DAPO rows differ from the decontaminated English export")
    if len(ids) != len(rows) or len(set(map(str, ids))) != len(ids) or digest(ids) != spec["ids_digest"]:
        raise ValueError("DAPO IDs differ from the decontamination provenance")


def load_data(config, token=None, local_data=None):
    spec = config["data"]
    if local_data:
        envelopes = [json.loads(line) for line in Path(local_data).read_text().splitlines()]
        rows, ids = [r["original"] for r in envelopes], [r["id"] for r in envelopes]
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi(token=token).dataset_info(spec["repo"], revision=spec["revision"]).sha
        spec["revision"] = revision
        rows = list(load_dataset(spec["repo"], name=spec["config"], split=spec["split"],
                                 revision=revision, token=token))
        provenance = json.loads(Path(hf_hub_download(spec["repo"], "provenance.json", repo_type="dataset",
                                                     revision=revision, token=token)).read_text())
        if provenance["retained_originals_digest"] != spec["content_digest"]:
            raise ValueError("DAPO provenance does not match the configured export")
        ids = provenance["retained_preparation_ids"]
    verify_rows(rows, ids, spec)
    return rows, ids


def prepare_data(rows, ids, tokenizer, config):
    prepared, dropped = [], []
    for row, identifier in zip(rows, ids, strict=True):
        problem = core_dapo_problem(row["prompt"])
        answer = str(row["solution"]).strip()
        if not re.fullmatch(r"[+-]?\d+", answer):
            raise ValueError(f"Expected a final integer answer in DAPO/{identifier}")
        prompt = training_prompt(tokenizer, problem, config)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > config["data"]["max_prompt_tokens"]:
            dropped.append({"id": identifier, "prompt_tokens": len(token_ids)})
            continue
        prepared.append({"id": str(identifier), "prompt": prompt, "prompt_token_ids": token_ids,
                         "gold": answer})
    if not prepared:
        raise ValueError("No DAPO questions fit the prompt budget")
    report = {"dataset": deepcopy(config["data"]), "input_rows": len(rows), "kept_rows": len(prepared),
              "dropped_prompts": dropped, "truncation_applied": False, "prepared_digest": digest(prepared),
              "min_prompt_tokens": min(len(r["prompt_token_ids"]) for r in prepared),
              "max_prompt_tokens": max(len(r["prompt_token_ids"]) for r in prepared)}
    if "prompt_template_kwargs" in config:
        report["prompt_template_kwargs"] = deepcopy(config["prompt_template_kwargs"])
    return prepared, report


def candidate_order(rows, seed, step):
    # Entire dataset is eligible each update. No mutable sampler state is needed on resume.
    indices = list(range(len(rows)))
    random.Random(f"{seed}:{step}").shuffle(indices)
    return [rows[i] for i in indices]
