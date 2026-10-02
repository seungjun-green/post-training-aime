"""Full-parameter English s1-style SFT. Run from the repository with --config."""

import argparse
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from common.io import digest, read_jsonl, write_json, write_jsonl  # noqa: E402
from eval.run_english_eval import code_fingerprint, git_identity  # noqa: E402
from train.sft_data import (  # noqa: E402
    AssistantCollator,
    load_rows,
    prepare_examples,
    sanity_text,
)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    # Resolve a published Pro dataset once; all subsequent reads and the saved run
    # manifest use this immutable commit, never the moving branch name.
    if config["data"]["revision"] == "main":
        from huggingface_hub import HfApi

        config["data"]["revision"] = HfApi().dataset_info(
            config["data"]["repo"], revision="main"
        ).sha
    for field in ["stage", "run_name"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", config[field]):
            raise ValueError(f"Unsafe {field}")
    for section in ["model", "data"]:
        if not re.fullmatch(r"[a-f0-9]{40}", config[section]["revision"]):
            raise ValueError(f"Pin {section} to an immutable revision")
    training = config["training"]
    if training["packing"] or training["max_length"] is not None:
        raise ValueError("Packing and TRL truncation must remain disabled")
    if training["dataset_kwargs"] != {"skip_prepare_dataset": True}:
        raise ValueError("TRL must preserve precomputed input IDs and labels")
    if training["save_strategy"] != "epoch" or training["save_total_limit"] is not None:
        raise ValueError("Keep every epoch checkpoint")
    if training["eval_strategy"] != "no" or training["load_best_model_at_end"]:
        raise ValueError("No validation split or checkpoint selection")
    if training["num_train_epochs"] != int(training["num_train_epochs"]):
        raise ValueError("Training must finish complete epochs")
    return config


def tokenizer_identity(tokenizer):
    return {
        "chat_template_digest": digest(tokenizer.chat_template),
        "tokenizer_vocab_digest": digest(tokenizer.get_vocab()),
        "special_tokens": tokenizer.special_tokens_map,
    }


def check_hardware(config):
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage 1 requires exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    if (config["hardware"]["name_contains"] not in gpu.name
            or gpu.total_memory < config["hardware"]["min_memory_gib"] * 1024**3):
        raise RuntimeError("Stage 1 requires the RTX PRO 6000 Blackwell 96GB")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")
    return {"name": gpu.name, "memory_bytes": gpu.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)), "cuda": torch.version.cuda}


def choose_attention(config, model_config):
    import torch

    reasons = []
    for backend in config["attention"]["preference"]:
        if backend == "sdpa":
            return backend, reasons
        if backend != "flash_attention_2":
            raise ValueError(f"Unknown attention backend: {backend}")
        try:
            from flash_attn import flash_attn_func

            shape = (1, config["attention"]["probe_sequence_length"],
                     model_config.num_attention_heads,
                     model_config.hidden_size // model_config.num_attention_heads)
            q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16,
                                   requires_grad=True) for _ in range(3)]
            flash_attn_func(q, k, v, causal=True).float().sum().backward()
            torch.cuda.synchronize()
            return backend, reasons
        except torch.cuda.OutOfMemoryError:
            raise
        except (ImportError, RuntimeError, OSError) as exc:
            reasons.append(f"FlashAttention-2 unavailable or failed forward/backward probe: {exc}")
    raise RuntimeError("No attention backend available")


def build_arguments(config, checkpoint_dir, log_dir):
    from trl import SFTConfig

    return SFTConfig(output_dir=str(checkpoint_dir), logging_dir=str(log_dir), **config["training"])


def validate_resume(checkpoint_dir, log_dir, resume, identity):
    manifest_path = log_dir / "run_manifest.json"
    if not resume:
        if manifest_path.exists() or any(checkpoint_dir.iterdir()):
            raise ValueError("Existing run: explicitly resume the latest complete epoch or use a new run")
        return
    path = Path(resume).resolve()
    completed = sorted(checkpoint_dir.glob("epoch_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not completed or path != completed[-1].resolve():
        raise ValueError("Resume only the latest complete epoch in this run")
    previous = json.loads(manifest_path.read_text())
    if previous["identity"] != identity:
        raise ValueError("Training config, code, dataset, tokenizer or runtime changed")
    marker = json.loads((path / "stage1_checkpoint.json").read_text())
    if marker["run_identity"] != identity:
        raise ValueError("Checkpoint belongs to another run")
    records = read_jsonl(log_dir / "steps.jsonl", repair_tail=True)
    abandoned = [r for r in records if r["step"] > marker["global_step"]]
    if abandoned:
        write_jsonl(log_dir / f"abandoned_steps_{time.time_ns()}.jsonl", abandoned)
        write_jsonl(log_dir / "steps.jsonl", [r for r in records if r["step"] <= marker["global_step"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_root", help="Storage location only; hyperparameters stay in config")
    parser.add_argument("--prepare-only", action="store_true", help="Tokenize/report without loading weights")
    parser.add_argument("--local-data", help="Verified local decontamination JSONL; prepare-only")
    parser.add_argument("--local-tokenizer", help="Pinned tokenizer snapshot; prepare-only")
    parser.add_argument("--resume_from_checkpoint", help="Latest completed epoch directory")
    args = parser.parse_args()
    if (args.local_data or args.local_tokenizer) and not args.prepare_only:
        parser.error("Local inspection inputs are only supported with --prepare-only")
    config = load_config(args.config)
    root = Path(args.output_root or config["output_root"]).resolve()
    log_dir = root / "logs" / config["stage"] / config["run_name"]
    checkpoint_dir = root / "checkpoints" / config["stage"] / config["run_name"]
    from transformers import AutoTokenizer, set_seed

    set_seed(config["training"]["seed"])
    token = os.getenv("HF_TOKEN")
    model_spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.local_tokenizer or model_spec["repo"], revision=model_spec["revision"],
        trust_remote_code=model_spec["trust_remote_code"], token=token,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Expected the unchanged EXAONE tokenizer with a native padding token")
    rows, ids = load_rows(config, token, args.local_data)
    examples, report, preview = prepare_examples(tokenizer, rows, ids, config)
    original_tokenizer = tokenizer_identity(tokenizer)
    report["tokenizer"] = original_tokenizer
    batch = config["training"]["per_device_train_batch_size"]
    accumulation = config["training"]["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(math.ceil(len(examples) / batch) / accumulation)
    report["optimizer_steps_per_epoch"] = steps_per_epoch
    report["total_optimizer_steps"] = steps_per_epoch * config["training"]["num_train_epochs"]
    sanity = sanity_text(tokenizer, preview, config["masked_placeholder"])
    print(sanity, flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "kept_ids"}, indent=2), flush=True)
    if args.prepare_only:
        # Preparation never overwrites training logs or claims a completed training run.
        destination = log_dir / "preparation"
        write_json(destination / "data_report.json", report)
        (destination / "sanity_check.txt").write_text(sanity)
        print("Preparation report:", destination, flush=True)
        return

    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM

    from train.sft_trainer import Stage1Trainer

    hardware = check_hardware(config)
    commit = git_identity()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    packages = {name: importlib.metadata.version(name) for name in [
        "torch", "transformers", "trl", "accelerate", "datasets", "huggingface-hub",
        "tokenizers", "numpy", "safetensors",
    ]}
    model_config = AutoConfig.from_pretrained(
        model_spec["repo"], revision=model_spec["revision"],
        trust_remote_code=model_spec["trust_remote_code"], token=token,
    )
    attention, attention_notes = choose_attention(config, model_config)
    manifest = {
        "config": config, "git_commit": commit, "hardware": hardware, "packages": packages,
        "attention": attention, "attention_notes": attention_notes,
        "data_report_digest": digest(report), "frozen_eval_code_digest": code_fingerprint(),
        "tokenizer": original_tokenizer,
    }
    identity = digest(manifest)
    validate_resume(checkpoint_dir, log_dir, args.resume_from_checkpoint, identity)
    write_json(log_dir / "run_manifest.json", {**manifest, "identity": identity})
    write_json(log_dir / "data_report.json", report)
    (log_dir / "sanity_check.txt").write_text(sanity)
    (log_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    status = "failed"
    trainer = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_spec["repo"], revision=model_spec["revision"],
            trust_remote_code=model_spec["trust_remote_code"], token=token,
            torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float32,
            attn_implementation=attention,
        )
        model.config.use_cache = False
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError("All parameters must be trainable; no adapters or frozen parameters")
        trainer = Stage1Trainer(
            model=model, args=build_arguments(config, checkpoint_dir, log_dir),
            train_dataset=Dataset.from_list(examples), processing_class=tokenizer,
            data_collator=AssistantCollator(tokenizer.pad_token_id),
            log_dir=log_dir, run_identity=identity,
        )
        if tokenizer_identity(tokenizer) != original_tokenizer:
            raise ValueError("Trainer changed the frozen tokenizer")
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        if trainer.state.global_step != report["total_optimizer_steps"]:
            raise ValueError("Training did not finish the configured complete epochs")
        for epoch in range(1, int(config["training"]["num_train_epochs"]) + 1):
            if not (checkpoint_dir / f"epoch_{epoch}/stage1_checkpoint.json").is_file():
                raise ValueError(f"Missing completed epoch {epoch}")
        write_json(log_dir / "train_metrics.json", train_result.metrics)
        status = "complete"
    finally:
        elapsed = time.perf_counter() - started
        attempt = {
            "status": status, "elapsed_seconds": elapsed,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "global_step": trainer.state.global_step if trainer is not None else None,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        attempts = read_jsonl(log_dir / "attempts.jsonl", repair_tail=True) + [attempt]
        write_jsonl(log_dir / "attempts.jsonl", attempts)
        write_json(log_dir / "training_summary.json", {
            **attempt, "run_identity": identity,
            "total_training_time_seconds": sum(a["elapsed_seconds"] for a in attempts),
            "peak_gpu_allocated_bytes": max(a["peak_gpu_allocated_bytes"] for a in attempts),
            "peak_gpu_reserved_bytes": max(a["peak_gpu_reserved_bytes"] for a in attempts),
            "timing_scope": "model loading, training and epoch saves across attempts",
        })


if __name__ == "__main__":
    main()
