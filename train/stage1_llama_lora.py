"""Llama 3.1 BF16 LoRA SFT, isolated from the existing EXAONE training path."""

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from common.io import digest, read_jsonl, write_json, write_jsonl
from eval.run_english_eval import code_fingerprint, git_identity
from train.sft_data import AssistantCollator, load_rows, sanity_text
from train.stage1_sft import (
    load_config as load_stage1_config, tokenizer_identity, check_hardware,
    build_arguments, validate_resume,
)
from train.llama_sft_data import (
    configure_tokenizer, prepare_llama_examples, select_smoke_examples,
)


def load_config(path, smoke=False):
    config = load_stage1_config(path)
    if config["model"]["repo"] != "meta-llama/Llama-3.1-8B-Instruct":
        raise ValueError("This entry point requires Llama 3.1 8B Instruct")
    if config["attention"] != {"implementation": "sdpa", "backend": "flash_attention"}:
        raise ValueError("Use native SDPA with the FlashAttention backend")
    if not config["training"]["bf16"] or config["training"]["fp16"]:
        raise ValueError("Use BF16 base weights with unquantized LoRA")
    if config["lora"]["bias"] != "none" or config["lora"]["modules_to_save"] is not None:
        raise ValueError("Only LoRA adapter matrices should be trainable")
    if smoke:
        config["run_name"] += "_smoke"
        config["training"].update(config["smoke"]["training_overrides"])
    return config


def attach_lora(model, config):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        **config["lora"], revision=config["model"]["revision"],
    ))
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_A." not in n and "lora_B." not in n for n in trainable):
        raise ValueError("Unexpected trainable base-model parameters")
    targeted = {name for name, module in model.named_modules() if hasattr(module, "lora_A")}
    expected = config["lora"]["target_modules"]
    for suffix in expected:
        if sum(name.endswith("." + suffix) for name in targeted) != model.config.num_hidden_layers:
            raise ValueError(f"LoRA did not cover every {suffix} layer")
    report = {
        "base_model": config["model"], "lora": config["lora"],
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_dtypes": sorted({str(p.dtype) for p in trainable.values()}),
        "targeted_modules": sorted(targeted),
        "checkpoint_type": "peft_adapter_with_optimizer_state",
    }
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_root", help="Storage location only; hyperparameters stay in config")
    parser.add_argument("--prepare-only", action="store_true", help="Tokenize/report without loading weights")
    parser.add_argument("--local-data", help="Verified local decontamination JSONL; prepare-only")
    parser.add_argument("--local-tokenizer", help="Pinned tokenizer snapshot; prepare-only")
    parser.add_argument("--resume_from_checkpoint", help="Latest completed epoch directory")
    parser.add_argument("--smoke", action="store_true", help="Train the longest retained example in a separate run")
    args = parser.parse_args()
    if (args.local_data or args.local_tokenizer) and not args.prepare_only:
        parser.error("Local inspection inputs are only supported with --prepare-only")
    if args.smoke and (args.prepare_only or args.resume_from_checkpoint):
        parser.error("Smoke is a fresh training run, separate from preparation and resume")
    config = load_config(args.config, smoke=args.smoke)
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
    configure_tokenizer(tokenizer, config)
    rows, ids = load_rows(config, token, args.local_data)
    examples, report, preview = prepare_llama_examples(tokenizer, rows, ids, config)
    if args.smoke:
        examples, report = select_smoke_examples(examples, report, config)
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
    from transformers import AutoModelForCausalLM
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from train.sft_trainer import Stage1Trainer

    hardware = check_hardware(config)
    commit = git_identity()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    packages = {name: importlib.metadata.version(name) for name in [
        "torch", "transformers", "trl", "accelerate", "datasets", "huggingface-hub",
        "tokenizers", "numpy", "safetensors", "peft",
    ]}
    attention = config["attention"]["implementation"]
    manifest = {
        "config": config, "git_commit": commit, "hardware": hardware, "packages": packages,
        "attention": config["attention"],
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
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model, adapter_report = attach_lora(model, config)
        write_json(log_dir / "adapter_report.json", adapter_report)
        model.print_trainable_parameters()
        trainer = Stage1Trainer(
            model=model, args=build_arguments(config, checkpoint_dir, log_dir),
            train_dataset=Dataset.from_list(examples), processing_class=tokenizer,
            data_collator=AssistantCollator(tokenizer.pad_token_id),
            log_dir=log_dir, run_identity=identity,
        )
        if tokenizer_identity(tokenizer) != original_tokenizer:
            raise ValueError("Trainer changed the frozen tokenizer")
        # Force the native PyTorch FlashAttention kernel; never silently allocate
        # a quadratic attention matrix for a 20K-token sequence.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
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
