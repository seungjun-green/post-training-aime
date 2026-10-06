"""Full-parameter Qwen base SFT on the complete Kimi-style answer column."""

import argparse
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import yaml

from common.io import digest, read_jsonl, write_json, write_jsonl
from train.llama_sft_data import select_smoke_examples
from train.qwen_sft_data import configure_tokenizer, latest_checkpoint, training_code_digest
from train.sft_data import AssistantCollator, load_rows, prepare_examples, sanity_text, percentile
from train.stage1_sft import (
    load_config as load_stage1_config, tokenizer_identity, check_hardware, build_arguments, validate_resume,
)


def load_config(path, smoke=False):
    config = load_stage1_config(path)
    if config["model"]["repo"] != "Qwen/Qwen2.5-3B":
        raise ValueError("This experiment starts from Qwen2.5-3B BASE")
    if (config["data"]["response_format"] != "answer_only"
            or config["data"]["columns"] != {"question": "question", "answer": "kimi-style-reasoning-answer"}):
        raise ValueError("Use only question and the complete kimi-style-reasoning-answer target")
    if config["attention"] != {"implementation": "sdpa", "backend": "flash_attention"}:
        raise ValueError("Use native SDPA with the FlashAttention backend")
    if not config["training"]["bf16"] or config["training"]["fp16"]:
        raise ValueError("Use BF16 full-parameter SFT")
    if type(config["lm_head_chunk_tokens"]) is not int or config["lm_head_chunk_tokens"] < 1:
        raise ValueError("Invalid vocabulary projection chunk size")
    if smoke:
        config["run_name"] += "_smoke"
        config["training"].update(config["smoke"]["training_overrides"])
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--local-tokenizer", help="Pinned tokenizer snapshot; preparation only")
    parser.add_argument("--resume_from_checkpoint", help="Latest completed epoch in this run")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.local_tokenizer and not args.prepare_only:
        parser.error("Local tokenizer is only supported with --prepare-only")
    if args.smoke and args.prepare_only:
        parser.error("Smoke is a separate training run")
    if args.auto_resume and args.resume_from_checkpoint:
        parser.error("Choose automatic or explicit resume")
    config = load_config(args.config, smoke=args.smoke)
    root = Path(config["output_root"])
    log_dir = root / "logs" / config["stage"] / config["run_name"]
    checkpoint_dir = root / "checkpoints" / config["stage"] / config["run_name"]
    from transformers import AutoTokenizer, set_seed

    set_seed(config["training"]["seed"])
    token, model_spec = os.getenv("HF_TOKEN"), config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.local_tokenizer or model_spec["repo"], revision=model_spec["revision"],
        trust_remote_code=model_spec["trust_remote_code"], token=token,
    )
    configure_tokenizer(tokenizer, config)
    rows, ids = load_rows(config)
    examples, report, preview = prepare_examples(tokenizer, rows, ids, config)
    target_lengths = [sum(label != -100 for label in e["labels"]) for e in examples]
    report["supervised_tokens_including_eot"] = {
        "mean": sum(target_lengths) / len(target_lengths), "median": percentile(target_lengths, .5),
        "min": min(target_lengths), "max": max(target_lengths),
    }
    if args.smoke:
        examples, report = select_smoke_examples(examples, report, config)
    original_tokenizer = tokenizer_identity(tokenizer)
    report["tokenizer"] = original_tokenizer
    training = config["training"]
    report["optimizer_steps_per_epoch"] = math.ceil(
        math.ceil(len(examples) / training["per_device_train_batch_size"])
        / training["gradient_accumulation_steps"])
    report["total_optimizer_steps"] = report["optimizer_steps_per_epoch"] * training["num_train_epochs"]
    sanity = sanity_text(tokenizer, preview, config["masked_placeholder"])
    print(sanity, flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "kept_ids"}, indent=2), flush=True)
    if args.prepare_only:
        destination = log_dir / "preparation"
        write_json(destination / "data_report.json", report)
        (destination / "sanity_check.txt").write_text(sanity)
        print("Preparation report:", destination, flush=True)
        return

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from train.qwen_sft_trainer import QwenSFTTrainer

    hardware = check_hardware(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    packages = {name: importlib.metadata.version(name) for name in [
        "torch", "transformers", "trl", "accelerate", "datasets", "huggingface-hub",
        "tokenizers", "numpy", "safetensors",
    ]}
    manifest = {"config": config, "training_code_digest": training_code_digest(),
                "hardware": hardware, "packages": packages, "data_report_digest": digest(report),
                "tokenizer": original_tokenizer, "attention": config["attention"],
                "checkpoint_type": "full_model_with_optimizer_state"}
    identity = digest(manifest)
    resume = args.resume_from_checkpoint
    if args.auto_resume:
        selected = latest_checkpoint(checkpoint_dir, training["num_train_epochs"])
        resume = str(selected) if selected else None
    validate_resume(checkpoint_dir, log_dir, resume, identity)
    if resume:
        marker = json.loads((Path(resume) / "stage1_checkpoint.json").read_text())
        if marker["epoch"] == training["num_train_epochs"]:
            print("All configured epochs are already complete:", resume, flush=True)
            return
    write_json(log_dir / "run_manifest.json", {**manifest, "identity": identity})
    write_json(log_dir / "data_report.json", report)
    (log_dir / "sanity_check.txt").write_text(sanity)
    (log_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    status, trainer = "failed", None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_spec["repo"], revision=model_spec["revision"],
            trust_remote_code=model_spec["trust_remote_code"], token=token,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        model.config.use_cache = False
        # Save the end-of-turn stop in both HF configs, including for vLLM eval.
        for settings in [model.config, model.generation_config]:
            settings.eos_token_id = tokenizer.eos_token_id
            settings.pad_token_id = tokenizer.pad_token_id
        if model.config.model_type != "qwen2" or not all(p.requires_grad for p in model.parameters()):
            raise ValueError("Expected full-parameter native Qwen2 training")
        trainer = QwenSFTTrainer(
            model=model, args=build_arguments(config, checkpoint_dir, log_dir),
            train_dataset=Dataset.from_list(examples), processing_class=tokenizer,
            data_collator=AssistantCollator(tokenizer.pad_token_id), log_dir=log_dir,
            run_identity=identity, lm_head_chunk_tokens=config["lm_head_chunk_tokens"],
        )
        if tokenizer_identity(tokenizer) != original_tokenizer:
            raise ValueError("Trainer changed the configured tokenizer")
        # Prevent a silent quadratic attention fallback on long examples.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result = trainer.train(resume_from_checkpoint=resume)
        if trainer.state.global_step != report["total_optimizer_steps"]:
            raise ValueError("Training did not finish the configured complete epochs")
        for epoch in range(1, int(training["num_train_epochs"]) + 1):
            if not (checkpoint_dir / f"epoch_{epoch}/stage1_checkpoint.json").is_file():
                raise ValueError(f"Missing completed epoch {epoch}")
        write_json(log_dir / "train_metrics.json", result.metrics)
        status = "complete"
    finally:
        attempt = {"status": status, "elapsed_seconds": time.perf_counter() - started,
                   "resume_from_checkpoint": resume,
                   "global_step": trainer.state.global_step if trainer is not None else None,
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
        attempts = read_jsonl(log_dir / "attempts.jsonl", repair_tail=True) + [attempt]
        write_jsonl(log_dir / "attempts.jsonl", attempts)
        write_json(log_dir / "training_summary.json", {
            **attempt, "run_identity": identity,
            "total_training_time_seconds": sum(a["elapsed_seconds"] for a in attempts),
            "peak_gpu_allocated_bytes": max(a["peak_gpu_allocated_bytes"] for a in attempts),
            "peak_gpu_reserved_bytes": max(a["peak_gpu_reserved_bytes"] for a in attempts),
        })


if __name__ == "__main__":
    main()
