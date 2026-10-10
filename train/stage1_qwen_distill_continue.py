"""Continue distillation from epoch 5 through epochs 6–7 in a separate run."""

import argparse
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import yaml

from common.io import digest, read_jsonl, write_json, write_jsonl
from train.qwen_self_rft_data import configure_tokenizer, prepare_examples
from train.qwen_distill_continuation import (
    training_code_digest, validate_config, validate_prepared_report, select_resume,
)
from train.sft_data import AssistantCollator, load_rows, sanity_text, percentile
from train.stage1_sft import (
    tokenizer_identity, check_hardware, build_arguments,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--local-tokenizer", help="Pinned tokenizer snapshot; preparation only")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Save epoch 6 or 7 and exit; retain the two-additional-epoch LR schedule")
    args = parser.parse_args()
    if args.local_tokenizer and not args.prepare_only:
        parser.error("Local tokenizer is only supported with --prepare-only")
    config, parent_manifest, parent_report = validate_config(args.config)
    if args.stop_after_epoch is not None and (args.prepare_only or args.stop_after_epoch not in (6, 7)):
        parser.error("--stop-after-epoch must be 6 or 7")
    root = Path(config["output_root"])
    log_dir = root / "logs" / config["stage"] / config["run_name"]
    checkpoint_dir = root / "checkpoints" / config["stage"] / config["run_name"]
    from transformers import AutoTokenizer, set_seed

    set_seed(config["training"]["seed"])
    token, model_spec = os.getenv("HF_TOKEN"), config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.local_tokenizer or config["continuation"]["parent_checkpoint"],
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
    original_tokenizer = tokenizer_identity(tokenizer)
    report["tokenizer"] = original_tokenizer
    training = config["training"]
    report["optimizer_steps_per_epoch"] = math.ceil(
        math.ceil(len(examples) / training["per_device_train_batch_size"])
        / training["gradient_accumulation_steps"])
    report["total_optimizer_steps"] = report["optimizer_steps_per_epoch"] * training["num_train_epochs"]
    # Resume data order/masking must be identical to the parent, not merely the row count.
    validate_prepared_report(report, parent_manifest, parent_report)
    sanity = sanity_text(tokenizer, preview, config["masked_placeholder"])
    if args.prepare_only:
        print(sanity, flush=True)
        print(json.dumps({k: v for k, v in report.items() if k != "kept_ids"}, indent=2), flush=True)
        destination = log_dir / "preparation"
        write_json(destination / "data_report.json", report)
        (destination / "sanity_check.txt").write_text(sanity)
        print("Preparation report:", destination, flush=True)
        return

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from train.qwen_continuation_trainer import QwenContinuationTrainer

    hardware = check_hardware(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    packages = {name: importlib.metadata.version(name) for name in [
        "torch", "transformers", "trl", "accelerate", "datasets", "huggingface-hub",
        "tokenizers", "numpy", "safetensors",
    ]}
    if packages != parent_manifest['packages']:
        raise ValueError('Restore the same locked training package versions as the parent run')
    manifest = {"config": config, "training_code_digest": training_code_digest(),
                "hardware": hardware, "packages": packages, "data_report_digest": digest(report),
                "tokenizer": original_tokenizer, "attention": config["attention"],
                "checkpoint_type": "full_model_with_optimizer_state"}
    identity = digest(manifest)
    resume = select_resume(config, log_dir, checkpoint_dir, identity)
    if resume:
        marker = json.loads((Path(resume) / "stage1_checkpoint.json").read_text())
        if marker["epoch"] >= (args.stop_after_epoch or training["num_train_epochs"]):
            print("Requested epochs are already complete:", resume, flush=True)
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
            resume,
            trust_remote_code=model_spec["trust_remote_code"], token=token,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        model.config.use_cache = False
        # Preserve base model stopping behavior; never change EOS to ChatML im_end.
        for settings in [model.config, model.generation_config]:
            eos = settings.eos_token_id
            if (eos if isinstance(eos, list) else [eos]) != [tokenizer.eos_token_id]:
                raise ValueError("Model and tokenizer native EOS disagree")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if model.config.model_type != "qwen2" or not all(p.requires_grad for p in model.parameters()):
            raise ValueError("Expected full-parameter native Qwen2 training")
        trainer = QwenContinuationTrainer(
            parent_checkpoint=config["continuation"]["parent_checkpoint"],
            additional_steps=report["optimizer_steps_per_epoch"] * 2,
            model=model, args=build_arguments(config, checkpoint_dir, log_dir),
            train_dataset=Dataset.from_list(examples), processing_class=tokenizer,
            data_collator=AssistantCollator(tokenizer.pad_token_id), log_dir=log_dir,
            run_identity=identity, lm_head_chunk_tokens=config["lm_head_chunk_tokens"],
        )
        from train.self_rft_callbacks import install_progress

        install_progress(trainer, args.stop_after_epoch)
        if tokenizer_identity(tokenizer) != original_tokenizer:
            raise ValueError("Trainer changed the configured tokenizer")
        # Prevent a silent quadratic attention fallback on long examples.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result = trainer.train(resume_from_checkpoint=resume)
        completed_epoch = args.stop_after_epoch or int(training["num_train_epochs"])
        if trainer.state.global_step != report["optimizer_steps_per_epoch"] * completed_epoch:
            raise ValueError("Training did not finish the requested complete epochs")
        for epoch in range(6, completed_epoch + 1):
            if not (checkpoint_dir / f"epoch_{epoch}/stage1_checkpoint.json").is_file():
                raise ValueError(f"Missing completed epoch {epoch}")
        write_json(log_dir / f"epoch_{completed_epoch}_train_metrics.json", result.metrics)
        status = "complete" if completed_epoch == training["num_train_epochs"] else "epoch_complete"
        if status == "complete":
            write_json(log_dir / "train_metrics.json", result.metrics)
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
