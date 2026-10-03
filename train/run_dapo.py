"""Single-GPU DAPO from a pinned instruction model or configured SFT checkpoint."""

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from common.io import digest, read_jsonl, write_json, write_jsonl  # noqa: E402
from eval.run_english_eval import git_identity  # noqa: E402
from train.dapo_data import load_config, load_data, prepare_data, select_model  # noqa: E402
from train.dapo_model import configure_tokenizer, stop_token_ids  # noqa: E402
from train.dapo_resume import (  # noqa: E402
    check_checkpoint,
    check_extension_runtime,
    extension_source,
)
from train.stage1_sft import check_hardware, tokenizer_identity  # noqa: E402


def validate_resume(checkpoints, log_dir, resume, identity):
    complete = sorted((p for p in checkpoints.glob("checkpoint-*") if (p / "dapo_checkpoint.json").is_file()),
                      key=lambda p: int(p.name.split("-")[-1]))
    if not resume:
        if (log_dir / "run_manifest.json").exists() or any(checkpoints.iterdir()):
            raise ValueError("Existing DAPO run: resume the latest completed checkpoint or select a new OUTPUT_ROOT")
        return
    path = Path(resume).resolve()
    if not complete or complete[-1].resolve() != path:
        raise ValueError("Resume only the latest completed DAPO checkpoint")
    old = json.loads((log_dir / "run_manifest.json").read_text())
    marker = json.loads((path / "dapo_checkpoint.json").read_text())
    if old["identity"] != identity or marker["run_identity"] != identity:
        raise ValueError("DAPO source, data, settings, code or runtime changed")
    # Preserve partial checkpoint directories rather than treating them as completed.
    for folder in checkpoints.glob("checkpoint-*"):
        if not (folder / "dapo_checkpoint.json").is_file():
            folder.rename(checkpoints / f"incomplete_{folder.name}_{time.time_ns()}")
    records = read_jsonl(log_dir / "steps.jsonl", repair_tail=True)
    abandoned = [r for r in records if r["step"] > marker["global_step"]]
    if abandoned:
        write_jsonl(log_dir / f"abandoned_steps_{time.time_ns()}.jsonl", abandoned)
        write_jsonl(log_dir / "steps.jsonl", [r for r in records if r["step"] <= marker["global_step"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-kind", choices=["base", "sft"], required=True)
    parser.add_argument("--sft-root", default="", help="Required only for the EXAONE SFT source")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-dir", required=True, help="Local SSD directory for rollout weight transfer")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--extend-from-checkpoint", help="Continue a completed run into a separate output root")
    parser.add_argument("--local-data", help="Offline preparation only: verified envelope JSONL")
    parser.add_argument("--local-tokenizer", help="Offline preparation only: pinned tokenizer snapshot")
    args = parser.parse_args()
    if (args.local_data or args.local_tokenizer) and not args.prepare_only:
        parser.error("Local input overrides are only for preparation")
    if args.smoke and (args.prepare_only or args.resume_from_checkpoint or args.extend_from_checkpoint):
        parser.error("Smoke must be a separate fresh training run")
    config = load_config(args.config, smoke=args.smoke)
    config["model_kind"] = args.model_kind
    name = config["run_name_prefix"] + "_" + args.model_kind + ("_smoke" if args.smoke else "")
    root = Path(args.output_root).resolve()
    checkpoints = root / "checkpoints" / name
    log_dir = root / "logs" / name
    parent_manifest = continuation = None
    if args.extend_from_checkpoint:
        parent_manifest, continuation = extension_source(config, args.extend_from_checkpoint, root)
    resume_checkpoint = args.resume_from_checkpoint or args.extend_from_checkpoint
    if resume_checkpoint:
        marker, _ = check_checkpoint(resume_checkpoint)
        if marker["global_step"] % config["algorithm"].get("num_iterations", 1):
            raise ValueError("Resume must start at a complete rollout reuse cycle boundary")
    source, source_identity = select_model(config, args.model_kind, args.sft_root)
    # Keep the original resolved dataset revision on resume, even if HF main moved.
    previous_manifest = log_dir / "run_manifest.json"
    if args.resume_from_checkpoint and previous_manifest.is_file():
        previous = json.loads(previous_manifest.read_text())
        if config["data"]["revision"] == "main":
            config["data"]["revision"] = previous["config"]["data"]["revision"]
    from transformers import AutoTokenizer, set_seed
    token = os.getenv("HF_TOKEN")
    set_seed(config["training"]["seed"])
    revision = config["model"]["revision"] if args.model_kind == "base" else None
    tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer or resume_checkpoint or source,
                                              revision=None if resume_checkpoint else revision,
                                              trust_remote_code=config["model"]["trust_remote_code"], token=token)
    configure_tokenizer(tokenizer, config)
    generation_stops = stop_token_ids(tokenizer, config)
    rows, ids = load_data(config, token=token, local_data=args.local_data)
    prepared, report = prepare_data(rows, ids, tokenizer, config)
    original_tokenizer = tokenizer_identity(tokenizer)
    report["tokenizer"] = original_tokenizer
    if generation_stops is not None:
        report["generation_stop_token_ids"] = generation_stops
    print(json.dumps({"model": source_identity, "data": report, "algorithm": config["algorithm"]}, indent=2), flush=True)
    print("FIRST FORMATTED PROMPT\n" + prepared[0]["prompt"], flush=True)
    print("Checkpoints:", checkpoints, "\nLogs:", log_dir, flush=True)
    if args.prepare_only:
        write_json(log_dir / "preparation/data_report.json", report)
        write_json(log_dir / "preparation/resolved_config.json", config)
        return

    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    from train.dapo_rollout import VLLMRollout
    from train.dapo_trainer import DAPOTrainer, make_arguments

    hardware = check_hardware(config)
    packages = {p: importlib.metadata.version(p) for p in ["torch", "transformers", "trl", "vllm", "accelerate", "datasets", "math-verify"]}
    if packages["trl"] != "0.24.0" or packages["vllm"] != "0.14.1":
        raise ValueError("Use the pinned DAPO environment")
    manifest = {"config": config, "source": source_identity, "data_report_digest": digest(report),
                "git_commit": git_identity(), "packages": packages, "hardware": hardware,
                "parameter_precision": "float32", "compute_precision": "bfloat16"}
    if continuation:
        check_extension_runtime(parent_manifest, manifest)
        manifest["continuation"] = continuation
    identity = digest(manifest)
    checkpoints.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    validate_resume(checkpoints, log_dir, args.resume_from_checkpoint, identity)
    write_json(previous_manifest, {**manifest, "identity": identity})
    write_json(log_dir / "data_report.json", report)
    (log_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if resume_checkpoint:
        source = str(Path(resume_checkpoint).resolve())
    elif args.model_kind == "base":
        source = snapshot_download(config["model"]["repo"], revision=revision, token=token,
                                   allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt"])
    started = time.perf_counter()
    trainer = rollout = None
    status = "failed"
    torch.cuda.reset_peak_memory_stats()
    try:
        model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=config["model"]["trust_remote_code"],
                    torch_dtype=torch.float32, attn_implementation="sdpa", token=token)
        model.config.use_cache = False
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError("DAPO must update the full model")
        if generation_stops is not None:
            # Preserve all native stop IDs in saved checkpoints, including EOT.
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = generation_stops
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        # Initialize rollout before Trainer moves the FP32 training copy to GPU.
        # Sleep then frees the rollout GPU allocation during optimization.
        rollout = VLLMRollout(source, config, args.work_dir, tokenizer=tokenizer)
        trainer = DAPOTrainer(model=model, args=make_arguments(config, checkpoints, log_dir),
            processing_class=tokenizer, train_dataset=Dataset.from_list(prepared),
            dapo_config=config, prepared_rows=prepared, rollout=rollout,
            log_dir=log_dir, run_identity=identity)
        if tokenizer_identity(tokenizer) != original_tokenizer:
            raise ValueError("Trainer changed the shared tokenizer")
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        if trainer.state.global_step != config["training"]["max_steps"]:
            raise ValueError("DAPO did not complete the configured update budget")
        # Save the final update even if max_steps is not a multiple of save_steps.
        final = checkpoints / f"checkpoint-{trainer.state.global_step}"
        if not (final / "dapo_checkpoint.json").is_file():
            trainer._save_checkpoint(trainer.model, trial=None)
        write_json(log_dir / "train_metrics.json", result.metrics)
        status = "complete"
    finally:
        write_json(log_dir / f"attempt_{time.time_ns()}.json", {
            "status": status, "elapsed_seconds": time.perf_counter() - started,
            "global_step": trainer.state.global_step if trainer else None,
            "resume_from_checkpoint": resume_checkpoint,
            "peak_training_process_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_training_process_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "memory_scope": "Training process only; vLLM may run in another process",
        })
        if rollout:
            rollout.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
