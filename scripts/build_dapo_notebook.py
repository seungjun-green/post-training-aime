"""Build the EXAONE DAPO training notebook with plain Python settings."""

from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # EXAONE 2.4B — DAPO on English DAPO-Math-17K
        Set **MODEL_KIND = "base"** for the original instruction model, or **"sft"** for
        the original DeepSeek-column SFT epoch 5. This is the EXAONE experiment, not Llama
        or the Pro-column SFT. Both choices share the same English prompt and DAPO settings.

        This notebook runs a **100-optimizer-update pilot with rollout reuse**. It does not mean 100
        dataset epochs or a complete pass through every question. Default settings:
        G=8, 16 retained mixed-correctness questions (128 responses) per rollout, microbatch 1,
        **two optimizer updates per rollout** (50 fresh rollouts in 100 updates),
        LR 1e-6, 20-step linear warmup then constant LR, T=1/top-p=1, clip 0.20/0.28,
        token-level loss, group-normalized shaped rewards, no KL and no learned reward model.
        Full responses may use 20,480 tokens. A soft penalty starts at 16,384 and reaches -1
        at 20,480. No thinking/answer delimiter is injected during RL generation.

        Data comes from `Seungjun/dp_removed_DAPO-Math-17k-Processed` (`en`), checked against
        the existing decontamination record: **14,068 rows**. Source wrappers are replaced by
        the shared English prompt. Overlong prompts are dropped and reported, never truncated.
        Dynamic sampling retains only groups with both correct and incorrect answers. After
        ten candidate batches without enough mixed groups, it stops with saved diagnostics.

        Use the **RTX PRO 6000 Blackwell 96GB**, enable `HF_TOKEN`, and push the implementation
        to GitHub main before setup. BF16 compute and rollouts use FP32 training/master weights
        and optimizer state so small RL updates are not lost to BF16 rounding. Training and
        rollout share one GPU; vLLM sleeps during optimization and runs up to 16 sequences
        concurrently during generation. Updated weights transfer through a temporary local-SSD
        snapshot before each fresh rollout. GPU compatibility and memory require the smoke test.

        Outputs go to Drive. Five full resumable checkpoints plus a smoke checkpoint can require
        approximately **200 GB**; they include model and optimizer state. This notebook performs
        training and reward checks, not benchmark evaluation. All hyperparameters are in
        `configs/dapo_reuse.yaml`. The original one-pass configs and 100-to-300 continuation
        notebook remain available for existing runs. Settings below are plain Python variables.
        """),
        code("""
        MODEL_KIND = "base"  # "base" or "sft"
        SFT_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-Reuse2"
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-dapo"
        RL_ENV = "/content/lg-dapo-env"
        WORK_DIR = "/content/lg-dapo-work"
        CONFIG = "configs/dapo_reuse.yaml"
        """),
        markdown("""
        ## Setup
        Mount Drive and install the separate pinned training/generation environment.
        Uploading this notebook alone does not update the GitHub code it clones.
        """),
        code("""
        import os, sys, subprocess, json
        from pathlib import Path
        from google.colab import drive, userdata
        drive.mount("/content/drive")
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        if not Path(CODE_ROOT).exists():
            subprocess.check_call(["git", "clone", "--branch", "main", REPO_URL, CODE_ROOT])
        def git(*args):
            return subprocess.check_output(["git", "-C", CODE_ROOT, *args], text=True).strip()
        if git("remote", "get-url", "origin") != REPO_URL or git("status", "--porcelain"):
            raise ValueError("Use a clean checkout of the configured repository")
        if git("branch", "--show-current") != "main":
            raise ValueError("Use a main checkout")
        subprocess.check_call(["git", "-C", CODE_ROOT, "pull", "--ff-only", "origin", "main"])
        print("Code commit:", git("rev-parse", "HEAD"))
        for required in [CONFIG, "train/run_dapo.py", "train/dapo_trainer.py", "train/dapo_data.py",
                         "train/dapo_rollout.py", "train/dapo_sampling.py", "scripts/setup_dapo_runtime.py",
                         "requirements-dapo.lock"]:
            if not (Path(CODE_ROOT) / required).is_file():
                raise FileNotFoundError(f"Push the DAPO implementation to GitHub main first; missing {required}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_dapo_runtime.py", "--venv", RL_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        """),
        markdown("""
        ## Select the starting model
        Rerun this cell after changing MODEL_KIND or the Drive folders. Base needs no SFT files.
        SFT loads `SFT_ROOT/checkpoints/stage1/sft_s1k/epoch_5/` and verifies its original run
        manifest. Each starting model has separate output names, so the two experiments cannot
        overwrite each other. Existing runs require an explicit resume or a different OUTPUT_ROOT.
        """),
        code("""
        from train.dapo_data import load_config, select_model
        cfg = load_config(Path(CODE_ROOT) / CONFIG)
        source, source_identity = select_model(cfg, MODEL_KIND, SFT_ROOT)
        RUN_NAME = cfg["run_name_prefix"] + "_" + MODEL_KIND
        run_logs = Path(OUTPUT_ROOT) / "logs" / RUN_NAME
        checkpoint_root = Path(OUTPUT_ROOT) / "checkpoints" / RUN_NAME
        TRAIN_COMMAND = [str(Path(RL_ENV) / "bin/python"), "train/run_dapo.py",
                         "--config", CONFIG, "--model-kind", MODEL_KIND, "--sft-root", SFT_ROOT,
                         "--output-root", OUTPUT_ROOT, "--work-dir", str(Path(WORK_DIR) / RUN_NAME)]
        print("Starting model:", source)
        print("Checkpoints:", checkpoint_root)
        print("Training logs:", run_logs)
        print("Completed checkpoints:", [str(p.parent) for p in sorted(checkpoint_root.glob("checkpoint-*/dapo_checkpoint.json"))])
        """),
        markdown("""
        ## Prepare data and inspect the prompt
        Verifies the HF export and tokenizes the actual shared prompt. Loads no model weights.
        Saves data provenance, prompt-length exclusions and the resolved configuration on Drive.
        """),
        code("""
        run_logged(TRAIN_COMMAND + ["--prepare-only"], cwd=CODE_ROOT,
                   log_path=run_logs / "prepare_console.log")
        """),
        markdown("""
        ## GPU smoke test — two updates on one rollout
        Enable RUN_SMOKE. Smoke uses G=8 and the full 20,480-token limit, retaining two mixed
        groups for two updates. Warmup is disabled for these steps so the learning rate
        is nonzero. It checks generation, rule scoring, refill, backward/optimizer
        updates, reuse of fixed old-policy probabilities and saving a resumable checkpoint.
        In steps.jsonl, policy_iteration should be 1 then 2; the second step should have
        reused_rollout=true and new_generated_tokens=0. Clipping can still be zero if the
        policy changes too little to cross 0.8/1.28; nonzero clipping is not a smoke requirement. Each attempt has its own timestamped folder.
        Smoke is not a benchmark and its weights are never used to initialize the full run.
        If too few mixed groups are found, inspect its saved rollouts; do not treat the stopped
        smoke as a successful update. The generated-token and acceptance logs explain its cost.
        """),
        code("""
        RUN_SMOKE = False
        if RUN_SMOKE:
            from datetime import datetime, timezone
            smoke_root = Path(OUTPUT_ROOT) / "smoke_attempts" / RUN_NAME / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            command = list(TRAIN_COMMAND)
            command[command.index("--output-root") + 1] = str(smoke_root)
            run_logged(command + ["--smoke"], cwd=CODE_ROOT, log_path=smoke_root / "smoke_console.log",
                       compact_progress=True)
            print("Smoke results:", smoke_root)
        else:
            print("Smoke is off. Set RUN_SMOKE = True to test two DAPO updates on one rollout.")
        """),
        markdown("""
        ## Train the 100-update pilot
        Enable RUN_TRAINING after preparation and smoke. Saves checkpoints at updates
        20, 40, 60, 80 and 100. Each fresh rollout is used for two optimizer updates with
        fixed old-policy log-probabilities and advantages. Then updated weights generate the next
        rollout. Dynamic sampling may generate more than 128 responses to obtain 16 mixed groups.
        max_steps and save_steps must be multiples of num_iterations, so every saved checkpoint
        finishes a complete reuse cycle. This starts a fresh experiment from base/SFT; do not use
        the original one-pass checkpoint with RESUME_CHECKPOINT.

        To resume, set RESUME_CHECKPOINT to the latest completed `checkpoint-<step>` folder.
        Keep code, config, source and runtime unchanged. Work after the last saved checkpoint
        is replayed; earlier raw rollout attempts remain available for audit. The pinned data
        revision is recovered from the run manifest. If no checkpoint completed, start with a
        new OUTPUT_ROOT. No existing run is silently overwritten.
        """),
        code("""
        RUN_TRAINING = False
        RESUME_CHECKPOINT = ""
        if RUN_TRAINING:
            command = TRAIN_COMMAND + (["--resume-from-checkpoint", RESUME_CHECKPOINT] if RESUME_CHECKPOINT else [])
            run_logged(command, cwd=CODE_ROOT, log_path=run_logs / "training_console.log",
                       compact_progress=True)
        else:
            print("Training is off. Set RUN_TRAINING = True to run the 100-update pilot.")
        """),
        markdown("""
        ## Drive outputs
        Under OUTPUT_ROOT, base and SFT use `dapo_exaone_base` and `dapo_exaone_sft` respectively:
        - `checkpoints/<run>/checkpoint-<step>/`: full model, tokenizer, optimizer,
          scheduler, RNG, Trainer state and completion marker.
        - `logs/<run>/steps.jsonl`: policy loss, reward/entropy/clipping metrics,
          accuracy and acceptance counts, truncation counts, generated tokens and update duration.
          Rollout statistics repeat on reused steps; count new_generated_tokens for fresh generation.
          policy_iteration and rollout_first_update identify the pass and source rollout.
        - `logs/<run>/rollouts/attempt_*/update_*/`: generated text, extracted answers,
          correctness, length penalties and retained/rejected group records.
        - `logs/<run>/`: console output, resolved config, data report, run manifest and
          per-attempt timing/memory summaries. GPU memory summaries cover the training process;
          vLLM can own an additional process.
        - `smoke_attempts/`: separate smoke logs and checkpoint.

        Console logs are saved automatically. Save a copy of the notebook in Drive through
        Colab if you also want the `.ipynb` with displayed cell outputs.
        """),
    ]


if __name__ == "__main__":
    write_notebook("train_dapo_exaone.ipynb", cells())
