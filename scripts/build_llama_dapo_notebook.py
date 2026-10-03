"""Build the standalone Llama 3.2 3B Instruct DAPO launcher."""

from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # Llama 3.2 3B Instruct — DAPO
        Starts from **meta-llama/Llama-3.2-3B-Instruct**, pinned to a model commit.
        Full-parameter training on one **RTX PRO 6000 Blackwell 96GB**. This notebook
        runs DAPO training only. It does not load EXAONE, an SFT checkpoint or a LoRA adapter.

        Settings in `configs/dapo_llama32_3b.yaml`:
        - G=8, 16 retained mixed-correctness questions (128 responses) per rollout.
        - **Two optimizer updates per rollout**, with fixed old-policy probabilities and advantages.
        - 100 optimizer updates = 50 fresh rollout batches; checkpoints every 20 updates.
        - LR 1e-6, 20-update warmup then constant LR; temperature/top-p 1; clip 0.20/0.28.
        - Rule-based +1/-1 reward, group normalization, token-level DAPO loss, no KL.
        - Maximum response 20,480 tokens; soft length penalty begins at 16,384. No budget forcing.

        Uses the same verified 14,068-row decontaminated English DAPO dataset and English
        instruction. Llama's native chat template uses a fixed date for reproducibility.
        Native EOT, end-of-text and end-of-message tokens stop generation; padding is separate.
        FP32 parameters/Adam state, BF16 compute/rollouts, gradient checkpointing and microbatch 1.
        vLLM generates up to 16 concurrent responses and sleeps during optimization.

        Preparation, GPU smoke and full training are separate cells. Local CPU tests do not
        establish GPU memory fit: run smoke on the target GPU first. Five full resumable
        checkpoints plus smoke need roughly **250 GB** of Drive space. All settings below
        are plain Python variables. This is a new experiment; EXAONE continuation is separate.
        """),
        code("""
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-DAPO-Llama32-3B"
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-dapo-llama32"
        RL_ENV = "/content/lg-dapo-env"
        WORK_DIR = "/content/llama32-dapo-work"
        CONFIG = "configs/dapo_llama32_3b.yaml"
        """),
        markdown("""
        ## Setup
        Enable `HF_TOKEN` in Colab secrets. Its Hugging Face account needs access to
        [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
        and the dataset. Push the updated repository code to GitHub main before running:
        uploading this notebook alone does not update the code it clones.
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
        for required in [CONFIG, "train/run_dapo.py", "train/dapo_model.py", "train/dapo_trainer.py",
                         "train/dapo_data.py", "train/dapo_rollout.py", "train/dapo_sampling.py",
                         "scripts/setup_dapo_runtime.py", "requirements-dapo.lock"]:
            if not (Path(CODE_ROOT) / required).is_file():
                raise FileNotFoundError(f"Push the updated repository first; missing {required}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_dapo_runtime.py", "--venv", RL_ENV,
                               "--architecture", "LlamaForCausalLM"], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        """),
        markdown("""
        ## Paths and training command
        Full checkpoints and logs are isolated from all previous EXAONE and Llama SFT runs.
        """),
        code("""
        from train.dapo_data import load_config
        cfg = load_config(Path(CODE_ROOT) / CONFIG)
        if cfg["model"]["repo"] != "meta-llama/Llama-3.2-3B-Instruct":
            raise ValueError("Use the Llama 3.2 3B Instruct config")
        RUN_NAME = cfg["run_name_prefix"] + "_base"
        run_logs = Path(OUTPUT_ROOT) / "logs" / RUN_NAME
        checkpoint_root = Path(OUTPUT_ROOT) / "checkpoints" / RUN_NAME
        TRAIN_COMMAND = [str(Path(RL_ENV) / "bin/python"), "train/run_dapo.py", "--config", CONFIG,
                         "--model-kind", "base", "--output-root", OUTPUT_ROOT, "--work-dir", WORK_DIR]
        print("Starting model:", cfg["model"]["repo"])
        print("Updates per rollout:", cfg["algorithm"]["num_iterations"])
        print("Checkpoints:", checkpoint_root)
        print("Training logs:", run_logs)
        """),
        markdown("""
        ## Prepare and inspect the prompt
        Checks dataset provenance, tokenizer access, native stop/padding tokens and prompt lengths.
        Does not load model weights. Saves resolved configuration and the data report to Drive.
        """),
        code("""
        run_logged(TRAIN_COMMAND + ["--prepare-only"], cwd=CODE_ROOT,
                   log_path=run_logs / "prepare_console.log")
        """),
        markdown("""
        ## GPU smoke — two updates on one rollout
        Uses G=8, two retained mixed groups, the full token cap, and no warmup.
        Verifies generation, rewards, reuse, optimizer updates and checkpoint saving on your GPU.
        Smoke has its own timestamped directory; its weights are never used for the full run.
        The second log row must have policy_iteration=2, reused_rollout=true and new_generated_tokens=0.
        Clipping can still be zero if the update is small. A nonzero clip fraction is not required.
        """),
        code("""
        RUN_SMOKE = False
        if RUN_SMOKE:
            from datetime import datetime, timezone
            smoke_root = Path(OUTPUT_ROOT) / "smoke_attempts" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            command = list(TRAIN_COMMAND)
            command[command.index("--output-root") + 1] = str(smoke_root)
            run_logged(command + ["--smoke"], cwd=CODE_ROOT,
                       log_path=smoke_root / "smoke_console.log", compact_progress=True)
            smoke_log = smoke_root / "logs" / (RUN_NAME + "_smoke") / "steps.jsonl"
            records = [json.loads(line) for line in smoke_log.read_text().splitlines()]
            assert [r["policy_iteration"] for r in records] == [1, 2]
            assert records[-1]["reused_rollout"] and records[-1]["new_generated_tokens"] == 0
            print("Two-update smoke completed. Logs:", smoke_log)
        else:
            print("Set RUN_SMOKE = True to run the separate GPU smoke.")
        """),
        markdown("""
        ## Full training — 100 optimizer updates
        Run after preparation and smoke. Saves checkpoints 20/40/60/80/100, with full model,
        tokenizer, optimizer, scheduler and RNG state. Dynamic sampling may generate extra
        candidate groups to find 16 mixed groups. Each retained batch is used for two updates.

        After interruption, set RESUME_CHECKPOINT to this run's latest complete checkpoint.
        Keep code/config/runtime unchanged. Saving only at complete reuse cycles makes resume
        start with a fresh rollout. max_steps and save_steps must be multiples of num_iterations.
        An existing run is never silently overwritten; choose a new OUTPUT_ROOT for a new run.
        """),
        code("""
        RUN_TRAINING = False
        RESUME_CHECKPOINT = ""
        if RUN_TRAINING:
            command = TRAIN_COMMAND + (["--resume-from-checkpoint", RESUME_CHECKPOINT] if RESUME_CHECKPOINT else [])
            run_logged(command, cwd=CODE_ROOT, log_path=run_logs / "training_console.log",
                       compact_progress=True)
        else:
            print("Set RUN_TRAINING = True to start the full run.")
        """),
        markdown("""
        ## Drive outputs
        Under OUTPUT_ROOT:
        - `checkpoints/dapo_llama32_3b_base/checkpoint-<step>/`: full resumable checkpoints.
        - `logs/dapo_llama32_3b_base/steps.jsonl`: loss, gradient norm, learning rate, reward,
          entropy/clipping, candidate accuracy, acceptance counts, length and timing metrics.
        - `logs/dapo_llama32_3b_base/rollouts/attempt_*/update_*/`: raw answers and scoring records.
        - `logs/dapo_llama32_3b_base/`: console logs, resolved config, data report and run manifest.
        - `smoke_attempts/<timestamp>/`: separate smoke logs and checkpoint.

        `rollout_first_update` and `policy_iteration` identify reused batches. Rollout statistics
        repeat on the second pass; sum `new_generated_tokens` for actual new generation.
        Reused-batch accuracy is not a fresh measurement of the updated model.
        Console logs are automatic. Save the notebook itself in Drive through Colab if you
        also want its displayed cell outputs. No checkpoint weights are packaged for sharing.
        """),
    ]


if __name__ == "__main__":
    write_notebook("train_dapo_llama32_3b.ipynb", cells())
