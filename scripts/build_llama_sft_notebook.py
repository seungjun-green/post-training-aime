"""Build the separate Llama 3.1 8B LoRA training launcher."""

from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # Llama 3.1 8B Instruct — LoRA SFT on original DeepSeek traces
        Starts from **meta-llama/Llama-3.1-8B-Instruct**, using the original
        `deepseek_thinking_trajectory` plus `deepseek_attempt` columns from
        `Seungjun/dp_removed_s1K-1.1` at the pinned original revision.
        Train **5 epochs** with rank **32**, alpha **64**, dropout **0.05**, all seven
        attention/MLP projections, learning rate **5e-5**, batch **1**, accumulation **16**,
        cosine schedule, warmup **5%**, and seed **42**. Base weights are frozen BF16;
        adapters are trainable. No quantization. All settings live in the committed YAML.

        Use an **RTX PRO 6000 Blackwell 96GB** runtime. Enable the `HF_TOKEN` Colab secret
        for an account with access to the gated Llama model. Push this implementation to
        GitHub main before setup; uploading the notebook alone does not upload its modules.
        A separate Python 3.12 environment installs the pinned training stack and PEFT.

        Llama's native template and tokenizer count the **whole formatted sequence**.
        Rows over **20,480 tokens are dropped, never truncated**. Both reasoning and answer,
        plus the end-of-turn token, receive loss; prompt and padding do not. The retained
        count is recomputed and may differ from EXAONE. Existing correctness grades do not filter rows.

        Preparation, GPU smoke training and full training are separate cells. Run preparation,
        then the smoke test, then enable full training. Smoke uses the longest retained example
        and its own output folder; full training starts from fresh base weights and adapters.
        Native PyTorch SDPA is restricted to its **FlashAttention** backend, with nonreentrant
        gradient checkpointing. The longest-example smoke checks GPU compatibility and memory.
        This notebook performs training only, with tqdm progress and loss logs on Drive.
        """),
        code("""
        # @title Settings
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-llama-lora"
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Llama31-LoRA" # @param {type:"string"}
        TRAIN_ENV = "/content/llama-lora-env"
        CONFIG = "configs/stage1_sft_llama31_lora.yaml"
        """),
        code("""
        # @title Mount Drive and install the isolated training environment
        import os, subprocess, sys, json
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
        for required in [CONFIG, "train/stage1_llama_lora.py", "train/llama_sft_data.py",
                         "scripts/setup_llama_sft_runtime.py", "requirements-stage1-lora.lock"]:
            if not (Path(CODE_ROOT) / required).is_file():
                raise FileNotFoundError(f"Push the Llama LoRA implementation to GitHub main first; missing {required}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_llama_sft_runtime.py", "--venv", TRAIN_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        TRAIN_COMMAND = [str(Path(TRAIN_ENV) / "bin/python"), "train/stage1_llama_lora.py",
                         "--config", CONFIG, "--output_root", TRAIN_ROOT]
        """),
        markdown("""
        ## Prepare data and inspect the supervised tokens
        This loads the tokenizer and dataset, without loading the 8B model weights.
        Check the displayed columns, kept/dropped counts and loss mask. `[MASKED]` represents
        a token excluded from loss. The answer's trailing whitespace follows Llama's native
        template trimming. Reports and the complete preview are saved to Drive.
        If Hugging Face reports gated access, enable Llama access for the account owning `HF_TOKEN`.
        """),
        code("""
        run_logged(TRAIN_COMMAND + ["--prepare-only"], cwd=CODE_ROOT,
                   log_path=Path(TRAIN_ROOT) / "prepare_console.log")
        """),
        markdown("""
        ## GPU smoke training — longest retained example
        Set `RUN_SMOKE = True`. This runs one optimizer step, saves an adapter checkpoint
        with optimizer/RNG state and records finite loss, gradient norm and peak GPU memory.
        It uses the same rank, precision, attention kernel and sequence filtering as the full
        run; the committed smoke settings use one example, one epoch and accumulation 1.
        Each smoke attempt has a separate timestamped output folder. It does not modify the
        full training run. Smoke updates are discarded when starting full training.
        """),
        code("""
        RUN_SMOKE = False # @param {type:"boolean"}
        if RUN_SMOKE:
            from datetime import datetime, timezone
            smoke_root = Path(TRAIN_ROOT) / "smoke_attempts" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            command = list(TRAIN_COMMAND)
            command[command.index("--output_root") + 1] = str(smoke_root)
            run_logged(command + ["--smoke"], cwd=CODE_ROOT,
                       log_path=smoke_root / "smoke_console.log")
            print("Smoke output:", smoke_root)
        else:
            print("Smoke is off. Enable RUN_SMOKE to check the longest retained example on your GPU.")
        """),
        markdown("""
        ## Full LoRA training — five epochs
        Set `RUN_TRAINING = True` after preparation and smoke pass. Save all five epoch
        adapters with tokenizer, optimizer, scheduler and RNG state. These are **LoRA adapter
        checkpoints**, not standalone merged 8B models; inference must load the pinned base
        plus the adapter, or merge them first. No merge or evaluation is performed here.

        For interruption recovery, set `RESUME_CHECKPOINT` to the latest completed epoch
        directory. Keep code, config and runtime unchanged. An incomplete epoch is replayed
        from the latest completed one. To start over, choose a new `TRAIN_ROOT` in Settings
        and rerun setup; existing runs are not overwritten.
        """),
        code("""
        RUN_TRAINING = False # @param {type:"boolean"}
        RESUME_CHECKPOINT = "" # @param {type:"string"}
        if RUN_TRAINING:
            command = TRAIN_COMMAND + (["--resume_from_checkpoint", RESUME_CHECKPOINT] if RESUME_CHECKPOINT else [])
            run_logged(command, cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / "training_console.log")
        else:
            print("Training is off. Enable RUN_TRAINING for five epochs from the original Llama model.")
        """),
        markdown("""
        ## Saved outputs
        All paths are relative to `TRAIN_ROOT`:
        - Adapters: `checkpoints/stage1/sft_llama31_8b_lora_s1k/epoch_1/` through `epoch_5/`.
        - Per-step loss, learning rate, gradient norm and speed: `logs/stage1/sft_llama31_8b_lora_s1k/steps.jsonl`.
        - Data report, pinned config, manifest, adapter parameter report, timing and peak memory:
          `logs/stage1/sft_llama31_8b_lora_s1k/`.
        - Console output: `prepare_console.log` and `training_console.log`.
        - Smoke runs: `smoke_attempts/<timestamp>/`.

        The preparation report prints the actual optimizer-step count after Llama tokenization.
        """),
    ]


if __name__ == "__main__":
    write_notebook("train_stage1_llama31_lora.ipynb", cells())
