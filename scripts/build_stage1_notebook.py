"""Build the thin Stage 1 launcher without rebuilding any Spec 1 notebook."""

from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # Stage 1 — English s1-style SFT on EXAONE
        Use the **RTX PRO 6000 Blackwell 96GB** runtime and enable the `HF_TOKEN` secret.
        Push the Stage 1 implementation to your GitHub repository before running setup.
        This notebook uses a separate training environment and preserves the frozen evaluation
        environment. All training settings are in the committed `configs/stage1_sft.yaml`.
        A default Run all prepares and inspects data only. Training and evaluation have separate
        switches below. Keep enough Drive space for five full model/optimizer checkpoints
        (allow at least 100GB), plus the baseline and evaluation outputs.
        """),
        code("""
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-stage1"
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"
        BASELINE_ROOT = "/content/drive/MyDrive/LG-AIME-English-Eval-compatible"
        TRAIN_ENV = "/content/lg-sft-env"
        EVAL_ENV = "/content/lg-eval-env"
        CONFIG = "configs/stage1_sft.yaml"
        """),
        code("""
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_stage1_runtime.py", "--venv", TRAIN_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        TRAIN_COMMAND = [str(Path(TRAIN_ENV) / "bin/python"), "train/stage1_sft.py",
                         "--config", CONFIG, "--output_root", TRAIN_ROOT]
        """),
        markdown("""
        ## Data and loss-mask inspection
        This tokenizes all 996 original published rows without loading model weights. Review the
        printed full text and masked text. The report records overlength exclusions and the count
        of incorrect grades; grades never filter training. Inputs longer than 20,480 tokens are
        dropped, never truncated. Training repeats these checks before the first optimizer step.
        """),
        code("""
        run_logged(TRAIN_COMMAND + ["--prepare-only"], cwd=CODE_ROOT,
                   log_path=Path(TRAIN_ROOT) / "prepare_console.log")
        """),
        markdown("""
        ## Five-epoch full fine-tuning
        Each complete epoch is saved with its tokenizer, model code, optimizer, scheduler and RNG.
        Set `RESUME_CHECKPOINT` to the latest complete epoch directory to resume an interrupted run.
        Keep the same code commit and config. An incomplete epoch is replayed from its start.
        An existing run will not be silently overwritten. Do not rerun setup during training.
        """),
        code("""
        RUN_TRAINING = False
        RESUME_CHECKPOINT = None
        if RUN_TRAINING:
            command = TRAIN_COMMAND + (["--resume_from_checkpoint", RESUME_CHECKPOINT] if RESUME_CHECKPOINT else [])
            run_logged(command, cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / "training_console.log")
        else:
            print("Training is off. Review preparation, then enable RUN_TRAINING.")
        """),
        markdown("""
        ## Evaluate epoch 5 and the five AMC checkpoints
        These run sequentially in the original locked evaluation environment after training exits.
        `BASELINE_ROOT/full/results` must contain the original `eval_protocol.json` and
        `eval_runtime.json`, alongside your stage-0 results. Use the existing Drive baseline root.
        The full evaluation generates 6,160 responses; each AMC evaluation generates 1,280.
        The English full evaluator is unchanged. AMC uses a separate entry point with the same
        sampling, prompts, seeds and scoring. All reports remain under the baseline's
        `full/results/stage1/`. Repeating the same command resumes saved evaluation responses.
        """),
        code("""
        RUN_STAGE1_EVAL = False
        if RUN_STAGE1_EVAL:
            subprocess.check_call([sys.executable, "scripts/setup_eval_runtime.py", "--venv", EVAL_ENV], cwd=CODE_ROOT)
            import yaml
            cfg = yaml.safe_load((Path(CODE_ROOT) / CONFIG).read_text())
            checkpoint_root = Path(TRAIN_ROOT) / "checkpoints" / cfg["stage"] / cfg["run_name"]
            eval_python = str(Path(EVAL_ENV) / "bin/python")
            run_logged([eval_python, "-m", "eval.run_english_eval", "--model", str(checkpoint_root / "epoch_5"),
                        "--stage", cfg["stage"], "--run_name", cfg["run_name"], "--output_root", BASELINE_ROOT],
                       cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / "full_eval_console.log",
                       progress_totals={"aime_2024": 30, "aime_2025": 30, "aime_2026": 30, "amc23": 40, "math_500": 500})
            for epoch in range(1, int(cfg["training"]["num_train_epochs"]) + 1):
                run_logged([eval_python, "-m", "eval.run_stage1_amc", "--model", str(checkpoint_root / f"epoch_{epoch}"),
                            "--run_name", f"{cfg['run_name']}_epoch{epoch}_amc", "--output_root", BASELINE_ROOT],
                           cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / f"amc_epoch{epoch}_console.log",
                           progress_totals={"amc23": 40})
        else:
            print("Stage 1 evaluation is off. Enable after all five checkpoints are complete.")
        """),
    ]


if __name__ == "__main__":
    write_notebook("train_stage1_sft.ipynb", cells())
