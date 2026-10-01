"""Build separate Stage 1 training and evaluation launchers."""

from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # Stage 1 — English s1-style SFT on EXAONE
        Use the **RTX PRO 6000 Blackwell 96GB** runtime and enable the `HF_TOKEN` secret.
        Push the Stage 1 implementation to your GitHub repository before running setup.
        This notebook uses a separate training environment and preserves the frozen evaluation
        environment. All training settings are in the committed `configs/stage1_sft.yaml`.
        This notebook performs SFT and saves all five epoch checkpoints. It never runs evaluation.
        A default Run all prepares and inspects data only; enable `RUN_TRAINING` to train.
        Keep enough Drive space for five full model/optimizer checkpoints (allow at least 100GB).
        After training, open `evaluate_stage1_sft.ipynb` to evaluate every saved epoch.
        """),
        code("""
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-stage1"
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"
        TRAIN_ENV = "/content/lg-sft-env"
        CONFIG = "configs/stage1_sft.yaml"
        """),
        code("""
        import os, subprocess, sys
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
    ]


def eval_cells():
    return [
        markdown("""
        # Stage 1 — Evaluate epoch 5 first
        This notebook evaluates existing checkpoints; it never trains a model.
        Epoch 5 is evaluated on **AIME 2024, AIME 2025, AIME 2026, AMC 2023,
        and MATH-500** using continuous batching. Choose the same `EVAL_PROFILE` as your baseline:
        - **greedy** (default): temperature 0, one answer per problem, **630 responses**, pass@1.
        - **sample8**: temperature 1.0/top-p 0.7, eight answers per AIME/AMC problem and
          four per MATH-500 problem, **3,040 responses**. Reports pass@1/4/8 on AIME/AMC
          and pass@1/4 on MATH-500.
        The default checkpoint selection is **epoch 5 only**. Optionally enable
        `INCLUDE_EARLIER_EPOCHS` to run epochs 1–4 after epoch 5.
        Run this after the SFT notebook has saved the final checkpoint to Drive.
        Use the **RTX PRO 6000 Blackwell 96GB** runtime and enable the `HF_TOKEN` secret.
        Push the notebook changes to your GitHub repository before running setup.
        Evaluation runs sequentially in the original locked evaluation environment, releasing
        GPU memory after each checkpoint. No training environment is installed here.
        """),
        code("""
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-stage1-eval"
        TRAIN_ROOT = "/content/drive/MyDrive/LG-AIME-Stage1"
        BASELINE_ROOT = "/content/drive/MyDrive/LG-AIME-English-Eval-compatible"
        EVAL_ENV = "/content/lg-eval-env"
        CONFIG = "configs/stage1_sft.yaml"
        EVAL_PROFILE = "greedy" # @param ["greedy", "sample8"]
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
        print("Evaluation code commit:", git("rev-parse", "HEAD"))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_eval_runtime.py", "--venv", EVAL_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        """),
        markdown("""
        ## Check saved checkpoints
        Keep `TRAIN_ROOT` the same as in the training notebook. SFT evaluation can run before
        the baseline. The first run creates the selected profile's protocol/runtime manifests
        under `BASELINE_ROOT/profiles/<profile>/full/results/`. Later runs validate the same
        sampling, prompts, datasets, scoring and runtime for comparisons within that profile.
        To switch from an older notebook, stop evaluation, push the updated code, and run setup.
        New profiles start fresh and leave previous 32-sample results intact; old responses are
        not imported. Once a run starts, keep its code commit and settings unchanged for resume.
        """),
        code("""
        import yaml
        from eval.profiles import profile_root
        INCLUDE_EARLIER_EPOCHS = False
        cfg = yaml.safe_load((Path(CODE_ROOT) / CONFIG).read_text())
        epochs = int(cfg["training"]["num_train_epochs"])
        selected_epochs = [epochs]
        if INCLUDE_EARLIER_EPOCHS:
            selected_epochs.extend(range(1, epochs))
        checkpoint_root = Path(TRAIN_ROOT) / "checkpoints" / cfg["stage"] / cfg["run_name"]
        result_root = profile_root(BASELINE_ROOT, EVAL_PROFILE) / "full/results"
        run_identities = set()
        eval_runs = []
        for epoch in selected_epochs:
            checkpoint = checkpoint_root / f"epoch_{epoch}"
            marker = json.loads((checkpoint / "stage1_checkpoint.json").read_text())
            if marker["epoch"] != epoch:
                raise ValueError(f"Checkpoint epoch mismatch: {checkpoint}")
            run_identities.add(marker["run_identity"])
            # Profiles use separate directories for baseline and SFT comparisons.
            run_name = cfg["run_name"] if epoch == epochs else f"{cfg['run_name']}_epoch{epoch}"
            eval_runs.append((epoch, checkpoint, run_name))
        if len(run_identities) != 1:
            raise ValueError("Selected epoch checkpoints must belong to the same training run")
        suite = json.loads((Path(CODE_ROOT) / "configs/english_eval_suite.json").read_text())
        progress_totals = {name: entry["rows"] for name, entry in suite["datasets"].items()}
        print("Benchmarks:", ", ".join(progress_totals))
        for epoch, checkpoint, run_name in eval_runs:
            print(f"Epoch {epoch}: {checkpoint} -> {result_root / cfg['stage'] / (run_name + '.json')}")
        """),
        markdown("""
        ## Full evaluation — epoch 5, then optional earlier checkpoints
        Enable `RUN_STAGE1_EVAL` to evaluate epoch 5. Leave `INCLUDE_EARLIER_EPOCHS = False`
        above for the main result only. To include the earlier checkpoints, set it to `True`
        and rerun the check cell; the evaluation order will be **5, 1, 2, 3, 4**.
        A default Run all performs setup and checks only. Every selected epoch uses all five benchmarks, with the same
        prompts, sampling, seeds and scoring as the baseline. Epoch 5 remains the Stage 1 result.
        Reports, raw generations and manifests are saved under `BASELINE_ROOT/profiles/<profile>/full/results/stage1/`:
        `sft_s1k_epoch1.json` through `sft_s1k_epoch4.json`, and `sft_s1k.json` for epoch 5. Completed problems are durably saved during the run
        in `*_problems.jsonl`; the standard `*_generations.jsonl` export is written at completion.
        Console logs are saved under `TRAIN_ROOT`. Repeating the same commands resumes completed
        problems. `configs/eval_profiles.yaml` defines both options and queues up to 64 greedy
        or 16 sampled problems to feed the existing 32 GPU response slots. Exact sampled text can
        change with batching even though seeds and sampling settings are unchanged.
        Progress counts completed problems, which can finish out of order within each benchmark.
        No AMC-only runs are launched by this notebook.
        """),
        code("""
        RUN_STAGE1_EVAL = False
        if RUN_STAGE1_EVAL:
            eval_python = str(Path(EVAL_ENV) / "bin/python")
            for epoch, checkpoint, run_name in eval_runs:
                command = [eval_python, "-m", "eval.run_batched_eval", "--model", str(checkpoint),
                           "--stage", cfg["stage"], "--run_name", run_name, "--output_root", BASELINE_ROOT,
                           "--profile", EVAL_PROFILE,
                           "--execution_config", "configs/eval_execution.yaml"]
                run_logged(command,
                           cwd=CODE_ROOT, log_path=Path(TRAIN_ROOT) / f"full_eval_epoch{epoch}_{EVAL_PROFILE}_console.log",
                           progress_totals=progress_totals)
        else:
            print("Evaluation is off. Enable RUN_STAGE1_EVAL to run the selected checkpoints, starting with epoch 5.")
        """),
    ]


if __name__ == "__main__":
    write_notebook("train_stage1_sft.ipynb", cells())
    write_notebook("evaluate_stage1_sft.ipynb", eval_cells())
