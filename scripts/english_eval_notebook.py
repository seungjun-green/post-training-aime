"""Thin Colab launcher: clone the project, install, invoke evaluation CLI."""


def english_eval_cells(markdown, code):
    return [
        markdown("""
        # EXAONE 3.5 2.4B Instruct — English baseline
        Connect to the **RTX PRO 6000 Blackwell 96GB** GPU runtime.
        Add `HF_TOKEN` to Colab Secrets for the private uploaded datasets. No translation API is used.
        All evaluation logic, English prompts, dataset revisions and dependencies live in the Git repo.
        **Before starting:** push your local evaluation fixes to GitHub yourself. This notebook
        clones the repository; it cannot use changes that exist only on your computer.
        Then run these cells in order in a fresh runtime. No manual commit field is needed.
        The notebook clones `main` from your repository and records the actual commit automatically.
        Setup clones `main` on a fresh runtime or updates an existing clean `main` checkout.
        Run setup before evaluation, not while an evaluation process is running.
        For a private Git repo, configure Git authentication in the runtime before cloning;
        do not put tokens in the URL or notebook.

        The pinned English suite has AIME 2024/2025/2026 (30 each), AMC23 (40), MATH-500 (500).
        Full evaluation uses **6,160 responses**: 32 per AIME/AMC problem, 4 per MATH problem.
        Temperature 1.0, top-p 0.7, maximum 20,480 output tokens, native chat template,
        an English step-by-step/boxed-answer instruction, and no added system message.
        Scores include avg@n, pass@k and response lengths. The Korean-letter ratio remains a
        diagnostic only; it does not affect correctness.
        """),
        code("""
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/lg-aime-eval"
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-English-Eval-compatible"
        GPU_ENV = "/content/lg-eval-env"
        MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
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
            raise ValueError("Checkout differs or has local changes; choose a fresh CODE_ROOT")
        if git("branch", "--show-current") != "main":
            raise ValueError("Expected the main branch; choose a fresh CODE_ROOT")
        subprocess.check_call(["git", "-C", CODE_ROOT, "pull", "--ff-only", "origin", "main"])
        required = ["common/process.py", "eval/smoke_report.py"]
        if any(not (Path(CODE_ROOT) / name).is_file() for name in required):
            raise RuntimeError("GitHub is missing the evaluation fixes. Push your local changes, then rerun this cell.")
        print("Evaluation code commit:", git("rev-parse", "HEAD"))
        """),
        code("""
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_eval_runtime.py", "--venv", GPU_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        EVAL_COMMAND = [str(Path(GPU_ENV) / "bin/python"), "-m", "eval.run_english_eval",
                        "--model", MODEL, "--revision", "e949c91dec92095908d34e6b560af77dd0c993f8",
                        "--output_root", OUTPUT_ROOT]
        """),
        markdown("""
        ## Smoke check — five problems, one response each
        Checks actual GPU loading, English generation and scoring. It is not a benchmark result.
        Stored under `OUTPUT_ROOT/smoke/`; full baseline results use `OUTPUT_ROOT/full/`.
        The smoke cell downloads `smoke_test_result.zip`; send it for review before enabling
        full evaluation. A default Run all performs this smoke check only. Inspect the responses before
        enabling the full run. Subprocess output, including errors, is displayed live and saved
        to `OUTPUT_ROOT/smoke_console.log`. The traceback includes the last error lines on failure.
        Each CLI process releases its GPU memory when it exits.
        """),
        code("""
        run_logged(EVAL_COMMAND + ["--smoke"], cwd=CODE_ROOT,
                   log_path=Path(OUTPUT_ROOT) / "smoke_console.log")
        from google.colab import files
        files.download(str(Path(OUTPUT_ROOT) / "smoke/archives/stage0/baseline_english/smoke_test_result.zip"))
        """),
        markdown("""
        ## Full English baseline
        Enable this after smoke completes. Full evaluation can take a long time; results and
        generations are saved on Drive after every response. Rerun with the same commit,
        configuration and output root to resume completed problems. Partially saved problems
        are regenerated with their original seed and must match saved responses exactly.
        The base-model revision is pinned in the evaluation code to the loader compatible with
        the locked Transformers runtime; the original weights and tokenizer are unchanged.
        The recorded model revision is reused on resume.
        """),
        code("""
        RUN_FULL_EVAL = False
        if RUN_FULL_EVAL:
            run_logged(EVAL_COMMAND, cwd=CODE_ROOT,
                       log_path=Path(OUTPUT_ROOT) / "full_console.log")
        else:
            print("Full evaluation is off. Enable RUN_FULL_EVAL after the smoke check.")
        """),
        code("""
        import json
        mode = "full" if RUN_FULL_EVAL else "smoke"
        result = Path(OUTPUT_ROOT) / mode / "results/stage0/baseline_english.json"
        if result.exists():
            print(json.dumps(json.loads(result.read_text())["metrics"], indent=2))
            print("Results and raw generations:", result.parent)
        """),
        markdown("""
        The model remains EXAONE-3.5-2.4B-Instruct. No training or uploads run here.
        Keep the English evaluation config, prompt, suite and dependency lock fixed for later
        checkpoint comparisons. Older Korean evaluation results use a different protocol.
        Disconnect/delete the GPU runtime when finished to stop rental charges.
        """),
    ]
