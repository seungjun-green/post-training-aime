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
        Choose `EVAL_PROFILE` below; use the same option in the SFT evaluation notebook:
        - **greedy** (default): temperature 0, one answer per problem, **630 responses**, pass@1.
        - **sample8**: temperature 1.0/top-p 0.7, eight answers per AIME/AMC problem and
          four per MATH-500 problem, **3,040 responses**. Reports pass@1/4/8 on AIME/AMC
          and pass@1/4 on MATH-500, plus average accuracy.
        Both use continuous batching, maximum 20,480 output tokens, the native chat template,
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
        EVAL_PROFILE = "greedy" # @param ["greedy", "sample8"]
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
        required = ["eval/run_batched_eval.py", "eval/profiles.py", "configs/eval_profiles.yaml"]
        if any(not (Path(CODE_ROOT) / name).is_file() for name in required):
            raise RuntimeError("GitHub is missing the evaluation fixes. Push your local changes, then rerun this cell.")
        print("Evaluation code commit:", git("rev-parse", "HEAD"))
        """),
        code("""
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_eval_runtime.py", "--venv", GPU_ENV], cwd=CODE_ROOT)
        sys.path.insert(0, CODE_ROOT)
        from common.process import run_logged
        from eval.profiles import profile_root
        PROFILE_ROOT = profile_root(OUTPUT_ROOT, EVAL_PROFILE)
        print("Selected profile:", EVAL_PROFILE, "Results:", PROFILE_ROOT)
        import json
        suite = json.loads((Path(CODE_ROOT) / "configs/english_eval_suite.json").read_text())
        FULL_PROGRESS = {name: spec["rows"] for name, spec in suite["datasets"].items()}
        EVAL_COMMAND = [str(Path(GPU_ENV) / "bin/python"), "-m", "eval.run_batched_eval",
                        "--model", MODEL, "--revision", "e949c91dec92095908d34e6b560af77dd0c993f8",
                        "--stage", "stage0", "--run_name", "baseline_english",
                        "--profile", EVAL_PROFILE, "--output_root", OUTPUT_ROOT]
        """),
        markdown("""
        ## Smoke check — five problems, one response each
        Checks actual GPU loading, English generation and scoring. It is not a benchmark result.
        Stored under `OUTPUT_ROOT/profiles/<profile>/smoke/`; full results use the sibling `full/`.
        The smoke cell downloads `smoke_test_result.zip` for inspection before full evaluation. A default Run all performs this smoke check only. Inspect the responses before
        enabling the full run. Subprocess output, including errors, is displayed live and saved
        to `PROFILE_ROOT/smoke_console.log`. The traceback includes the last error lines on failure.
        Each CLI process releases its GPU memory when it exits.
        A tqdm bar shows completed problems, elapsed time and estimated time remaining.
        It advances after all answers for a problem have been generated and scored.
        """),
        code("""
        run_logged(EVAL_COMMAND + ["--smoke"], cwd=CODE_ROOT,
                   log_path=PROFILE_ROOT / "smoke_console.log",
                   progress_totals={name: 1 for name in FULL_PROGRESS})
        from google.colab import files
        files.download(str(PROFILE_ROOT / "smoke/archives/stage0/baseline_english/smoke_test_result.zip"))
        """),
        markdown("""
        ## Full English baseline
        Enable this after smoke completes. Full evaluation can take a long time; results and
        completed problems are saved on Drive to `*_problems.jsonl`. Rerun with the same commit,
        profile and output root to resume. Incomplete problems are regenerated in full; the
        flat `*_generations.jsonl` export is written at completion. Existing 32-sample results
        remain separate. These new profiles start fresh; they do not reuse old responses.
        The base-model revision is pinned in the evaluation code to the loader compatible with
        the locked Transformers runtime; the original weights and tokenizer are unchanged.
        The recorded model revision is reused on resume.
        """),
        code("""
        RUN_FULL_EVAL = False
        if RUN_FULL_EVAL:
            run_logged(EVAL_COMMAND, cwd=CODE_ROOT,
                       log_path=PROFILE_ROOT / "full_console.log", progress_totals=FULL_PROGRESS)
        else:
            print("Full evaluation is off. Enable RUN_FULL_EVAL after the smoke check.")
        """),
        code("""
        import json
        mode = "full" if RUN_FULL_EVAL else "smoke"
        result = PROFILE_ROOT / mode / "results/stage0/baseline_english.json"
        if result.exists():
            print(json.dumps(json.loads(result.read_text())["metrics"], indent=2))
            print("Results and raw generations:", result.parent)
        """),
        markdown("""
        The model remains EXAONE-3.5-2.4B-Instruct. No training or uploads run here.
        Run this full baseline before or after SFT for each profile you want to compare. Choose the same
        profile in both notebooks. Profiles have separate manifests and cannot mix results.
        Switching profiles requires rerunning the command-setup cell before smoke or full evaluation.
        Disconnect/delete the GPU runtime when finished to stop rental charges.
        """),
    ]
