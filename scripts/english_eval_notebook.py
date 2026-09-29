"""Thin Colab launcher: checkout project commit, install, invoke evaluation CLI."""


def english_eval_cells(markdown, code):
    return [
        markdown("""
        # EXAONE 3.5 2.4B Instruct — English baseline
        Connect to the **RTX PRO 6000 Blackwell 96GB** GPU runtime.
        Add `HF_TOKEN` to Colab Secrets for the private uploaded datasets. No translation API is used.
        All evaluation logic, English prompts, dataset revisions and dependencies live in the Git repo.
        The repository URL and a **full 40-character evaluation commit** are preconfigured below.
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
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git" # @param {type:"string"}
        GIT_COMMIT = "ff0ea8dd1b4471a970082797cc4ff47bfd99fde4" # @param {type:"string"}
        CODE_ROOT = "/content/lg-aime-eval"
        OUTPUT_ROOT = "/content/drive/MyDrive/LG-AIME-English-Eval"
        GPU_ENV = "/content/lg-eval-env"
        MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
        """),
        code("""
        import os, re, subprocess, sys
        from pathlib import Path
        from google.colab import drive, userdata
        drive.mount("/content/drive")
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        if not REPO_URL or not re.fullmatch(r"[0-9a-fA-F]{40}", GIT_COMMIT):
            raise ValueError("Set REPO_URL and the full commit SHA pushed to that repository")
        if not Path(CODE_ROOT).exists():
            subprocess.check_call(["git", "clone", REPO_URL, CODE_ROOT])
        def git(*args):
            return subprocess.check_output(["git", "-C", CODE_ROOT, *args], text=True).strip()
        if git("remote", "get-url", "origin") != REPO_URL or git("status", "--porcelain"):
            raise ValueError("Checkout differs or has local changes; choose a fresh CODE_ROOT")
        git("fetch", "origin", GIT_COMMIT)
        git("checkout", "--detach", GIT_COMMIT)
        assert git("rev-parse", "HEAD").lower() == GIT_COMMIT.lower()
        print("Evaluation code commit:", git("rev-parse", "HEAD"))
        """),
        code("""
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv==0.11.22"])
        subprocess.check_call([sys.executable, "scripts/setup_eval_runtime.py", "--venv", GPU_ENV], cwd=CODE_ROOT)
        EVAL_COMMAND = [str(Path(GPU_ENV) / "bin/python"), "-m", "eval.run_english_eval",
                        "--model", MODEL, "--output_root", OUTPUT_ROOT]
        """),
        markdown("""
        ## Smoke check — five problems, one response each
        Checks actual GPU loading, English generation and scoring. It is not a benchmark result.
        Stored under `OUTPUT_ROOT/smoke/`; full baseline results use `OUTPUT_ROOT/full/`.
        The smoke cell downloads `smoke_test_result.zip`; send it for review before enabling
        full evaluation. A default Run all performs this smoke check only. Inspect the responses before
        enabling the full run. Each CLI process releases its GPU memory when it exits.
        """),
        code("""
        subprocess.check_call(EVAL_COMMAND + ["--smoke"], cwd=CODE_ROOT)
        from google.colab import files
        files.download(str(Path(OUTPUT_ROOT) / "smoke/archives/stage0/baseline_english/smoke_test_result.zip"))
        """),
        markdown("""
        ## Full English baseline
        Enable this after smoke completes. Full evaluation can take a long time; results and
        generations are saved on Drive after every response. Rerun with the same commit,
        configuration and output root to resume completed problems. Partially saved problems
        are regenerated with their original seed and must match saved responses exactly.
        The model revision is pinned on first run and reused on resume.
        """),
        code("""
        RUN_FULL_EVAL = False # @param {type:"boolean"}
        if RUN_FULL_EVAL:
            subprocess.check_call(EVAL_COMMAND, cwd=CODE_ROOT)
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
