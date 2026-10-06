"""Build the thin CPU Colab launcher for OpenR1 filtering and publication."""

from build_notebooks import code, markdown, write_notebook


def cells():
    return [
        markdown("""
        # Prepare OpenR1-Math-220k for Qwen2.5-3B

        Use a **CPU Colab runtime** and enable a write-capable `HF_TOKEN` in Colab Secrets.
        Source: `open-r1/OpenR1-Math-220k`, **default/train** (about 94k problems).
        No GPU, generation, translation, or training is needed.

        Push the new project files yourself before running: setup clones/updates
        `https://github.com/seungjun-green/post-training-aime`.
        Run the cells in order. The final upload cell publishes the prepared dataset.
        Work and checkpoints are saved in Drive; changed preparation settings get a new folder.
        Allow several GB of storage for source downloads, filtered traces and Parquet output.
        """),
        code("""
        REPO_URL = "https://github.com/seungjun-green/post-training-aime.git"
        CODE_ROOT = "/content/openr1-preparation-code"
        PROJECT_ROOT = "/content/drive/MyDrive/OpenR1-Math-Preparation"
        HF_USERNAME = "Seungjun"
        HF_PRIVATE = True
        TOKENIZER = "Qwen/Qwen2.5-3B"
        MAX_TOKENS = 20480
        MISSING_FINISH = "drop"
        """),
        code("""
        import os, subprocess, sys
        from pathlib import Path
        from google.colab import drive, userdata, files
        drive.mount("/content/drive")
        HF_TOKEN = userdata.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Enable a write-capable HF_TOKEN in Colab Secrets")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "datasets==5.0.1", "huggingface-hub==0.36.2", "transformers==4.57.6",
            "tqdm==4.70.1"])
        if not Path(CODE_ROOT).exists():
            subprocess.check_call(["git", "clone", "--branch", "main", REPO_URL, CODE_ROOT])
        def git(*args):
            return subprocess.check_output(["git", "-C", CODE_ROOT, *args], text=True).strip()
        if git("remote", "get-url", "origin") != REPO_URL or git("status", "--porcelain"):
            raise ValueError("Unexpected or edited checkout; use a fresh CODE_ROOT")
        if git("branch", "--show-current") != "main":
            raise ValueError("Expected a main-branch checkout")
        subprocess.check_call(["git", "-C", CODE_ROOT, "pull", "--ff-only", "origin", "main"])
        required = ["pipeline/openr1_preparation.py", "pipeline/decontamination.py",
                    "common/io.py", "configs/english_eval_suite.json"]
        missing = [name for name in required if not (Path(CODE_ROOT) / name).is_file()]
        if missing:
            raise RuntimeError("GitHub main is missing project files: " + ", ".join(missing)
                + ". Commit and push the new project files from your computer, then rerun this cell.")
        sys.path.insert(0, CODE_ROOT)
        import importlib
        importlib.invalidate_caches()
        from pipeline.openr1_preparation import initialize_run, filter_traces, decontaminate_filtered, publish
        print("Project commit:", git("rev-parse", "HEAD"))
        """),
        markdown("""
        ## Step 1 — keep correct, completed traces within 20,480 tokens

        Keep a trace only if `correctness_math_verify` is true, `finish_reasons` is `stop`,
        and `is_reasoning_complete` is not explicitly false. **Null finish reasons are dropped.**
        All parallel arrays are checked for alignment before filtering.

        Length is Qwen2.5-3B tokens(problem) + tokens(generation), with no special tokens.
        Exactly 20,480 is allowed; longer pairs are dropped, never truncated.
        This excludes training chat-template overhead; account for it when packing SFT sequences.

        Keep all qualifying traces for each problem; drop a problem if no trace remains.
        Source text is unchanged, including reasoning tags. The original `messages` field is
        not copied because it may contain a trace that failed filtering.
        A tqdm counter tracks source problems. Batches are checkpointed to Drive.
        """),
        code("""
        RUN_FOLDER = initialize_run(PROJECT_ROOT, HF_TOKEN, tokenizer=TOKENIZER,
            max_tokens=MAX_TOKENS, missing_finish=MISSING_FINISH)
        print("Run folder:", RUN_FOLDER)
        filter_report = filter_traces(RUN_FOLDER, HF_TOKEN)
        """),
        markdown("""
        ## Step 2 — remove training/evaluation overlap

        Reuse the existing normalization-v2, distinct 8-gram index, and **70% per-eval-problem
        coverage** rule. Compare against the pinned AIME 2024/2025/2026, AMC23 and MATH-500
        datasets. All reference rows are checked against their saved content digests.
        Short eval problems use normalized exact substring matching.

        If a problem overlaps, remove the entire problem and all its retained traces.
        Keep `source` for provenance and source-level counts; do not drop entire source categories.
        This does not remove repeated problems within the training dataset.
        Report every qualifying match with coverage and kept near misses at 50–70% coverage.
        No evaluation dataset is modified or uploaded.
        """),
        code("""
        decontamination_report = decontaminate_filtered(RUN_FOLDER, HF_TOKEN)
        print("Target:", f"{HF_USERNAME}/dp_removed_OpenR1-Math-220k")
        print("Prepared counts:", decontamination_report["counts"])
        """),
        markdown("""
        ## Step 3 — upload to Hugging Face

        This cell uploads **`dp_removed_OpenR1-Math-220k`** under your username, private by default.
        Files are checksum-verified first. An identical rerun is allowed; the uploader refuses
        to replace a repository containing a different preparation run or unknown contents.

        Schema: one row per retained problem, `generations` as a list of retained targets,
        aligned correctness/finish/token-count lists, original generation indices, source and IDs.
        Iterate over `generations` to create individual SFT examples. Other original columns are omitted.
        The dataset card, source card, filtering report, decontamination report and pinned provenance
        are uploaded with the Parquet data. The upload receipt records the exact Hub revision.
        """),
        code("""
        upload = publish(RUN_FOLDER, HF_USERNAME, HF_PRIVATE, HF_TOKEN)
        """),
        code("""
        from zipfile import ZipFile, ZIP_DEFLATED
        archive = RUN_FOLDER / "openr1_preparation_reports.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED) as z:
            for name in ["manifest.json", "upload_receipt.json", "export/filter_report.json",
                         "export/decontamination_report.json", "export/README.md"]:
                z.write(RUN_FOLDER / name, name)
        files.download(str(archive))
        """),
    ]


if __name__ == "__main__":
    write_notebook("prepare_openr1_math_220k.ipynb", cells())
