"""Cells for the separate local-GPU Qwen translation notebook."""

import yaml


def qwen_cells(root, bundle, markdown, code, bootstrap):
    config = yaml.safe_load((root / "configs/translation_qwen.yaml").read_text())
    config_source = "\n".join(f"{k} = {v!r}" for k, v in config.items())
    config_source += "\n\nCONFIG = {name: globals()[name] for name in " + repr(list(config)) + "}\n"
    config_source += "\nDATASETS = " + repr(
        yaml.safe_load((root / "configs/datasets.yaml").read_text())
    )
    boot = bootstrap(bundle)
    boot.source = boot.source.replace('/content/lg-korean-aime"', '/content/lg-korean-aime-qwen"')
    return [
        markdown("""
        # Korean math translation with local Qwen3-32B
        Connect Colab to your **RTX PRO 6000 Blackwell 96GB** GPU runtime.
        This notebook runs `Qwen/Qwen3-32B` locally in BF16 with **thinking disabled**.
        No Anthropic or OpenAI API key is used. Add only `HF_TOKEN` (write scope) to Colab Secrets.
        GPU rental still costs money. Allow roughly 70GB for model weights plus package/download space.

        Start with `SMOKE_TEST=True`: **5 rows × 7 datasets = 35 rows**, including the longest
        retained s1K reasoning trace. Download and review `smoke_review.html` and the checks first.
        Then set `SMOKE_TEST=False` and `SMOKE_REVIEWED=True` and rerun.
        This is translation only; no training or baseline evaluation runs here.

        Outputs use a separate `LG-Korea-AIME-Qwen/runs/<settings-id>/` Drive folder.
        Existing Claude outputs remain untouched and are not reused. Matching Qwen rows resume by ID.
        Full mode uploads to the same configured seven HF dataset names; it updates those repositories.
        Do not run two notebooks against the same output folder simultaneously.
        """),
        markdown("""
        ## 1. Configuration
        `MODEL_REVISION=None` resolves and pins a commit on first use; restarts reuse it.
        Changed model/generation settings create a separate run folder and require a new smoke test.
        `CHUNK_CHARS=16000` accommodates the known 13,453-character s1K paragraph.
        A token-budget check also guards the 32K context; paragraph/math blocks are never silently cut.
        Sampling follows Qwen's non-thinking defaults: temperature 0.7, top-p 0.8, top-k 20.
        """),
        code(config_source),
        code("""
        import os, subprocess, sys
        from pathlib import Path
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"
        ], text=True).strip().splitlines()[0]
        print("GPU 0:", gpu)
        if float(gpu.rsplit(",", 1)[1]) < 90000:
            raise RuntimeError("This BF16 configuration needs the 96GB GPU; connect that runtime first")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "uv==0.11.22", "datasets==5.0.1", "huggingface-hub==0.36.2",
            "transformers==4.57.6", "httpx==0.28.1", "PyYAML==6.0.3"])
        from google.colab import drive, userdata
        drive.mount("/content/drive")
        HF_TOKEN = userdata.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Add HF_TOKEN to Colab Secrets and allow notebook access")
        """),
        boot,
        code("""
        # The GPU server gets its own Python 3.12 environment, even if Colab uses 3.13.
        # This avoids replacing Colab's loaded torch/CUDA libraries.
        GPU_ENV = Path("/content/qwen-gpu-env")
        GPU_PYTHON = GPU_ENV / "bin/python"
        if not GPU_PYTHON.exists():
            subprocess.check_call([sys.executable, "-m", "uv", "venv",
                                   "--python", "3.12", str(GPU_ENV)])
        subprocess.check_call([sys.executable, "-m", "uv", "pip", "sync",
            "--python", str(GPU_PYTHON), str(CODE_ROOT / "requirements-eval.lock")])
        subprocess.check_call([str(GPU_PYTHON), "-c",
            "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); "
            "assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"])
        """),
        code("""
        import json, httpx
        from transformers import AutoTokenizer
        from common.io import write_json
        from pipeline.datasets import download_sources
        from pipeline.decontamination import prepare_decontamination, print_decontamination_report
        from pipeline.decontamination_cache import refresh_translation_cache
        from pipeline.translation import prepare_run, translate_all, check_all
        from pipeline.qwen_translation import (
            configure_run, require_smoke_review, QwenTranslator, write_smoke_review)
        from pipeline.qwen_runtime import start_server, stop_server
        from pipeline.publish import publish, archive_outputs

        # Build from the visible settings each time, so rerunning this cell never nests run folders.
        CONFIG = {name: globals()[name] for name in CONFIG if name in globals()}
        CONFIG["PROJECT_ROOT"] = PROJECT_ROOT
        CONFIG["MODEL_REVISION"] = MODEL_REVISION
        CONFIG = configure_run(CONFIG, HF_TOKEN)
        ROOT = Path(CONFIG["PROJECT_ROOT"])
        require_smoke_review(CONFIG)
        print("Pinned Qwen commit:", CONFIG["MODEL_REVISION"])
        print("Run / resume folder:", ROOT)
        print("After upload, set the baseline notebook's PROJECT_ROOT to:", str(ROOT))
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["TRANSLATION_MODEL"],
            revision=CONFIG["MODEL_REVISION"], token=HF_TOKEN)
        """),
        markdown("## 2. Download the pinned English sources"),
        code("data, source_manifest = download_sources(DATASETS, ROOT, HF_TOKEN)"),
        markdown("""
        ## 3. Coverage-based decontamination (v2)
        Uses distinct 8-gram coverage **per training/eval pair**, with a 70% removal threshold.
        Evaluation sets are unchanged. The report includes per-eval counts, near misses,
        future-contest warnings for s1K and the ten lowest-coverage removals.
        Smoke and full translation both select from these cleaned outputs.
        """),
        code("""
        clean, decontamination_report = prepare_decontamination(
            data, DATASETS, ROOT, ngram_size=DECONTAM_NGRAM_SIZE,
            coverage_threshold=DECONTAM_COVERAGE_THRESHOLD,
            near_miss_min=DECONTAM_NEAR_MISS_MIN)
        print_decontamination_report(decontamination_report)
        refresh_translation_cache(data, clean, DATASETS, CONFIG, decontamination_report)
        selected, run_folder, run_manifest = prepare_run(CONFIG, DATASETS, clean)
        write_json(run_folder / "qwen_runtime.json",
                   json.loads((ROOT / "qwen_runtime.json").read_text()))
        print("SMOKE TEST" if SMOKE_TEST else "FULL TRANSLATION")
        print({name: len(rows) for name, rows in selected.items()})
        """),
        markdown("""
        ## 4. Start the local GPU server
        The first run downloads about 66GB of weights. Startup may take several minutes.
        The server listens only on localhost. It does not call a hosted model provider.
        If loading fails, inspect `qwen_server.log` in the printed Drive run folder.
        Rerunning this cell stops only the server process this notebook started.
        """),
        code("""
        stop_server(globals().get("qwen_server"))
        qwen_server = await start_server(GPU_PYTHON, CONFIG, HF_TOKEN)
        LOCAL_URL = f"http://127.0.0.1:{SERVER_PORT}"
        """),
        markdown("""
        ## 5. Translate and save each row
        Thinking is disabled explicitly on every request. The translator preserves the supplied
        reasoning, equations and answers; it is instructed not to solve or correct problems.
        Truncated output is discarded and retried in smaller paragraph chunks.
        A server failure stops the job; restart it and rerun to reuse saved rows.
        """),
        code("""
        async with httpx.AsyncClient(base_url=LOCAL_URL,
                timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
            translator = QwenTranslator(client, CONFIG, tokenizer)
            await translate_all(selected, DATASETS, run_folder, translator)
        """),
        markdown("""
        ## 6. Quality checks and smoke review
        The existing checks cover Hangul, length ratio, boxed answers and completed chunks.
        Each flagged row is retried once; persistent failures are excluded from upload.
        These automatic checks cannot prove semantic accuracy: review the Korean alongside the English.
        """),
        code("""
        async with httpx.AsyncClient(base_url=LOCAL_URL,
                timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
            translator = QwenTranslator(client, CONFIG, tokenizer)
            accepted, checks_report = await check_all(
                selected, DATASETS, run_folder, run_manifest, translator)
        for name, report in checks_report["datasets"].items():
            print(name, report)
        if SMOKE_TEST:
            review = write_smoke_review(selected, DATASETS, run_folder)
            print("Review all English/Korean pairs in:", review)
        """),
        markdown("""
        ## 7. Download smoke results / publish the full run
        Smoke mode downloads only. Open `smoke_review.html` from the ZIP and inspect every field,
        especially the long traces, math, paragraph joins and any flagged rows.
        To proceed after review, set `SMOKE_TEST=False`, `SMOKE_REVIEWED=True`, and rerun.
        A completed smoke run with matching sources/model/settings is also required.

        Full mode publishes the same seven configured HF dataset repositories and saves `eval_suite.json`.
        The existing unspecified-source-license upload gate remains in place.
        Use the printed run folder as `PROJECT_ROOT` in the existing baseline notebook after upload.
        """),
        code("""
        from google.colab import files
        try:
            if SMOKE_TEST:
                print("Smoke only: no upload. Review smoke_review.html and checks_report.json in the ZIP.")
            else:
                require_smoke_review(CONFIG)
                uploads = publish(accepted, DATASETS, CONFIG, decontamination_report,
                                  checks_report, run_manifest, HF_TOKEN)
                print(uploads)
                print("Baseline PROJECT_ROOT:", str(ROOT))
        finally:
            archive = archive_outputs(ROOT, "smoke_test" if SMOKE_TEST else "translations")
            print("Saved archive:", archive)
            files.download(str(archive))
        """),
        markdown("""
        ## 8. Release GPU memory
        Run this after translation/checks, especially before starting baseline evaluation.
        End the Colab runtime when finished to stop GPU charges.
        """),
        code("stop_server(globals().get('qwen_server'))\nprint('Local Qwen server stopped.')"),
    ]
