"""Standalone Colab notebook for local EXAONE full-text translation."""

import yaml


def exaone_cells(root, bundle, markdown, code, bootstrap):
    config = yaml.safe_load((root / "configs/translation_exaone.yaml").read_text())
    settings = "\n".join(f"{k} = {v!r}" for k, v in config.items())
    settings += "\nCONFIG = {name: globals()[name] for name in " + repr(list(config)) + "}\n"
    settings += "\nDATASETS = " + repr(yaml.safe_load((root / "configs/datasets.yaml").read_text()))
    boot = bootstrap(bundle)
    boot.source = boot.source.replace('/content/lg-korean-aime"', '/content/lg-korean-aime-exaone"')
    return [
        markdown("""
        # Korean math translation — local EXAONE 4.5-33B
        Connect Colab to your **RTX PRO 6000 Blackwell 96GB** runtime.
        Add only `HF_TOKEN` to Colab Secrets (write scope is needed for later publication).
        Translation runs on your GPU, with **thinking disabled** and **original, unmasked text**.
        No hosted translation API or translation API key is used. GPU rental still costs money.

        Run setup, then the **35-row smoke test**. Review the downloaded English/Korean pairs.
        Enable the separate full-run cell only after review; publication has its own switch.
        A default Run all runs only smoke translation, then stops the GPU server.
        This notebook does not train or evaluate the 2.4B student model.

        Plan for at least 100GB free runtime disk for weights and packages. Startup downloads
        roughly 66GB of weights. GPU compatibility and translation quality must be verified
        by this smoke run; this notebook has not been executed on your Colab GPU.
        """),
        markdown("""
        ## 1. Configuration
        Model: `LGAI-EXAONE/EXAONE-4.5-33B`. First setup pins a Hugging Face commit in Drive;
        restarts reuse it. Text-only sampling starts at LG's temperature 1.0 / top-p 0.95.
        Long fields split only at safe paragraph boundaries, with a token-budget check before
        generation. Full text means no math/number masking; long traces can span several requests.

        Numbers, ordered LaTeX/code spans and boxed answers are hard checks: retry only the failed
        chunk once, then flag the row if it still fails. Length/Hangul/wording signals are review
        warnings. There is no second-model semantic reviewer: inspect meanings yourself.

        Outputs: `LG-Korea-AIME-EXAONE/runs/<settings-id>/`. Matching chunks and rows resume from
        Drive. Changed model/prompt/generation settings create a separate run requiring fresh smoke
        review. Run one notebook per output root. Other provider outputs are not reused.
        """),
        code(settings),
        code("""
        import subprocess, sys
        from pathlib import Path
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"
        ], text=True).strip().splitlines()[0]
        print("GPU 0:", gpu)
        if float(gpu.rsplit(",", 1)[1]) < 90000:
            raise RuntimeError("Connect the 96GB GPU runtime before running this BF16 notebook")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "uv==0.11.22", "datasets==5.0.1", "huggingface-hub==0.36.2",
            "httpx==0.28.1", "PyYAML==6.0.3"])
        from google.colab import drive, userdata, files
        drive.mount("/content/drive")
        HF_TOKEN = userdata.get('HF_TOKEN')
        if not HF_TOKEN:
            raise ValueError("Enable notebook access to HF_TOKEN in Colab Secrets")
        """),
        boot,
        markdown("""
        ## 2. Install the separate GPU runtime
        Uses Python 3.12, vLLM 0.20.2 and Transformers 5.8.0 with resolved dependencies.
        This is separate from Colab's Python and the frozen baseline runtime.
        The CUDA/driver check below must pass before downloading model weights.
        """),
        code("""
        GPU_ENV = Path("/content/exaone-gpu-env")
        GPU_PYTHON = GPU_ENV / "bin/python"
        if not GPU_PYTHON.exists():
            subprocess.check_call([sys.executable, "-m", "uv", "venv", "--python", "3.12", str(GPU_ENV)])
        subprocess.check_call([sys.executable, "-m", "uv", "pip", "sync",
            "--python", str(GPU_PYTHON), str(CODE_ROOT / "requirements-exaone.lock")])
        subprocess.check_call([str(GPU_PYTHON), "-c",
            "import torch, transformers, vllm; "
            "print('torch', torch.__version__, 'CUDA', torch.version.cuda, "
            "'transformers', transformers.__version__, 'vllm', vllm.__version__); "
            "assert torch.cuda.is_available(), 'GPU or CUDA driver unavailable'; "
            "x=torch.ones(1, device='cuda'); print(torch.cuda.get_device_name(0), x.item()); "
            "from vllm.model_executor.models.registry import ModelRegistry; "
            "assert 'Exaone4_5_ForConditionalGeneration' in ModelRegistry.get_supported_archs()"])
        """),
        code("""
        from common.io import write_json
        from pipeline.datasets import download_sources
        from pipeline.decontamination import prepare_decontamination, print_decontamination_report
        from pipeline.exaone_translation import pin_model, run_model
        from pipeline.exaone_runtime import start_server, stop_server
        from pipeline.publish import publish, archive_outputs
        CONFIG = pin_model(CONFIG, HF_TOKEN)
        ROOT = Path(CONFIG["PROJECT_ROOT"])
        ROOT.mkdir(parents=True, exist_ok=True)
        print("Pinned model:", CONFIG["TRANSLATION_MODEL"], CONFIG["MODEL_REVISION"])
        data, source_manifest = download_sources(DATASETS, ROOT, HF_TOKEN)
        clean, decontamination_report = prepare_decontamination(
            data, DATASETS, ROOT, ngram_size=DECONTAM_NGRAM_SIZE,
            coverage_threshold=DECONTAM_COVERAGE_THRESHOLD, near_miss_min=DECONTAM_NEAR_MISS_MIN)
        print_decontamination_report(decontamination_report)
        """),
        markdown("""
        ## 3. Start the GPU server
        Startup can take several minutes; progress prints every 30 seconds.
        The server listens only on localhost. Logs: `LG-Korea-AIME-EXAONE/exaone_server.log`.
        If you already ran the cleanup cell, rerun this cell before full translation.
        """),
        code("""
        stop_server(globals().get("exaone_server"))
        exaone_server = await start_server(GPU_PYTHON, CONFIG, HF_TOKEN)
        environment = subprocess.check_output([
            sys.executable, "-m", "uv", "pip", "freeze", "--python", str(GPU_PYTHON)
        ], text=True)
        (ROOT / "gpu_environment.txt").write_text(environment)
        """),
        markdown("""
        ## 4. Smoke test — 35 rows
        Five rows per dataset, including the longest retained s1K trace. Both smoke and full
        selection use the coverage-decontaminated outputs; eval sets remain unchanged.
        Open `smoke_test/human_review.html` in the ZIP. Review all pairs, particularly 82/92,
        diameter/radius, ordered triples, and joined conditions. Automated checks cannot prove
        semantic correctness. `failed_chunks.jsonl` retains rejected outputs for inspection.
        Interrupted runs reuse completed chunks/rows; server errors stop the run.
        """),
        code("""
        # @title Smoke test EXAONE 4.5-33B (35 rows)
        smoke_result = await run_model(CONFIG, DATASETS, data, clean, decontamination_report, smoke=True)
        smoke_root = Path(smoke_result["config"]["PROJECT_ROOT"])
        smoke_zip = archive_outputs(smoke_root, "smoke_test")
        print("Open smoke_test/human_review.html:", smoke_zip)
        files.download(str(smoke_zip))
        """),
        markdown("""
        ## 5. Full translation — off until smoke review
        Uses exactly the model/settings tested above, reusing matching smoke rows and chunks.
        After a runtime restart, rerun setup and server startup. A completed matching smoke run
        is required. Review all 130 AIME/AMC problems and flagged rows in the full review HTML.
        Persistent hard failures are excluded from accepted outputs; inspect these before upload.
        """),
        code("""
        # @title Full translation — enable after reviewing smoke output
        RUN_FULL_TRANSLATION = False # @param {type:"boolean"}
        full_result = None
        if not RUN_FULL_TRANSLATION:
            print("Full translation is off. Review smoke results before enabling it.")
        else:
            full_result = await run_model(CONFIG, DATASETS, data, clean, decontamination_report,
                                          smoke=False, reviewed=True)
            full_root = Path(full_result["config"]["PROJECT_ROOT"])
            full_zip = archive_outputs(full_root, "translations")
            print("Review translations/human_review.html:", full_zip)
            files.download(str(full_zip))
        """),
        markdown("""
        ## 6. Publish reviewed results — off by default
        Updates the same seven configured Hugging Face datasets. Existing source-license handling,
        dataset formats and baseline evaluation remain unchanged. Soft warnings do not exclude rows.
        Use the printed run root as the baseline notebook's PROJECT_ROOT after publication.
        """),
        code("""
        # @title Publish the reviewed full translation
        UPLOAD_DATASETS = False # @param {type:"boolean"}
        if not UPLOAD_DATASETS:
            print("Upload is off.")
        elif full_result is None:
            raise ValueError("Complete the full-run cell first; rerunning resumes saved work")
        else:
            uploads = publish(full_result["accepted"], DATASETS, full_result["config"],
                decontamination_report, full_result["checks"], full_result["manifest"], HF_TOKEN)
            print(uploads)
            print("Baseline PROJECT_ROOT:", full_result["config"]["PROJECT_ROOT"])
            files.download(str(archive_outputs(Path(full_result["config"]["PROJECT_ROOT"]), "translations")))
        """),
        markdown("""
        ## 7. Release GPU memory
        Stops the server started by this notebook. Rerun server startup to translate again.
        Disconnect/delete the Colab runtime when finished to stop GPU rental charges.
        """),
        code("""
        stop_server(globals().get("exaone_server"))
        print("EXAONE server stopped.")
        """),
        markdown("""
        References: [LG model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B),
        [vLLM EXAONE support](https://docs.vllm.ai/en/v0.20.2/api/vllm/model_executor/models/exaone4_5/).
        Translation records identify the pinned model commit. `api_usage.jsonl` contains local GPU
        generation token counts; these are not hosted API charges.
        """),
    ]
