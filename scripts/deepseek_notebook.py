"""Build the CPU Colab notebook with separate smoke and full-run cells."""

import yaml


def deepseek_cells(root, bundle, markdown, code, bootstrap):
    config = yaml.safe_load((root / "configs/translation_deepseek.yaml").read_text())
    config_source = "\n".join(f"{k} = {v!r}" for k, v in config.items())
    config_source += "\n\nCONFIG = {name: globals()[name] for name in " + repr(list(config)) + "}\n"
    config_source += "\nDATASETS = " + repr(
        yaml.safe_load((root / "configs/datasets.yaml").read_text())
    )
    boot = bootstrap(bundle)
    boot.source = boot.source.replace(
        '/content/lg-korean-aime"', '/content/lg-korean-aime-deepseek"'
    )
    return [
        markdown("""
        # Korean math translation: compare DeepSeek Flash and Pro
        Run on **Colab CPU**. No local GPU or model download is needed.
        Add `DEEPSEEK_API_KEY` and `HF_TOKEN` (write scope for publication) to Colab Secrets.

        1. Run setup, source download and decontamination below.
        2. Run the **smoke cell**: both `deepseek-flash` and `deepseek-v4-pro` translate the
           **same 35 rows** (5 per dataset, including the longest retained s1K trace).
        3. Download the ZIP and open `smoke_comparison.html`: English, Flash and Pro appear together.
        4. In the separate **full-run cell**, choose a model and enable `RUN_FULL_TRANSLATION`
           after reviewing the smoke results. There is no global smoke/full switch to change.
        5. Publish from the final cell when the full run's checks are complete.

        Both smoke and full cells make paid DeepSeek API calls. Thinking is explicitly disabled.
        A default **Run all** executes only the two smoke tests; full translation and upload stay off.
        Outputs are saved in `LG-Korea-AIME-DeepSeek`, separately for each model and settings version.
        Completed matching rows resume from Drive. Claude/Qwen outputs are not mixed into these runs.
        Run only one instance against this Drive root at a time.
        """),
        markdown("""
        ## 1. Shared settings
        Both models use the same prompt, source rows, chunking, temperature and quality checks.
        `SEED` controls smoke sampling; the API is not assumed to provide deterministic generation.
        `CHUNK_CHARS=16000` accommodates the known long s1K paragraph. Truncations are discarded,
        then split at safe paragraph boundaries. Original math/code and final answers must be preserved.
        Before each request, recognized LaTeX/code blocks and numeric literals (including `82`,
        `92`, `8n`, and `S0`) are replaced with unique placeholders. Code restores the exact source
        literals only after every placeholder appears exactly once. Added/changed numeric notation,
        changed math symbols, and altered protected blocks fail checks and receive one quality retry.
        This does not prove semantic fidelity: review diameter/radius, conditions and omissions.
        These hosted model names are aliases, not immutable weight revisions. Returned model IDs,
        backend fingerprints, timestamps and usage are recorded in each run's `api_usage.jsonl`.
        """),
        code(config_source),
        code("""
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "datasets==5.0.1", "huggingface-hub==0.36.2", "httpx==0.28.1", "PyYAML==6.0.3"])
        from google.colab import drive, userdata, files
        drive.mount("/content/drive")
        DEEPSEEK_API_KEY = userdata.get('DEEPSEEK_API_KEY')
        HF_TOKEN = userdata.get('HF_TOKEN')
        if not DEEPSEEK_API_KEY or not HF_TOKEN:
            raise ValueError("Enable notebook access to DEEPSEEK_API_KEY and HF_TOKEN in Colab Secrets")
        """),
        boot,
        code("""
        from pipeline.datasets import download_sources
        from pipeline.decontamination import prepare_decontamination, print_decontamination_report
        from pipeline.deepseek_translation import MODELS, run_model, comparison_archive
        from pipeline.publish import publish, archive_outputs
        ROOT = Path(PROJECT_ROOT)
        ROOT.mkdir(parents=True, exist_ok=True)
        data, source_manifest = download_sources(DATASETS, ROOT, HF_TOKEN)
        clean, decontamination_report = prepare_decontamination(
            data, DATASETS, ROOT,
            ngram_size=DECONTAM_NGRAM_SIZE,
            coverage_threshold=DECONTAM_COVERAGE_THRESHOLD,
            near_miss_min=DECONTAM_NEAR_MISS_MIN)
        print_decontamination_report(decontamination_report)
        """),
        markdown("""
        ## 2. Smoke test — both models
        This cell translates and checks 35 identical source rows with **each** model.
        Each flagged row is retried once. Persistent failures remain visible in the comparison
        and are excluded from accepted outputs. No datasets are uploaded here.

        If interrupted, rerun this cell: completed rows and completed quality retries are reused.
        API authentication, balance and persistent service errors stop the run rather than flagging
        every row. Source/decontamination changes require fresh checks before full translation.
        **Preservation update:** start this updated notebook in a fresh Colab runtime and rerun setup
        and smoke. The updated code automatically creates new model run folders; old results stay
        untouched and are not reused. Both models need a fresh smoke comparison before selection.
        """),
        code("""
        # @title Smoke test both DeepSeek models (35 rows each)
        smoke_results = {}
        for model in MODELS:
            smoke_results[model] = await run_model(
                CONFIG, DATASETS, data, clean, decontamination_report,
                model=model, smoke=True, api_key=DEEPSEEK_API_KEY)
        comparison_zip = comparison_archive(smoke_results, DATASETS, ROOT)
        print("Smoke only: review smoke_comparison.html and comparison_summary.json in this ZIP.")
        print("Saved comparison:", comparison_zip)
        files.download(str(comparison_zip))
        """),
        markdown("""
        ## 3. Full translation — choose the model here
        Review Korean meaning, terminology, math, omitted paragraphs, reasoning structure and flagged
        rows in the smoke comparison. Then choose the model below and enable `RUN_FULL_TRANSLATION`.
        Both model choices use non-thinking mode. A completed matching smoke test for your chosen
        model is required even if you enable the checkbox.

        You can run **this cell directly** after review; no need to rerun setup or smoke.
        After a runtime restart, rerun setup/decontamination, then this cell; saved smoke evidence is
        loaded from Drive. The selected model's existing smoke translations are reused by ID.
        This cell translates/checks the cleaned full datasets and downloads an archive; publication
        is in the next cell. It never selects a winner automatically.
        """),
        code("""
        # @title Full translation — run after reviewing the smoke comparison
        FULL_MODEL = "deepseek-flash" # @param ["deepseek-flash", "deepseek-v4-pro"]
        RUN_FULL_TRANSLATION = False # @param {type:"boolean"}

        full_result = None
        if not RUN_FULL_TRANSLATION:
            print("Full translation is off. Review the smoke comparison, choose a model, then enable it.")
        else:
            full_result = await run_model(
                CONFIG, DATASETS, data, clean, decontamination_report,
                model=FULL_MODEL, smoke=False, api_key=DEEPSEEK_API_KEY, reviewed=True)
            full_root = Path(full_result["config"]["PROJECT_ROOT"])
            full_archive = archive_outputs(full_root, "translations")
            print("Full translation/checks complete:", full_archive)
            print("After publication, baseline PROJECT_ROOT:", str(full_root))
            files.download(str(full_archive))
        """),
        markdown("""
        ## 4. Publish the selected full run
        Enable `UPLOAD_DATASETS` when ready. This updates the same seven configured Hugging Face
        repositories, retains original columns and source cards, and excludes persistently flagged rows.
        The existing unspecified-source-license gate remains: resolve that setting before publication.
        After upload, copy the printed `PROJECT_ROOT` into the existing baseline notebook. Its
        `eval_suite.json` points to this selected model's uploaded Korean evaluation datasets.
        """),
        code("""
        # @title Publish checked full datasets
        UPLOAD_DATASETS = False # @param {type:"boolean"}
        if not UPLOAD_DATASETS:
            print("Upload is off.")
        elif full_result is None:
            raise ValueError("Complete the full-run cell first (rerunning resumes saved results).")
        else:
            full_config = full_result["config"]
            uploads = publish(full_result["accepted"], DATASETS, full_config,
                decontamination_report, full_result["checks"], full_result["manifest"], HF_TOKEN)
            print(uploads)
            print("Baseline PROJECT_ROOT:", full_config["PROJECT_ROOT"])
            files.download(str(archive_outputs(Path(full_config["PROJECT_ROOT"]), "translations")))
        """),
        markdown("""
        API references: [DeepSeek thinking toggle](https://api-docs.deepseek.com/guides/thinking_mode/),
        [chat completions](https://api-docs.deepseek.com/api/create-chat-completion/),
        [current models and pricing](https://api-docs.deepseek.com/quick_start/pricing/).
        Usage summaries include responses returned for retries and discarded truncated chunks;
        requests whose responses were lost may still be billed. Check the provider dashboard for billing.
        """),
    ]
