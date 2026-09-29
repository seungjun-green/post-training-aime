"""Build the standalone Colab GPT-5.6 Sol full-text translation notebook."""

import yaml


def openai_cells(root, bundle, markdown, code, bootstrap):
    config = yaml.safe_load((root / "configs/translation_openai.yaml").read_text())
    settings = "\n".join(f"{key} = {value!r}" for key, value in config.items())
    settings += "\n\nCONFIG = {name: globals()[name] for name in " + repr(list(config)) + "}\n"
    settings += "\nDATASETS = " + repr(yaml.safe_load((root / "configs/datasets.yaml").read_text()))
    boot = bootstrap(bundle)
    boot.source = boot.source.replace('/content/lg-korean-aime"', '/content/lg-korean-aime-openai"')
    return [
        markdown("""
        # Korean math translation — OpenAI GPT-5.6 Sol
        Run on **Colab CPU** in a fresh runtime. Add `OPENAI_API_KEY` and `HF_TOKEN`
        to Colab Secrets and enable notebook access. An OpenAI API billing account is required.

        The model receives the **original, unmasked text**, including numbers and formulas.
        Long fields are split only at paragraph boundaries; each request contains its complete
        original chunk. No placeholders or automatic source repairs are used.

        1. Run setup, source download and coverage decontamination.
        2. Run the **smoke cell**: 35 rows, five per dataset, including the longest retained s1K trace.
        3. Open `smoke_test/human_review.html` in the downloaded ZIP and review English/Korean pairs.
        4. Enable the separate **full-run cell** after reviewing the smoke results.
        5. Review the full outputs, particularly all AIME/AMC problems, before enabling publication.

        A default **Run all** runs the paid smoke test only. Full translation and upload default
        to off. The model is exactly `gpt-5.6-sol`; no automatic model fallback is configured.
        This version makes translation calls only, with no second-model semantic review call.
        """),
        markdown("""
        ## 1. Settings and setup
        `REASONING_EFFORT='none'` is the initial translation setting; you can change it before
        starting a new smoke run. Sampling temperature is left at the API default. `SEED` controls
        row selection only. `CHUNK_CHARS=16000` accommodates the known long s1K paragraph.

        **Hard checks:** exact numeric literals as multisets; LaTeX/code spans verbatim in source
        order; final boxed answer. A failed chunk is retried once (`HARD_CHECK_RETRIES=1`). Persistent
        hard failures exclude the row, while successfully translated chunks remain cached. Truncated
        output is discarded and split at safe paragraph boundaries, never accepted as complete.

        **Review warnings:** length ratio, Hangul, paragraph counts and undelimited math symbols.
        These do not trigger retries or exclude rows. Ordered LaTeX checks can reject otherwise
        natural reordering. Number words must remain words (`zero` → `영`, rather than a new `0`).
        Literal checks do not establish semantic correctness.

        Results live in `LG-Korea-AIME-OpenAI/models/gpt-5.6-sol/<settings-id>/`.
        Source download/decontamination is shared within this OpenAI root. Code/settings changes
        select a fresh run folder automatically. Run one notebook instance per root at a time.
        """),
        code(settings),
        code("""
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "datasets==5.0.1", "huggingface-hub==0.36.2", "httpx==0.28.1", "PyYAML==6.0.3"])
        from google.colab import drive, userdata, files
        drive.mount("/content/drive")
        OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
        HF_TOKEN = userdata.get('HF_TOKEN')
        if not OPENAI_API_KEY or not HF_TOKEN:
            raise ValueError("Enable notebook access to OPENAI_API_KEY and HF_TOKEN in Colab Secrets")
        """),
        boot,
        code("""
        from pipeline.datasets import download_sources
        from pipeline.decontamination import prepare_decontamination, print_decontamination_report
        from pipeline.openai_translation import OpenAITranslator, run_model
        from pipeline.api_diagnostics import install_error_details
        from pipeline.publish import publish, archive_outputs
        install_error_details(OpenAITranslator)
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
        ## 2. Smoke test — GPT-5.6 Sol (35 rows)
        This makes paid OpenAI API calls. Review meanings, mathematical conditions and omissions,
        as well as automatic failures. `failed_chunks.jsonl` retains rejected responses with their
        original English and failure reasons. `api_usage.jsonl` records response model IDs, response
        IDs and token usage, including returned retries/truncations. API keys are never written there.
        HTTP failures display the API's error message, code, parameter and request ID with key
        redaction, and save those fields to `api_errors.jsonl`. This diagnostic update preserves
        the existing model run folder and completed chunks.

        If interrupted, rerun: completed rows and chunks resume from Drive. Authentication, quota
        and persistent service failures stop the run rather than marking every row as invalid.
        """),
        code("""
        # @title Smoke test GPT-5.6 Sol (35 rows)
        smoke_result = await run_model(
            CONFIG, DATASETS, data, clean, decontamination_report,
            smoke=True, api_key=OPENAI_API_KEY)
        smoke_root = Path(smoke_result["config"]["PROJECT_ROOT"])
        smoke_zip = archive_outputs(smoke_root, "smoke_test")
        print("Smoke only. Open smoke_test/human_review.html in this ZIP:", smoke_zip)
        files.download(str(smoke_zip))
        """),
        markdown("""
        ## 3. Full dataset translation — off until reviewed
        Set `RUN_FULL_TRANSLATION=True` only after reviewing the GPT-5.6 Sol smoke output.
        This runs the same model/settings. A completed matching smoke run is required.
        This cell can run directly after review; after a runtime restart, first rerun setup and
        decontamination. Matching smoke rows and successful chunks are reused.

        In the full archive, `translations/human_review.html` lists every AIME 2024–2026 and
        AMC 2023 problem first (130 total source problems), plus flagged MATH-500/training rows.
        Soft warnings leave rows eligible for publication; review them before uploading.
        """),
        code("""
        # @title Full translation with GPT-5.6 Sol — run after smoke review
        RUN_FULL_TRANSLATION = False # @param {type:"boolean"}
        full_result = None
        if not RUN_FULL_TRANSLATION:
            print("Full translation is off. Review the smoke output, then enable this cell.")
        else:
            full_result = await run_model(
                CONFIG, DATASETS, data, clean, decontamination_report,
                smoke=False, api_key=OPENAI_API_KEY, reviewed=True)
            full_root = Path(full_result["config"]["PROJECT_ROOT"])
            full_zip = archive_outputs(full_root, "translations")
            print("Full translation complete. Review translations/human_review.html:", full_zip)
            print("After publication, baseline PROJECT_ROOT:", str(full_root))
            files.download(str(full_zip))
        """),
        markdown("""
        ## 4. Publish reviewed full outputs
        Enable `UPLOAD_DATASETS` after review. This updates the same seven configured Hugging Face
        repositories, retaining original columns and source cards. Hard-failing rows are excluded;
        soft warnings alone do not exclude rows. The existing source-license policy still applies.
        Copy the printed model run root into the baseline notebook's `PROJECT_ROOT` after publication
        so it reads this run's uploaded Korean datasets through `eval_suite.json`.
        """),
        code("""
        # @title Publish the reviewed full translation
        UPLOAD_DATASETS = False # @param {type:"boolean"}
        if not UPLOAD_DATASETS:
            print("Upload is off.")
        elif full_result is None:
            raise ValueError("Complete the full-run cell first; rerunning resumes saved work.")
        else:
            full_config = full_result["config"]
            uploads = publish(full_result["accepted"], DATASETS, full_config,
                decontamination_report, full_result["checks"], full_result["manifest"], HF_TOKEN)
            print(uploads)
            print("Baseline PROJECT_ROOT:", full_config["PROJECT_ROOT"])
            files.download(str(archive_outputs(Path(full_config["PROJECT_ROOT"]), "translations")))
        """),
        markdown("""
        API references: [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
        [Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
        Requests use `store=False`. Hosted model IDs are not a guarantee of immutable weights.
        Usage totals cover returned responses in each mode; requests with lost responses can still
        be billable. Check the OpenAI dashboard for actual billing.
        """),
    ]
