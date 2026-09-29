"""CPU-only English decontamination and publication notebook."""

import yaml


def english_cells(root, bundle, markdown, code, bootstrap):
    registry = yaml.safe_load((root / "configs/datasets.yaml").read_text())
    boot = bootstrap(bundle)
    boot.source = boot.source.replace('/content/lg-korean-aime"', '/content/lg-aime-english"')
    names = "\n".join(
        f"- `Seungjun/dp_removed_{s['repo'].split('/')[-1]}`" for s in registry.values()
    )
    return [
        markdown("""
        # English datasets: remove train/eval overlap and upload
        Run on **Colab CPU**. Add a write-capable `HF_TOKEN` to Colab Secrets.
        This notebook downloads the seven pinned source splits, removes overlapping training
        rows using the existing coverage rule, and uploads the English datasets to Hugging Face.
        All retained original columns and text are preserved. Evaluation rows remain unchanged.
        This does not separately remove repeated training problems within/between training sets.

        The base model remains `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`; this notebook performs data
        preparation only. No GPU, translation model, API key for translation, or smoke test is needed.
        Running all cells uploads the prepared datasets (private by default).
        """),
        code(
            """
        PROJECT_ROOT = "/content/drive/MyDrive/LG-AIME-English"
        HF_USERNAME = "Seungjun" # @param {type:"string"}
        HF_PRIVATE = True # @param {type:"boolean"}
        DECONTAM_NGRAM_SIZE = 8
        DECONTAM_COVERAGE_THRESHOLD = 0.7
        DECONTAM_NEAR_MISS_MIN = 0.5
        """
            + "\n        DATASETS = "
            + repr(registry)
        ),
        markdown(
            """
        Names use the exact source repository name (without its owner), prefixed by `dp_removed_`:
        \n"""
            + names
            + """

        For example, the source `math-ai/aime25` becomes `dp_removed_aime25`.
        Only the configured source config/split is copied, including DAPO's `en` config.
        Existing translation repositories are separate. Rerunning updates these same destinations.
        """
        ),
        code("""
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "datasets==5.0.1", "huggingface-hub==0.36.2", "PyYAML==6.0.3"])
        from google.colab import drive, userdata, files
        drive.mount("/content/drive")
        HF_TOKEN = userdata.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Add HF_TOKEN to Colab Secrets and enable notebook access")
        """),
        boot,
        code("""
        from pipeline.datasets import download_sources
        from pipeline.decontamination import prepare_decontamination, print_decontamination_report
        from pipeline.english_publish import stage_english, publish_english, upload_name
        ROOT = Path(PROJECT_ROOT)
        ROOT.mkdir(parents=True, exist_ok=True)
        data, source_manifest = download_sources(DATASETS, ROOT, HF_TOKEN)
        """),
        markdown("""
        ## Remove training/evaluation overlap
        Remove a training row if it covers at least 70% of any one eval problem's distinct
        normalized 8-grams. Eval problems shorter than eight tokens use exact normalized
        substring matching. This is the existing v2 rule, not the old one-shared-8-gram rule.
        Normalization is only for comparison: uploaded text is untouched.
        The report includes per-eval removal counts, near misses and borderline pairs.
        """),
        code("""
        clean, decontamination_report = prepare_decontamination(
            data, DATASETS, ROOT, ngram_size=DECONTAM_NGRAM_SIZE,
            coverage_threshold=DECONTAM_COVERAGE_THRESHOLD, near_miss_min=DECONTAM_NEAR_MISS_MIN)
        print_decontamination_report(decontamination_report)
        print("\\nUpload summary:")
        for name, spec in DATASETS.items():
            print(f"{HF_USERNAME}/{upload_name(spec)}: {len(data[name])} -> {len(clean[name])} "
                  f"({len(data[name]) - len(clean[name])} removed); {spec['role']}")
        """),
        markdown("""
        ## Prepare original English rows
        Stages Parquet files with the source schema, original dataset cards, provenance IDs and
        the decontamination report. No `ko_` fields or synthetic data columns are added.
        Source licenses are retained; unspecified licenses remain unspecified.
        """),
        code("""
        export_manifest = stage_english(data, clean, DATASETS, source_manifest,
                                        decontamination_report, ROOT)
        print("Prepared files:", export_manifest["folder"])
        """),
        markdown("""
        ## Upload to Hugging Face
        This cell uploads all seven datasets under the names printed above. It verifies every
        staged file first and records exact Hub commit IDs in `uploads.json` and
        `english_dataset_suite.json`. Rerunning safely republishes the same prepared content.
        """),
        code("""
        uploads = publish_english(export_manifest, HF_USERNAME, HF_PRIVATE, HF_TOKEN)
        for name, item in uploads.items():
            print(item["url"])
        """),
        code("""
        # Download a small record of this preparation/upload (data remains in Drive and on HF).
        from zipfile import ZipFile, ZIP_DEFLATED
        export_root = Path(export_manifest["folder"])
        archive = ROOT / "english_preparation_reports.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED) as z:
            for filename in ["manifest.json", "uploads.json", "english_dataset_suite.json"]:
                z.write(export_root / filename, filename)
            report_path = ROOT / decontamination_report["cache_directory"] / "decontamination_report.json"
            z.write(report_path, "decontamination_report.json")
        files.download(str(archive))
        """),
        markdown("""
        Load any published dataset using its recorded revision, config and split:
        ```python
        from datasets import load_dataset
        item = uploads["s1k_1.1"]
        ds = load_dataset(item["repo"], name=item["config"], split=item["split"],
                          revision=item["revision"], token=HF_TOKEN)
        ```
        These are English datasets. The previous Korean baseline notebook expects Korean columns
        and prompts; it should be adapted before evaluating this new English pipeline.
        """),
    ]
