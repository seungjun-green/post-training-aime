"""Build standalone Colab notebooks from the tested modules (no external repo needed)."""

import argparse
import base64
import io
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source, collapsed=False):
    cell = nbformat.v4.new_code_cell(textwrap.dedent(source).strip())
    if collapsed:
        cell.metadata["cellView"] = "form"
    return cell


def payload(paths):
    content = io.BytesIO()
    with ZipFile(content, "w", ZIP_DEFLATED) as archive:
        for path in paths:
            info = ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return base64.b64encode(content.getvalue()).decode()


def bootstrap(encoded):
    return code(
        f"""
        # @title Install the bundled, tested project code (self-contained notebook)
        import base64, io, sys, zipfile
        from pathlib import Path
        CODE_ROOT = Path("/content/lg-korean-aime")
        CODE_ROOT.mkdir(parents=True, exist_ok=True)
        BUNDLE = {encoded!r}
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as bundle:
            bundle.extractall(CODE_ROOT)
        sys.path.insert(0, str(CODE_ROOT))
        print("Project code ready:", CODE_ROOT)
    """,
        collapsed=True,
    )


def write_notebook(name, cells, selected=None):
    if selected is not None and name != selected:
        return
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        language_info={"name": "python"},
        colab={"name": name},
    )
    # Stable cell IDs keep notebook rebuilds reviewable.
    for index, cell in enumerate(nb.cells):
        cell.id = f"cell-{index:02d}"
    nbformat.validate(nb)
    nbformat.write(nb, ROOT / "notebooks" / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        choices=[
            "translate_datasets.ipynb",
            "evaluate_baseline.ipynb",
            "translate_datasets_qwen.ipynb",
            "translate_datasets_deepseek.ipynb",
            "translate_datasets_openai.ipynb",
            "translate_datasets_exaone.ipynb",
            "prepare_english_datasets.ipynb",
            "evaluate_baseline_english.ipynb",
        ],
    )
    args = parser.parse_args()
    if args.notebook in {None, "evaluate_baseline_english.ipynb"}:
        from english_eval_notebook import english_eval_cells

        write_notebook("evaluate_baseline_english.ipynb", english_eval_cells(markdown, code))
        if args.notebook == "evaluate_baseline_english.ipynb":
            return
    if args.notebook in {None, "prepare_english_datasets.ipynb"}:
        from english_notebook import english_cells

        bundle = payload(
            [
                ROOT / p
                for p in [
                    "common/__init__.py",
                    "common/io.py",
                    "pipeline/__init__.py",
                    "pipeline/datasets.py",
                    "pipeline/decontamination.py",
                    "pipeline/english_publish.py",
                    "configs/datasets.yaml",
                ]
            ]
        )
        write_notebook(
            "prepare_english_datasets.ipynb", english_cells(ROOT, bundle, markdown, code, bootstrap)
        )
        if args.notebook == "prepare_english_datasets.ipynb":
            return
    paths = [
        p for folder in ["common", "pipeline", "eval"] for p in sorted((ROOT / folder).glob("*.py"))
    ]
    paths += sorted((ROOT / "configs").glob("*.yaml"))
    paths += [ROOT / "pyproject.toml"]
    gpu_lock = ROOT / "requirements-eval.lock"
    if gpu_lock.exists():
        paths.append(gpu_lock)
    paths.append(ROOT / "requirements-exaone.lock")
    bundle = payload(paths)
    if args.notebook in {None, "translate_datasets_exaone.ipynb"}:
        from exaone_notebook import exaone_cells

        write_notebook(
            "translate_datasets_exaone.ipynb",
            exaone_cells(ROOT, bundle, markdown, code, bootstrap),
            selected=args.notebook,
        )
    if args.notebook == "translate_datasets_exaone.ipynb":
        return
    if args.notebook in {None, "translate_datasets_openai.ipynb"}:
        from openai_notebook import openai_cells

        write_notebook(
            "translate_datasets_openai.ipynb",
            openai_cells(ROOT, bundle, markdown, code, bootstrap),
            selected=args.notebook,
        )
    if args.notebook == "translate_datasets_openai.ipynb":
        return
    if args.notebook in {None, "translate_datasets_deepseek.ipynb"}:
        from deepseek_notebook import deepseek_cells

        write_notebook(
            "translate_datasets_deepseek.ipynb",
            deepseek_cells(ROOT, bundle, markdown, code, bootstrap),
            selected=args.notebook,
        )
    if args.notebook == "translate_datasets_deepseek.ipynb":
        return
    if args.notebook in {None, "translate_datasets_qwen.ipynb"}:
        from qwen_notebook import qwen_cells

        write_notebook(
            "translate_datasets_qwen.ipynb",
            qwen_cells(ROOT, bundle, markdown, code, bootstrap),
            selected=args.notebook,
        )
    if args.notebook == "translate_datasets_qwen.ipynb":
        return
    config = yaml.safe_load((ROOT / "configs/translation.yaml").read_text())
    config_source = "\n".join(f"{k} = {v!r}" for k, v in config.items())
    config_source += "\n\nCONFIG = {name: globals()[name] for name in " + repr(list(config)) + "}\n"
    registry = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text())
    config_source += "\n# Pinned dataset schema/config/revision settings\nDATASETS = " + repr(
        registry
    )
    write_notebook(
        "translate_datasets.ipynb",
        [
            markdown("""
            # Korean math datasets: prepare, smoke test, translate, publish
            Run on **Colab CPU**. This notebook bundles the tested project modules and needs no repository checkout.
            Add `HF_TOKEN` (write scope) and `ANTHROPIC_API_KEY` to Colab Secrets and allow notebook access.
            Start with `SMOKE_TEST = True`: all seven datasets are sampled, including the longest retained s1K trace.
            Review the downloaded outputs before manually changing `SMOKE_TEST` to `False` and rerunning.
            The full run makes paid API requests and publishes seven private datasets under `Seungjun/`.
            Source revisions, IDs, translation settings and all quality-check exclusions are recorded in Drive.
            Only Spec 1 is implemented. Do not run this notebook simultaneously against the same Drive root.
        """),
            markdown(
                "## 1. Setup and configuration\nAll run settings are in the next cell; no secrets belong in it."
            ),
            code(config_source),
            code("""
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "datasets==5.0.1", "huggingface-hub==0.36.2",
                                   "anthropic==1.8.0", "PyYAML==6.0.3"])
            from google.colab import drive, userdata
            drive.mount("/content/drive")
            HF_TOKEN = userdata.get("HF_TOKEN")
            ANTHROPIC_API_KEY = userdata.get("ANTHROPIC_API_KEY")
            if not HF_TOKEN or not ANTHROPIC_API_KEY:
                raise ValueError("Both Colab Secrets are required")
        """),
            bootstrap(bundle),
            code("""
            from anthropic import AsyncAnthropic
            from common.io import write_json
            from pipeline.datasets import download_sources
            from pipeline.decontamination import prepare_decontamination
            from pipeline.translation import Translator, prepare_run, translate_all, check_all
            from pipeline.publish import publish, archive_outputs
            ROOT = Path(PROJECT_ROOT)
            ROOT.mkdir(parents=True, exist_ok=True)
            write_json(ROOT / "translation_config.json", CONFIG)
            assert TRANSLATION_MODEL and TRANSLATION_MODEL != "TODO"
        """),
            markdown(
                "## 2. Download all datasets\nInspect schemas, preserve source rows, and cache pinned revisions in Drive."
            ),
            code("data, source_manifest = download_sources(DATASETS, ROOT, HF_TOKEN)"),
            markdown(
                "## 3. English decontamination\nRemove training rows only when one evaluation problem reaches the configured distinct 8-gram coverage threshold. Normalization v2 drops punctuation-only tokens. Versioned caches preserve old outputs. Review per-eval counts, future-contest warnings, borderline removals and near misses below. Evaluation rows stay unchanged."
            ),
            code("""
            from pipeline.decontamination import print_decontamination_report
            from pipeline.decontamination_cache import refresh_translation_cache

            clean, decontamination_report = prepare_decontamination(
                data, DATASETS, ROOT,
                ngram_size=DECONTAM_NGRAM_SIZE,
                coverage_threshold=DECONTAM_COVERAGE_THRESHOLD,
                near_miss_min=DECONTAM_NEAR_MISS_MIN,
            )
            print_decontamination_report(decontamination_report)
            refresh_translation_cache(data, clean, DATASETS, CONFIG, decontamination_report)
        """),
            markdown("""
            ## 4. Translation
            Async requests use a semaphore, bounded retries and immediate JSONL persistence.
            Truncated chunks are discarded and split smaller at safe boundaries.
            Authentication/model errors stop the run. Completed rows are reused after a runtime restart.
            `claude-opus-5-5` is the requested model ID; the API must grant your account access to it.
        """),
            code("""
            selected, run_folder, run_manifest = prepare_run(CONFIG, DATASETS, clean)
            async with AsyncAnthropic(api_key=ANTHROPIC_API_KEY, max_retries=0,
                                      timeout=REQUEST_TIMEOUT_SECONDS) as client:
                translator = Translator(client, CONFIG)
                await translate_all(selected, DATASETS, run_folder, translator)
        """),
            markdown(
                "## 5. Automated checks\nRetry each flagged row once; persistent failures are saved and excluded from upload."
            ),
            code("""
            async with AsyncAnthropic(api_key=ANTHROPIC_API_KEY, max_retries=0,
                                      timeout=REQUEST_TIMEOUT_SECONDS) as client:
                translator = Translator(client, CONFIG)
                accepted, checks_report = await check_all(
                    selected, DATASETS, run_folder, run_manifest, translator)
            for name, report in checks_report["datasets"].items():
                print(name, report)
        """),
            markdown("""
            ## 6. Assemble, upload, archive and download
            In smoke mode this cell **only archives and downloads**; it never uploads or launches a full run.
            Review the Korean text, math, reasoning structure, chunk joins and flagged rows in the archive.
            After review, manually set `SMOKE_TEST = False` above and rerun the notebook.
            Full mode preserves original columns, excludes failed rows, writes source-linked dataset cards,
            publishes the seven datasets and saves `eval_suite.json` with immutable uploaded revisions.
            Keep that file for baseline and every subsequent stage.
        """),
            code("""
            from google.colab import files
            try:
                if SMOKE_TEST:
                    print("Smoke test complete. Review the archive before changing SMOKE_TEST to False.")
                else:
                    uploads = publish(accepted, DATASETS, CONFIG, decontamination_report,
                                      checks_report, run_manifest, HF_TOKEN)
                    print(uploads)
            finally:
                archive = archive_outputs(ROOT, "smoke_test" if SMOKE_TEST else "translations")
                print("Saved archive:", archive)
                files.download(str(archive))
        """),
        ],
        selected=args.notebook,
    )
    write_notebook(
        "evaluate_baseline.ipynb",
        [
            markdown("""
            # EXAONE baseline evaluation on the RTX PRO 6000 through Colab
            Connect this notebook to the **RTX PRO 6000 Blackwell 96GB runtime**.
            A normal Colab CPU/T4/A100 session does not satisfy this project's hardware requirement.
            Finish translation, review and upload first; `eval_suite.json` must be in the project Drive root.
            This notebook runs the same `eval/run_eval.py` used by every stage and saves results to Drive.
            No training is performed. Keep the environment lock and protocol unchanged for later evaluations.
        """),
            code("""
            PROJECT_ROOT = "/content/drive/MyDrive/LG-Korea-AIME"
            MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
            MODEL_REVISION = None  # Resolved and recorded as a Hugging Face commit on first run.
            STAGE = "stage0"
            RUN_NAME = "baseline"
            CUDA_VISIBLE_DEVICES = "0"
        """),
            code("""
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
            from google.colab import drive, userdata
            drive.mount("/content/drive")
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        """),
            bootstrap(bundle),
            code("""
            import subprocess
            if sys.version_info[:2] != (3, 12):
                raise RuntimeError("Use a Python 3.12 GPU runtime for the pinned evaluation environment")
            lock = CODE_ROOT / "requirements-eval.lock"
            if lock.exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(lock)])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", str(CODE_ROOT) + "[eval]"])
            # GPU packages run in a fresh subprocess, so notebook imports cannot retain old versions.
        """),
            markdown("""
            ## Run the fixed evaluation protocol
            `configs/eval.yaml` contains all generation settings: temperature 1.0, top-p 0.7,
            20,480 completion tokens, n=32 for AIME/AMC and n=4 for MATH-500.
            The first baseline freezes the configuration, shared evaluation code and uploaded dataset suite.
            A complete 630-problem suite produces 6,160 responses. Translation exclusions are recorded explicitly.
            If interrupted, rerun with the same model, settings and run name to resume saved generations.
        """),
            code("""
            command = [sys.executable, str(CODE_ROOT / "eval/run_eval.py"),
                       "--model", MODEL, "--stage", STAGE, "--run_name", RUN_NAME,
                       "--config", str(CODE_ROOT / "configs/eval.yaml"),
                       "--output_root", PROJECT_ROOT]
            if MODEL_REVISION:
                command += ["--revision", MODEL_REVISION]
            subprocess.check_call(command, cwd=CODE_ROOT)
        """),
            code("""
            import json
            from google.colab import files
            from zipfile import ZipFile, ZIP_DEFLATED
            root = Path(PROJECT_ROOT)
            result = root / "results" / STAGE / f"{RUN_NAME}.json"
            print(json.dumps(json.loads(result.read_text())["metrics"], ensure_ascii=False, indent=2))
            archive = root / "archives" / f"{STAGE}_{RUN_NAME}.zip"
            archive.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive, "w", ZIP_DEFLATED) as z:
                for path in (root / "results" / STAGE).glob(f"{RUN_NAME}*.json*"):
                    z.write(path, path.relative_to(root))
                z.write(root / "results/eval_protocol.json", "results/eval_protocol.json")
                z.write(root / "results/eval_runtime.json", "results/eval_runtime.json")
            files.download(str(archive))
        """),
        ],
        selected=args.notebook,
    )


if __name__ == "__main__":
    main()
