"""Build the self-contained, CPU-only training-log collection notebook."""

from build_notebooks import ROOT, code, markdown, payload, write_notebook

NAME = 'collect_qwen_training_logs.ipynb'
FILES = ['scripts/collect_qwen_training_logs.py']


def cells():
    encoded = payload([ROOT / name for name in FILES])
    return [
        markdown('''
        # Collect Qwen experiment logs for the final write-up

        **CPU Colab; no GPU, API token, model download, or package installation needed.**
        Run all cells. Mount Drive, collect the existing files, build and verify one ZIP,
        save a copy on Drive, and trigger a browser download. Send that ZIP for analysis.

        Includes all seven trained variants, the distillation extension (epochs 6–7),
        and the original instruct-RL history needed for steps 1–40 before recovery.
        Loss, reward, sequence length, entropy, gradient norm, learning rate and timing
        fields are retained exactly as recorded, including nested metrics. Available
        per-epoch math evaluation tables, run manifests and checkpoint JSON metadata
        are included for interpreting the training curves.

        **No model weights, optimizer states, datasets, raw rollouts, generated answers,
        console logs or credentials.** Sources on Drive are read only. No graphs or
        merged histories are generated yet. Missing files are reported rather than
        silently treated as complete. RunPod/HF transfer results are separate artifacts.
        '''),
        code('''
        from pathlib import Path

        DRIVE_ROOT = Path("/content/drive/MyDrive")
        LOCAL_OUTPUT = Path("/content/qwen-training-log-export")
        MAX_FILE_MB = 100  # Larger individual files are skipped and listed in the report.
        SAVE_COPY_TO_DRIVE = True
        DRIVE_EXPORT_ROOT = DRIVE_ROOT / "LG-AIME-Qwen25-3B-Experiments/writeup-artifacts"
        '''),
        code('''
        from google.colab import drive
        drive.mount("/content/drive")
        '''),
        code(f'''
        import base64, io, sys, zipfile

        CODE_ROOT = Path("/content/qwen-log-collector")
        CODE_ROOT.mkdir(parents=True, exist_ok=True)
        BUNDLE = {encoded!r}
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
            archive.extractall(CODE_ROOT)
        sys.path.insert(0, str(CODE_ROOT / "scripts"))
        from collect_qwen_training_logs import collect
        '''),
        code('''
        import shutil

        ZIP_PATH, report = collect(DRIVE_ROOT, LOCAL_OUTPUT, max_file_mb=MAX_FILE_MB)
        print("\\nCollection:", report["status"])
        print("Files:", len(report["files"]))
        print("ZIP:", ZIP_PATH, f"({ZIP_PATH.stat().st_size / 1024**2:.1f} MiB)")
        for category in ["missing", "skipped", "errors"]:
            if report[category]:
                print("\\n" + category.upper() + ":")
                for item in report[category]:
                    print(item)
        for run in report["runs"]:
            if run.get("malformed_lines"):
                print("WARNING:", run["key"], "has malformed step-log lines; raw bytes are preserved.")
        if SAVE_COPY_TO_DRIVE:
            DRIVE_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
            drive_copy = DRIVE_EXPORT_ROOT / ZIP_PATH.name
            shutil.copy2(ZIP_PATH, drive_copy)
            print("\\nDrive backup:", drive_copy)
        '''),
        markdown('''
        ## Download and send the ZIP

        The ZIP contains `README.md`, `collection_manifest.json` (file hashes and missing
        paths), `run_inventory.csv` (available metrics and step ranges), and the original
        files under `drive/`. A partial collection can still be analyzed; missing paths
        help identify what needs to be located later.

        The next cell starts the browser download. If it is blocked, rerun only that
        cell or download the ZIP from Colab's Files sidebar or the printed Drive backup.
        '''),
        code('''
        from google.colab import files
        files.download(str(ZIP_PATH))
        '''),
    ]


if __name__ == '__main__':
    write_notebook(NAME, cells())
