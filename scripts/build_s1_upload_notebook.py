"""Build the self-contained Colab notebook for publishing existing Pro outputs."""

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook


def cells():
    boot = bootstrap(payload([ROOT / path for path in [
        "common/__init__.py", "common/io.py", "pipeline/__init__.py",
        "pipeline/publish_s1_regeneration.py", "configs/publish_s1_regeneration.yaml",
    ]]))
    boot.source = boot.source.replace("/content/lg-korean-aime", "/content/lg-s1-upload")
    return [
        markdown("""
        # Save the regenerated s1 dataset to Hugging Face

        Run on **Colab CPU** with a **write-capable `HF_TOKEN`** in Colab Secrets.
        This self-contained notebook updates `Seungjun/dp_removed_s1K-1.1` (`default`, `train`)
        with `deepseek-v4-pro_reasoning` and `deepseek-v4-pro_answer`.
        It reads your existing Drive export; it makes no DeepSeek requests.

        All 996 rows and original fields are retained. The six unsuccessful rows keep null
        Pro fields. No correctness, heading-format, or training-length filter is applied.
        Both thinking and answer are saved separately for later combined training.
        Existing `deepseek_grade` fields still describe the old answers, not the new Pro answers.

        Run the cells in order: setup → paths → validate → upload → verify.
        **The upload cell writes to the existing HF repository** with its existing visibility.
        The dataset is published as Parquet, and a ZIP backs up the exact input JSONL files
        plus a provenance manifest. A receipt with commit IDs is saved on Drive.
        The original pinned revision remains the source for existing training notebooks;
        this notebook does not change their dataset revision or column selections.
        """),
        code("""
        # @title Install dependencies, mount Drive, and load your write token
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "datasets==5.0.1", "huggingface-hub==0.36.2", "PyYAML==6.0.3"])
        from google.colab import drive, userdata
        drive.mount("/content/drive")
        HF_TOKEN = userdata.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Enable notebook access to a write-capable HF_TOKEN in Colab Secrets")
        """),
        boot,
        code("""
        # @title Settings — defaults point to your existing full-run export
        import yaml
        CONFIG = yaml.safe_load((CODE_ROOT / "configs/publish_s1_regeneration.yaml").read_text())
        # If you move the files or finish the remaining six rows, change these paths here.
        # CONFIG["files"]["dataset"] = "/content/drive/MyDrive/.../dataset.jsonl"
        # CONFIG["files"]["status"] = "/content/drive/MyDrive/.../status.jsonl"
        print(yaml.safe_dump(CONFIG, sort_keys=False))
        """),
        code("""
        # @title Read files, verify row alignment, and prepare the two added columns
        import hashlib, json, shutil
        from datetime import datetime, timezone
        from pathlib import Path
        from zipfile import ZipFile, ZIP_DEFLATED
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_download
        from common.io import digest, read_jsonl, write_json
        from pipeline.publish_s1_regeneration import merge_export

        api = HfApi(token=HF_TOKEN)
        spec, target = CONFIG["source"], CONFIG["target"]
        # Capture immutable inputs before validation and upload.
        snapshot = Path("/content/s1-upload-snapshot")
        snapshot.mkdir(parents=True, exist_ok=True)
        for key, filename in [("dataset", "dataset.partial.jsonl"), ("status", "status.jsonl")]:
            path = Path(CONFIG["files"][key])
            if not path.is_file():
                raise FileNotFoundError(f"Update CONFIG['files']['{key}']; missing: {path}")
            shutil.copyfile(path, snapshot / filename)
        exported = read_jsonl(snapshot / "dataset.partial.jsonl")
        statuses = read_jsonl(snapshot / "status.jsonl")
        source = load_dataset(spec["repo"], name=spec["config"], split=spec["split"],
                              revision=spec["revision"], token=HF_TOKEN)
        before = api.dataset_info(target["repo"], revision=target["revision"]).sha
        current = load_dataset(target["repo"], name=target["config"], split=target["split"],
                               revision=before, token=HF_TOKEN)
        merged, summary = merge_export(source, current, exported, statuses, CONFIG)
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                    "source": spec, "target": target, "previous_revision": before,
                    "model": CONFIG["model"], "summary": summary,
                    "original_paths": CONFIG["files"],
                    "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in snapshot.iterdir() if p.suffix == ".jsonl"}}
        write_json(snapshot / "manifest.json", manifest)
        archive_path = Path("/content/s1-generation-backup.zip")
        with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
            for filename in ["dataset.partial.jsonl", "status.jsonl", "manifest.json"]:
                archive.write(snapshot / filename, arcname=filename)
        archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        backup_remote = f"generation_backups/{CONFIG['model']}/{archive_sha}.zip"
        receipt_dir = Path(CONFIG["files"]["receipt_directory"])
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"{archive_sha}.json"
        receipt = {**manifest, "backup_path": backup_remote, "backup_sha256": archive_sha,
                   "verified": False}
        write_json(receipt_path, receipt)
        print(json.dumps(summary, indent=2))
        print("Target:", target["repo"], target["config"], target["split"])
        print("Existing revision:", before)
        print("Backup:", backup_remote)
        """),
        markdown("""
        ## Upload the updated dataset and backup

        Run this cell to publish the prepared result. Identical reruns are supported;
        validation also permits filling missing Pro results later, but refuses to erase
        or replace existing non-null Pro text with different text.
        If a connection fails, rerun validation and upload. Use one uploader at a time.
        The receipt records each completed upload before verification.
        """),
        code("""
        # @title Push the two added columns and back up both input files
        if api.dataset_info(target["repo"], revision=target["revision"]).sha != before:
            raise RuntimeError("HF changed since validation. Rerun the validation cell first.")
        commit = merged.push_to_hub(
            target["repo"], config_name=target["config"], split=target["split"],
            revision=target["revision"], token=HF_TOKEN,
            commit_message="Add DeepSeek Pro reasoning and answers; preserve all source rows")
        receipt["dataset_revision"] = commit.oid
        write_json(receipt_path, receipt)
        backup_commit = api.upload_file(
            path_or_fileobj=str(archive_path), path_in_repo=backup_remote,
            repo_id=target["repo"], repo_type="dataset", revision=target["revision"],
            commit_message="Back up regeneration JSONL, row status, and provenance")
        receipt["revision"] = backup_commit.oid
        write_json(receipt_path, receipt)
        print("Uploaded:", f"https://huggingface.co/datasets/{target['repo']}/tree/{backup_commit.oid}")
        print("Receipt saved:", receipt_path)
        """),
        code("""
        # @title Reload the uploaded revision and verify all rows plus the backup
        receipt = json.loads(receipt_path.read_text())
        revision = receipt["revision"]
        uploaded = load_dataset(target["repo"], name=target["config"], split=target["split"],
                                revision=revision, token=HF_TOKEN)
        if digest(list(uploaded)) != summary["dataset_digest"]:
            raise ValueError("Uploaded dataset does not exactly match the prepared dataset")
        downloaded = hf_hub_download(repo_id=target["repo"], filename=backup_remote,
                                     repo_type="dataset", revision=revision, token=HF_TOKEN)
        if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != archive_sha:
            raise ValueError("Uploaded backup checksum mismatch")
        receipt["verified"] = True
        write_json(receipt_path, receipt)
        print("Verified:", len(uploaded), "rows; original fields, both Pro columns, and backup intact.")
        print("Pinned revision for later use:", revision)
        print("Drive receipt:", receipt_path)
        print(f"https://huggingface.co/datasets/{target['repo']}")
        """),
        markdown("""
        The six null pairs are intentionally retained for backup, not silently treated as training
        examples. Publishing does not judge correctness or shorten either output column.
        No API keys are included in the backup or receipt.

        References: [Dataset.push_to_hub](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.push_to_hub),
        [Hub file uploads](https://huggingface.co/docs/huggingface_hub/guides/upload).
        """),
    ]


if __name__ == "__main__":
    write_notebook("upload_s1_deepseek_to_huggingface.ipynb", cells())
