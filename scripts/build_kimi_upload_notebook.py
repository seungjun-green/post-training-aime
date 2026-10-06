"""Build the answer-only s1 column uploader; no generation or training."""

from build_notebooks import ROOT, bootstrap, code, markdown, payload, write_notebook


def cells():
    boot = bootstrap(payload([ROOT / p for p in [
        "common/__init__.py", "common/io.py", "pipeline/__init__.py",
        "pipeline/publish_kimi_answer.py", "configs/publish_kimi_answer.yaml",
    ]]))
    boot.source = boot.source.replace("/content/lg-korean-aime", "/content/lg-kimi-answer-upload")
    return [
        markdown('''
        # Upload the Kimi-style answer column to Hugging Face

        CPU Colab is sufficient. Run the cells in order with a write-capable `HF_TOKEN`
        in Colab Secrets. No DeepSeek calls or training are performed.

        Source: Drive regeneration run **63d2e34487188421**, also used by the answer-only
        EXAONE SFT notebook. Copy `deepseek-v4-pro_answer` verbatim into the new column
        **`kimi-style-reasoning-answer`** in `Seungjun/dp_removed_s1K-1.1` (`default/train`).
        This is the entire answer text, including its reasoning sections and final answer;
        the separate API reasoning channel is not concatenated.

        Keep all rows and existing HF columns, including the older Pro outputs.
        Failed generation rows retain null in the new column. No length or correctness
        filtering is applied. Existing grades still refer to their original answers.
        The upload cell publishes to the existing repository with its existing visibility.
        This version also repairs the stale feature metadata from an earlier upload.
        If that upload failed verification with a column-name CastError, rerun this
        notebook from the top in a fresh CPU runtime; no regeneration is needed.
        '''),
        code('''
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "datasets==5.0.1", "huggingface-hub==0.36.2", "PyYAML==6.0.3"])
        from google.colab import drive, userdata
        drive.mount("/content/drive")
        HF_TOKEN = userdata.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Enable a write-capable HF_TOKEN in Colab Secrets")
        '''),
        boot,
        code('''
        import yaml
        CONFIG = yaml.safe_load((CODE_ROOT / "configs/publish_kimi_answer.yaml").read_text())
        # Plain settings: edit only if you moved the source files.
        DATASET_PATH = CONFIG["files"]["dataset"]
        STATUS_PATH = CONFIG["files"]["status"]
        print("Source:", DATASET_PATH)
        print("Target:", CONFIG["target"]["repo"])
        print("Added column:", CONFIG["new_column"])
        '''),
        markdown('''
        ## Validate and preview

        Compare every original field with the pinned source and current Hub data before
        merging. An identical rerun is allowed; conflicting existing values are rejected.
        All other current Hub columns remain intact. The preview shows up to three answers.
        '''),
        code('''
        import hashlib, json
        from datetime import datetime, timezone
        from pathlib import Path
        from datasets import DatasetDict, get_dataset_split_names, load_dataset
        from huggingface_hub import HfApi, hf_hub_download
        from common.io import digest, read_jsonl, write_json
        from pipeline.publish_kimi_answer import merge_answer, train_parquet_files

        # Snapshot bytes once so the validated inputs cannot change mid-upload.
        snapshot = Path("/content/kimi-answer-upload-inputs")
        snapshot.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for filename, original in [("dataset.jsonl", DATASET_PATH), ("status.jsonl", STATUS_PATH)]:
            content = Path(original).read_bytes()
            (snapshot / filename).write_bytes(content)
            hashes[filename] = hashlib.sha256(content).hexdigest()
        exported = read_jsonl(snapshot / "dataset.jsonl")
        statuses = read_jsonl(snapshot / "status.jsonl")
        api = HfApi(token=HF_TOKEN)
        spec, target = CONFIG["source"], CONFIG["target"]
        before = api.dataset_info(target["repo"], revision=target["revision"]).sha
        source = load_dataset(spec["repo"], name=spec["config"], split=spec["split"],
                              revision=spec["revision"], token=HF_TOKEN)
        # This uploader targets the existing default/train-only repository.
        # Read actual Parquet schemas: older uploads may have left stale README features.
        if target["config"] != "default" or target["split"] != "train":
            raise ValueError("This uploader expects default/train")
        splits = get_dataset_split_names(target["repo"], config_name=target["config"],
                                         revision=before, token=HF_TOKEN)
        if splits != ["train"]:
            raise ValueError(f"Unexpected splits; refusing to replace them: {splits}")
        paths = train_parquet_files(api.list_repo_files(
            target["repo"], repo_type="dataset", revision=before))
        local_parquets = [hf_hub_download(target["repo"], filename=path, repo_type="dataset",
                                         revision=before, token=HF_TOKEN) for path in paths]
        current = load_dataset("parquet", data_files={"train": local_parquets}, split="train")
        merged, summary = merge_answer(source, current, exported, statuses, CONFIG)
        receipt_dir = Path(CONFIG["files"]["receipt_directory"])
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        receipt = {"config": CONFIG, "dataset_path": DATASET_PATH, "status_path": STATUS_PATH,
                   "input_sha256": hashes, "previous_revision": before, "summary": summary,
                   "verified": False}
        write_json(receipt_path, receipt)
        print(json.dumps(summary, indent=2))
        print("Preserved columns:", current.column_names)
        print("Drive receipt:", receipt_path)
        shown = 0
        for i, row in enumerate(merged):
            if row[CONFIG["new_column"]] is not None:
                print("\\nROW", i, "\\nQUESTION:", row["question"])
                print("ANSWER:\\n", row[CONFIG["new_column"]])
                shown += 1
                if shown == 3:
                    break
        '''),
        markdown('''
        ## Upload

        This cell writes the prepared column to Hugging Face. Use one uploader at a time.
        If the repository changes after validation, rerun the validation cell first.
        '''),
        code('''
        if api.dataset_info(target["repo"], revision=target["revision"]).sha != before:
            raise RuntimeError("HF changed since validation; rerun validation before upload")
        # DatasetDict refreshes the feature schema as well as the Parquet files.
        commit = DatasetDict({"train": merged}).push_to_hub(
            target["repo"], config_name=target["config"],
            revision=target["revision"], token=HF_TOKEN,
            commit_message="Add kimi-style-reasoning-answer from regeneration 63d2e34487188421")
        receipt["dataset_revision"] = commit.oid
        write_json(receipt_path, receipt)
        print("Uploaded revision:", commit.oid)
        print("Receipt:", receipt_path)
        '''),
        markdown("## Verify the uploaded revision"),
        code('''
        receipt = json.loads(receipt_path.read_text())
        uploaded = load_dataset(target["repo"], name=target["config"], split=target["split"],
                                revision=receipt["dataset_revision"], token=HF_TOKEN)
        if digest(list(uploaded)) != summary["dataset_digest"]:
            raise ValueError("Uploaded dataset differs from the prepared data")
        receipt["verified"] = True
        write_json(receipt_path, receipt)
        print("Verified all rows and columns:", len(uploaded))
        print("Pinned HF revision:", receipt["dataset_revision"])
        print("Saved receipt:", receipt_path)
        print(f"https://huggingface.co/datasets/{target['repo']}")
        '''),
    ]


if __name__ == "__main__":
    write_notebook("upload_s1_kimi_answer_to_huggingface.ipynb", cells())
