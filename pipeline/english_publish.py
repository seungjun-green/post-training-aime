"""Stage and publish decontaminated English rows with their original schema."""

import hashlib
import json
from pathlib import Path

import yaml

from common.io import digest, write_json


def upload_name(spec):
    return "dp_removed_" + spec["repo"].rsplit("/", 1)[-1]


def validate_clean(data, clean, registry, report):
    if set(data) != set(registry) or set(clean) != set(registry):
        raise ValueError("Expected every configured dataset")
    for name, spec in registry.items():
        if spec["role"] == "eval":
            if clean[name] != data[name]:
                raise ValueError(f"Evaluation rows changed: {name}")
        else:
            removed = {str(r["id"]) for r in report["datasets"][name]["removed"]}
            expected = [row for row in data[name] if str(row["id"]) not in removed]
            if clean[name] != expected or digest(clean[name]) != report["output_digests"][name]:
                raise ValueError(f"Clean rows disagree with decontamination report: {name}")
        if not clean[name]:
            raise ValueError(f"No rows remain in {name}; inspect the report before upload")


def dataset_card(name, spec, report, count):
    filename = f"data/{spec['split']}-00000-of-00001.parquet"
    metadata = {
        "language": ["en"],
        "source_datasets": [spec["repo"]],
        "tags": ["mathematics"],
        "configs": [
            {
                "config_name": spec["config"],
                "default": True,
                "data_files": [{"split": spec["split"], "path": filename}],
            }
        ],
    }
    if spec.get("license"):
        metadata["license"] = spec["license"]
    text = (
        "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n"
        f"# {upload_name(spec)}\n\n"
        f"Source: [{spec['repo']}](https://huggingface.co/datasets/{spec['repo']})\n\n"
        f"Pinned source revision: `{spec['revision']}`. Config: `{spec['config']}`. "
        f"Split: `{spec['split']}`. Retained rows: {count}.\n\n"
        "All retained rows and original columns are unchanged, including English problems, "
        "solutions, reasoning traces, prompt wrappers and source IDs. No translation, "
        "text repair, model generation, or added data columns were applied. "
        "Stable preparation IDs are recorded separately in `provenance.json`.\n\n"
    )
    if spec["role"] == "train":
        summary = {k: v for k, v in report["datasets"][name].items() if k != "removed"}
        text += (
            f"Training/evaluation overlap removal uses normalization v{report['normalization_version']}, "
            f"distinct {report['ngram_size']}-gram coverage per training/evaluation pair, "
            f"and a threshold of {report['coverage_threshold']}. Coverage uses the evaluation "
            "problem as the denominator. Short evaluation problems use exact normalized "
            "substring matching. References: AIME 2024, 2025, 2026; AMC 2023; MATH-500. "
            "Repeated training problems within or across training datasets are not separately removed.\n\n"
            f"Summary: `{json.dumps(summary)}`.\n\n"
        )
    else:
        text += (
            "This evaluation split is an unchanged copy of the source. The naming prefix "
            "identifies the collection; no evaluation rows were removed.\n\n"
        )
    text += (
        "Source license: "
        + (spec.get("license") or "not specified by the source; no new license is asserted.")
        + "\n\nThe original card is retained in `SOURCE_README.md`. "
        "See `decontamination_report.json` for removals, matching evaluation problems, "
        "coverage scores and retained near misses.\n"
    )
    return text


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_english(data, clean, registry, source_manifest, report, root):
    from datasets import Dataset, Features

    root = Path(root)
    validate_clean(data, clean, registry, report)
    # Validate all inputs before staging or uploading any dataset.
    for name, spec in registry.items():
        meta = source_manifest[name]
        if meta["revision"] != spec["revision"] or meta["content_digest"] != digest(data[name]):
            raise ValueError(f"Source manifest mismatch: {name}")
        if not (root / "sources" / f"{name}_source_card.md").is_file():
            raise FileNotFoundError(f"Missing source card for {name}; rerun source download")
    signature = digest(
        {"report": report, "sources": source_manifest, "implementation": Path(__file__).read_text()}
    )
    folder = root / "english_exports" / signature[:20]
    entries = {}
    for name, spec in registry.items():
        destination = folder / upload_name(spec)
        parquet = destination / "data" / f"{spec['split']}-00000-of-00001.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        originals = [row["original"] for row in clean[name]]
        features = Features.from_dict(source_manifest[name]["features"])
        ds = Dataset.from_list(originals, features=features)
        if ds.to_list() != originals:
            raise ValueError(f"Original rows changed during schema conversion: {name}")
        ds.to_parquet(str(parquet))
        (destination / "README.md").write_text(
            dataset_card(name, spec, report, len(ds)), encoding="utf-8"
        )
        (destination / "SOURCE_README.md").write_bytes(
            (root / "sources" / f"{name}_source_card.md").read_bytes()
        )
        write_json(destination / "decontamination_report.json", report)
        write_json(
            destination / "provenance.json",
            {
                "source": source_manifest[name],
                "role": spec["role"],
                "language": "en",
                "original_columns_preserved": True,
                "export_signature": signature,
                "retained_preparation_ids": [r["id"] for r in clean[name]],
                "retained_originals_digest": digest(originals),
            },
        )
        entries[name] = {
            "upload_name": upload_name(spec),
            "config": spec["config"],
            "split": spec["split"],
            "role": spec["role"],
            "rows": len(ds),
            "source_rows": len(data[name]),
            "removed_rows": len(data[name]) - len(ds),
            "problem_column": spec["problem_column"],
            "answer_column": spec["answer_column"],
            "files": {
                str(p.relative_to(destination)): file_hash(p)
                for p in sorted(destination.rglob("*"))
                if p.is_file()
            },
        }
    manifest = {
        "signature": signature,
        "folder": str(folder),
        "language": "en",
        "datasets": entries,
    }
    write_json(folder / "manifest.json", manifest)
    return manifest


def publish_english(manifest, username, private, token, api=None):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import validate_repo_id

    folder = Path(manifest["folder"])
    for name, entry in manifest["datasets"].items():
        validate_repo_id(f"{username}/{entry['upload_name']}")
        for relative, checksum in entry["files"].items():
            if file_hash(folder / entry["upload_name"] / relative) != checksum:
                raise ValueError(f"Staged file changed: {name}/{relative}; rerun preparation")
    api = api or HfApi(token=token)
    api.whoami()
    published = {}
    for name, entry in manifest["datasets"].items():
        repo = f"{username}/{entry['upload_name']}"
        api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
        api.update_repo_settings(repo, repo_type="dataset", private=private)
        # Data + schema/card + report become visible together in one commit per dataset.
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(folder / entry["upload_name"]),
            allow_patterns=list(entry["files"]),
            commit_message="Publish original English split with train/eval overlap removal",
        )
        published[name] = {k: v for k, v in entry.items() if k != "files"}
        published[name].update(
            repo=repo, revision=commit.oid, url=f"https://huggingface.co/datasets/{repo}"
        )
        write_json(folder / "uploads.json", published)
        print(f"{repo}: {entry['rows']} rows | {commit.oid}")
    write_json(
        folder / "english_dataset_suite.json",
        {
            "language": "en",
            "preparation_signature": manifest["signature"],
            "datasets": published,
        },
    )
    return published
