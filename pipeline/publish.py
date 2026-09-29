"""Assemble checked translations, retain source licenses, publish and archive outputs."""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from common.io import write_json


def assemble_row(row, record, name):
    output = deepcopy(row["original"])
    additions = dict(record["translations"])
    if "id" not in output:
        additions["id"] = row["id"]
    if name == "dapo_math_17k":
        additions["problem"] = row["problem"]
    collision = set(additions) & set(output)
    if collision:
        raise ValueError(f"Refusing to overwrite original columns: {collision}")
    output.update(additions)
    return output


def dataset_card(name, spec, config, decontamination, checks):
    metadata = {
        "language": ["en", "ko"],
        "source_datasets": [spec["repo"]],
        "tags": ["machine-translated", "mathematics"],
    }
    if spec["license"]:
        metadata["license"] = spec["license"]
    license_text = (
        spec["license"] or "Not specified by the source dataset; no new license is asserted."
    )
    text = (
        "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n"
        f"# {spec['upload_name']}\n\n"
        f"Source: [{spec['repo']}](https://huggingface.co/datasets/{spec['repo']})\n\n"
        f"Source revision: `{spec['revision']}`; config: `{spec['config']}`; split: `{spec['split']}`.\n\n"
        f"Source license: {license_text}\n\n"
        f"The `ko_` columns are machine-translated with `{config['TRANSLATION_MODEL']}`. "
        "All original columns are preserved unchanged. Additional IDs are stable source IDs "
        "or source split row indices recorded in the preparation manifest. "
        "DAPO's added `problem` column contains the English problem without its instruction wrapper.\n\n"
        f"Quality checks: `{json.dumps(checks['datasets'][name], ensure_ascii=False)}`. "
        "Rows still failing after one automatic retry are excluded.\n\n"
    )
    if spec["role"] == "train":
        summary = {k: v for k, v in decontamination["datasets"][name].items() if k != "removed"}
        text += (
            "Training rows were decontaminated against AIME 2024/2025/2026, AMC 2023 and "
            "MATH-500 using normalized English 8-grams before translation, including "
            f"exact-substring checking of short evaluation problems. Summary: `{json.dumps(summary)}`.\n\n"
        )
    else:
        text += "Evaluation rows are never decontaminated. See quality-check counts for translation exclusions.\n\n"
    text += "The original dataset card is retained in `SOURCE_README.md`.\n"
    return text


def publish(accepted, registry, config, decontamination, checks, manifest, token):
    from datasets import Dataset
    from huggingface_hub import HfApi

    if config["SMOKE_TEST"]:
        raise ValueError("Smoke tests must never upload")
    if not manifest.get("checks_completed") or checks["mode"] != "translations":
        raise ValueError("Full translation checks must complete before upload")
    if set(accepted) != set(registry) or any(not accepted[n] for n in registry):
        raise ValueError("All seven datasets must have checked, accepted rows")
    missing = [n for n, spec in registry.items() if not spec["license"]]
    if missing and config["UNSPECIFIED_LICENSE_POLICY"] != "retain-unspecified":
        raise ValueError(f"Source license unspecified; resolve policy before upload: {missing}")
    root = Path(config["PROJECT_ROOT"])
    api = HfApi(token=token)
    api.whoami()  # Validate authentication before publishing anything.
    published, eval_suite = {}, {}
    for name, spec in registry.items():
        rows = [assemble_row(row, record, name) for row, record in accepted[name]]
        repo = f"{config['HF_USERNAME']}/{spec['upload_name']}"
        api.create_repo(repo, repo_type="dataset", private=config["HF_PRIVATE"], exist_ok=True)
        api.update_repo_settings(repo, repo_type="dataset", private=config["HF_PRIVATE"])
        ds = Dataset.from_list(rows)
        ds.push_to_hub(repo, split=spec["split"], private=config["HF_PRIVATE"], token=token)
        card = dataset_card(name, spec, config, decontamination, checks)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
        source_card = root / "sources" / f"{name}_source_card.md"
        api.upload_file(
            path_or_fileobj=str(source_card),
            path_in_repo="SOURCE_README.md",
            repo_id=repo,
            repo_type="dataset",
        )
        revision = api.dataset_info(repo).sha
        published[name] = {
            "repo": repo,
            "revision": revision,
            "rows": len(rows),
            "split": spec["split"],
        }
        write_json(root / "translations" / "uploads.json", published)
        if spec["role"] == "eval":
            eval_suite[name] = {
                **published[name],
                "ids": [r["id"] for r in rows],
                "source_rows": spec["expected_rows"],
                "excluded_rows": checks["datasets"][name]["flagged_count"],
            }
    write_json(
        root / "eval_suite.json",
        {"translation_signature": manifest["signature"], "datasets": eval_suite},
    )
    return published


def archive_outputs(root, mode):
    root = Path(root)
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = archives / f"{mode}_{stamp}.zip"
    # No credentials or package caches live in these output directories.
    folders = [root / "decontamination", root / mode, root / "checks"]
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for folder in folders:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
        for path in [root / "sources" / "manifest.json", root / "eval_suite.json"]:
            if path.exists():
                archive.write(path, path.relative_to(root))
    return destination
