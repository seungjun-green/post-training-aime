"""Reconcile changed decontamination cohorts without changing the translator itself."""

import json
from pathlib import Path

from common.io import digest, latest_by_id, write_json, write_jsonl
from pipeline.translation import run_signature, select_smoke


def refresh_translation_cache(data, clean, registry, config, report):
    """Reuse verified rows by ID; archive old selections and require fresh smoke checks.

    Only a proven decontamination-cohort change is eligible. Different translation
    settings, source rows, prompt or translator code still fail the existing gate.
    No translation request or quality check is performed here.
    """
    root = Path(config["PROJECT_ROOT"])
    active_mode = "smoke_test" if config["SMOKE_TEST"] else "translations"
    signature = run_signature(config, registry, clean)
    selections = {"smoke_test": select_smoke(clean, config), "translations": clean}
    report_paths = [root / "decontamination" / "decontamination_report.json"]
    report_paths += sorted((root / "decontamination_v2").rglob("decontamination_report.json"))
    valid_signatures = {signature}
    for path in report_paths:
        if not path.exists():
            continue
        previous = json.loads(path.read_text())
        if set(previous.get("datasets", {})) != {
            n for n in registry if registry[n]["role"] == "train"
        }:
            continue
        prior = dict(data)
        for name, dataset in previous["datasets"].items():
            removed_ids = {str(r["id"]) for r in dataset["removed"]}
            prior[name] = [r for r in data[name] if str(r["id"]) not in removed_ids]
        # The source payload, including originals, must match the saved cohort.
        if any(
            digest(prior[n]) != value for n, value in previous.get("output_digests", {}).items()
        ):
            continue
        valid_signatures.add(run_signature(config, registry, prior))

    manifests, available = {}, {name: {} for name in data}
    for mode in selections:
        folder = root / mode
        path = folder / "manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        if manifest["signature"] not in valid_signatures:
            raise ValueError(
                "Translation settings or sources changed beyond decontamination; "
                "use a new project root"
            )
        manifests[mode] = manifest
        # Include previous cohorts so tightening and then relaxing the threshold
        # never loses translations for IDs temporarily excluded from a selection.
        journal_folders = [folder] + sorted((folder / "decontamination_history").glob("*"))
        for journal_folder in journal_folders:
            old_manifest = journal_folder / "manifest.json"
            if not old_manifest.exists():
                continue
            if json.loads(old_manifest.read_text())["signature"] not in valid_signatures:
                continue
            for name, source_rows in data.items():
                sources = {str(r["id"]): r for r in source_rows}
                for identifier, record in latest_by_id(journal_folder / f"{name}.jsonl").items():
                    source = sources.get(identifier)
                    if source is None or record.get("source_digest") != digest(source):
                        raise ValueError(f"Resume source mismatch: {name}/{identifier}")
                    if record.get("translation_model") != config["TRANSLATION_MODEL"]:
                        raise ValueError(f"Resume translation model mismatch: {name}/{identifier}")
                    old = available[name].get(identifier)
                    if old is None or record.get("attempt", 1) > old.get("attempt", 1):
                        available[name][identifier] = record

    for mode, rows in selections.items():
        folder = root / mode
        previous = manifests.get(mode)
        ids = {n: [r["id"] for r in rs] for n, rs in rows.items()}
        migrating = previous is not None and previous["signature"] != signature
        seed_from_other_mode = previous is None and mode == active_mode and any(available.values())
        if migrating or seed_from_other_mode:
            if migrating:
                history = folder / "decontamination_history" / previous["signature"]
                history.mkdir(parents=True, exist_ok=True)
                # Never overwrite the first snapshot if a migration is interrupted.
                for name in registry:
                    path = folder / f"{name}.jsonl"
                    backup = history / path.name
                    if path.exists() and not backup.exists():
                        backup.write_bytes(path.read_bytes())
                if not (history / "manifest.json").exists():
                    write_json(history / "manifest.json", previous)
            for name, selected in rows.items():
                reusable = [
                    available[name][str(r["id"])]
                    for r in selected
                    if str(r["id"]) in available[name]
                ]
                write_jsonl(folder / f"{name}.jsonl", reusable)
            updated = {
                "signature": signature,
                "mode": mode,
                "config": config,
                "selected_ids": ids,
                "checks_completed": False,
                "previous_decontamination_signature": previous["signature"] if previous else None,
            }
            write_json(folder / "manifest.json", updated)
            print(
                f"{mode}: updated decontamination selection; compatible translations reused by ID. "
                "Rerun smoke translation/checks before a full run."
            )
        # The unchanged archive step already includes the current mode folder.
        # Include the active V2 report there, alongside the historical V1 report.
        if mode == active_mode:
            write_json(folder / "decontamination_report.json", report)
