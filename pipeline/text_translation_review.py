"""Two-tier reports and a human review queue; soft flags never exclude rows."""

import html
from collections import Counter
from pathlib import Path

from common.io import digest, latest_by_id, write_json, write_jsonl
from pipeline.datasets import source_fields
from pipeline.text_translation_checks import hard_checks, soft_checks

MANUAL_EVAL_SETS = {"aime_2024", "aime_2025", "aime_2026", "amc23"}


def check_row(row, record, spec, config, name):
    hard, soft = list(record.get("errors", [])), []
    if record.get("source_digest") != digest(row):
        hard.append("source_digest_mismatch")
    for field, original in source_fields(row, spec).items():
        target = record.get("translations", {}).get("ko_" + field)
        if target is None:
            # The actual chunk error is already recorded; do not add misleading
            # Hangul/length/box-count failures for a field deliberately discarded.
            if not any(e.startswith(field + ":") for e in hard):
                hard.append(f"{field}: missing_translation")
            continue
        hard.extend(f"{field}: {s}" for s in hard_checks(original, target))
        soft.extend(f"{field}: {s}" for s in soft_checks(original, target, config))
        chunks = record.get("chunks", {}).get(field, [])
        if not chunks or any(c.get("stop_reason") != "end_turn" for c in chunks):
            hard.append(f"{field}: incomplete_chunks")
    return list(dict.fromkeys(hard)), list(dict.fromkeys(soft))


def check_all(rows, registry, folder, manifest, translator):
    config = translator.config
    checks_root = Path(config["PROJECT_ROOT"]) / "checks"
    report = {"mode": manifest["mode"], "signature": manifest["signature"], "datasets": {}}
    accepted, queue = {}, []
    for name, examples in rows.items():
        records = latest_by_id(folder / f"{name}.jsonl")
        if any(str(row["id"]) not in records for row in examples):
            raise ValueError(f"Translation is incomplete: {name}")
        flagged, warnings, accepted[name] = [], [], []
        hard_counts, soft_counts = Counter(), Counter()
        for row in examples:
            record = records[str(row["id"])]
            hard, soft = check_row(row, record, registry[name], config, name)
            if hard:
                flagged.append({"id": row["id"], "reasons": hard, "source": row, "record": record})
                hard_counts.update(hard)
            else:
                accepted[name].append((row, record))
            if soft:
                warnings.append({"id": row["id"], "warnings": soft})
                soft_counts.update(soft)
            if config["SMOKE_TEST"] or hard or soft or name in MANUAL_EVAL_SETS:
                queue.append(
                    {
                        "dataset": name,
                        "id": row["id"],
                        "hard_failures": hard,
                        "soft_warnings": soft,
                        "accepted": not hard,
                        "manual_eval_review": name in MANUAL_EVAL_SETS,
                        "english": source_fields(row, registry[name]),
                        "korean": record["translations"],
                    }
                )
        write_jsonl(checks_root / f"{name}_flagged.jsonl", flagged)
        write_jsonl(checks_root / f"{name}_review.jsonl", warnings)
        report["datasets"][name] = {
            "total": len(examples),
            "accepted": len(accepted[name]),
            "flagged_count": len(flagged),
            "review_count": len(warnings),
            "retried_count": sum(records[str(r["id"])].get("attempt", 1) > 1 for r in examples),
            "reasons": dict(hard_counts),
            "soft_reasons": dict(soft_counts),
        }
    # Put the queue inside the mode folder so both smoke/full archives include it.
    queue.sort(
        key=lambda r: (
            not r["manual_eval_review"],
            not bool(r["hard_failures"]),
            r["dataset"],
            str(r["id"]),
        )
    )
    write_jsonl(folder / "human_review_queue.jsonl", queue)
    sections = []
    for item in queue:
        title = f"{item['dataset']} / {item['id']}"
        reasons = item["hard_failures"] + item["soft_warnings"]
        status = "HARD FAILURE — excluded" if item["hard_failures"] else "Accepted — review only"
        fields = "".join(
            "<h3>"
            + html.escape(field)
            + "</h3><div class='pair'><pre>"
            + html.escape(str(original))
            + "</pre><pre>"
            + html.escape(item["korean"].get("ko_" + field, "[MISSING]"))
            + "</pre></div>"
            for field, original in item["english"].items()
        )
        sections.append(
            "<details><summary>"
            + html.escape(title)
            + " — "
            + status
            + "</summary><p>"
            + html.escape("; ".join(reasons) or "Review mathematical meaning and Korean wording.")
            + "</p>"
            + fields
            + "</details>"
        )
    (folder / "human_review.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Translation human review</title>"
        "<style>body{font:16px system-ui;margin:24px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{padding:12px;cursor:pointer}</style>"
        "<h1>Human review: English / Korean</h1><p>AIME 2024–2026 and AMC 2023 appear first, "
        "including unflagged problems (130 in the full source suite). Soft warnings retain rows. "
        "These literal checks do not verify meaning; no second-model semantic call is made.</p>"
        + "".join(sections),
        encoding="utf-8",
    )
    write_json(checks_root / "checks_report.json", report)
    write_json(folder / "checks_report.json", report)
    manifest.update(checks_completed=True, checks_digest=digest(report))
    write_json(folder / "manifest.json", manifest)
    return accepted, report
