"""Audit real prepared English data without making translation or generation requests."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from common.io import read_jsonl, write_json
from pipeline.datasets import load_registry
from pipeline.decontamination import prepare_decontamination
from pipeline.translation import TranslationFailure, select_smoke, split_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    root = Path(args.root)
    registry = load_registry()
    config = yaml.safe_load(Path("configs/translation.yaml").read_text())
    source_data = {name: read_jsonl(root / "sources" / f"{name}.jsonl") for name in registry}
    data, report = prepare_decontamination(
        source_data,
        registry,
        root,
        ngram_size=config["DECONTAM_NGRAM_SIZE"],
        coverage_threshold=config["DECONTAM_COVERAGE_THRESHOLD"],
        near_miss_min=config["DECONTAM_NEAR_MISS_MIN"],
    )
    smoke = select_smoke(data, config)
    oversized = []
    for row in data["s1k_1.1"]:
        for column in registry["s1k_1.1"]["translate"]:
            try:
                split_chunks(row["original"][column], config["CHUNK_CHARS"])
            except TranslationFailure as exc:
                oversized.append({"id": row["id"], "field": column, "reason": str(exc)})
    summary = {
        "decontamination_settings": {
            k: report[k]
            for k in [
                "ngram_size",
                "coverage_threshold",
                "near_miss_min",
                "normalization_version",
                "cache_directory",
            ]
        },
        "source_revisions": {name: spec["revision"] for name, spec in registry.items()},
        "decontamination": {
            name: {k: v for k, v in counts.items() if k != "removed"}
            for name, counts in report["datasets"].items()
        },
        "evaluation_rows": {
            name: len(data[name]) for name, spec in registry.items() if spec["role"] == "eval"
        },
        "smoke_selected_ids": {name: [row["id"] for row in rows] for name, rows in smoke.items()},
        "oversized_paragraphs": oversized,
        "translation_executed": False,
        "baseline_executed": False,
    }
    write_json("docs/data_preparation_summary.json", summary)
    print("Saved docs/data_preparation_summary.json")


if __name__ == "__main__":
    main()
