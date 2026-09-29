"""Download and decontaminate English sources without any translation API calls."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml

from pipeline.datasets import download_sources, load_registry
from pipeline.decontamination import prepare_decontamination, print_decontamination_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--registry", default="configs/datasets.yaml")
    parser.add_argument("--config", default="configs/translation.yaml")
    args = parser.parse_args()
    registry = load_registry(args.registry)
    data, _ = download_sources(registry, args.root, os.getenv("HF_TOKEN"))
    config = yaml.safe_load(Path(args.config).read_text())
    _, report = prepare_decontamination(
        data,
        registry,
        args.root,
        ngram_size=config["DECONTAM_NGRAM_SIZE"],
        coverage_threshold=config["DECONTAM_COVERAGE_THRESHOLD"],
        near_miss_min=config["DECONTAM_NEAR_MISS_MIN"],
    )
    print_decontamination_report(report)


if __name__ == "__main__":
    main()
