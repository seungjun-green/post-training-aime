"""Pairwise evaluation 8-gram coverage; only training rows can be removed."""

import json
import re
import textwrap
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from common.io import digest, read_jsonl, write_json, write_jsonl

NGRAM_SIZE = 8
NORMALIZATION_VERSION = 2
COVERAGE_THRESHOLD = 0.7
NEAR_MISS_MIN = 0.5


def normalize(text):
    text = text.lower().replace("$", "")
    text = re.sub(r"\\(?:left|right|quad)(?![a-z])|\\[()\[\],;!]", "", text)
    text = re.sub(r"([^\w\s])", r" \1 ", text)
    return " ".join(token for token in text.split() if any(c.isalnum() for c in token))


def grams(tokens, ngram_size=NGRAM_SIZE):
    return {tuple(tokens[i : i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)}


def validate_settings(ngram_size, coverage_threshold, near_miss_min):
    if isinstance(ngram_size, bool) or not isinstance(ngram_size, int) or ngram_size < 1:
        raise ValueError("ngram_size must be a positive integer")
    if not 0 <= near_miss_min < coverage_threshold <= 1:
        raise ValueError("Require 0 <= near_miss_min < coverage_threshold <= 1")


def decontaminate(
    data,
    registry,
    *,
    ngram_size=NGRAM_SIZE,
    coverage_threshold=COVERAGE_THRESHOLD,
    near_miss_min=NEAR_MISS_MIN,
):
    validate_settings(ngram_size, coverage_threshold, near_miss_min)
    index = defaultdict(set)
    short, references, gram_counts = [], {}, {}
    eval_sets = [name for name in data if registry[name]["role"] == "eval"]
    for name in eval_sets:
        for row in data[name]:
            key = (name, str(row["id"]))
            references[key] = {
                "eval_set": name,
                "eval_id": row["id"],
                "eval_problem": row["problem"],
            }
            normalized = normalize(row["problem"])
            if not normalized:
                raise ValueError(f"Empty normalized evaluation problem: {key}")
            tokens = normalized.split()
            if len(tokens) < ngram_size:
                short.append((normalized, key))
            else:
                eval_grams = grams(tokens, ngram_size)
                gram_counts[key] = len(eval_grams)
                for gram in eval_grams:
                    index[gram].add(key)
    clean, reports, near_misses = dict(data), {}, []
    for name, rows in data.items():
        if registry[name]["role"] != "train":
            continue
        kept, removed, near_count = [], [], 0
        removal_counts = Counter()
        for row in rows:
            normalized = normalize(row["problem"])
            # Every training gram is distinct. Each counter is therefore precisely
            # |training_grams intersect eval_grams| for one candidate eval problem.
            intersections = Counter()
            for gram in grams(normalized.split(), ngram_size):
                intersections.update(index.get(gram, ()))
            candidates = [
                {
                    **references[key],
                    "method": "ngram_coverage",
                    "coverage": count / gram_counts[key],
                    "matched_ngram_count": count,
                    "eval_ngram_count": gram_counts[key],
                }
                for key, count in sorted(intersections.items())
            ]
            for substring, key in short:
                if substring in normalized:
                    candidates.append(
                        {
                            **references[key],
                            "method": "short_exact_substring",
                            "coverage": 1.0,
                            "matched_ngram_count": 0,
                            "eval_ngram_count": 0,
                            "overlap": substring,
                        }
                    )
            candidates.sort(key=lambda m: (-m["coverage"], m["eval_set"], str(m["eval_id"])))
            matches = [m for m in candidates if m["coverage"] >= coverage_threshold]
            if matches:
                removed.append(
                    {
                        "id": row["id"],
                        "problem": row["problem"],
                        "best_coverage": matches[0]["coverage"],
                        "matches": matches,
                    }
                )
                removal_counts.update({m["eval_set"] for m in matches})
            else:
                kept.append(row)
                if candidates and near_miss_min <= candidates[0]["coverage"] < coverage_threshold:
                    near_count += 1
                    best = candidates[0]["coverage"]
                    near_misses.append(
                        {
                            "training_set": name,
                            "id": row["id"],
                            "problem": row["problem"],
                            "best_coverage": best,
                            "matches": [m for m in candidates if m["coverage"] == best],
                        }
                    )
        clean[name] = kept
        reports[name] = {
            "original_count": len(rows),
            "removed_count": len(removed),
            "remaining_count": len(kept),
            "removed": removed,
            "near_miss_count": near_count,
            "removal_counts_by_eval_set": {n: removal_counts[n] for n in eval_sets},
        }
    return clean, reports, near_misses


def prepare_decontamination(
    data,
    registry,
    root,
    *,
    ngram_size=NGRAM_SIZE,
    coverage_threshold=COVERAGE_THRESHOLD,
    near_miss_min=NEAR_MISS_MIN,
):
    validate_settings(ngram_size, coverage_threshold, near_miss_min)
    settings = {
        "normalization_version": NORMALIZATION_VERSION,
        "ngram_size": ngram_size,
        "coverage_threshold": coverage_threshold,
        "near_miss_min": near_miss_min,
    }
    signature = digest({"registry": registry, "data": data, **settings})
    # V1 is never overwritten. Later changes also preserve the first V2 run.
    folder = Path(root) / "decontamination_v2"
    base_report = folder / "decontamination_report.json"
    if base_report.exists() and json.loads(base_report.read_text())["input_digest"] != signature:
        folder = folder / signature
    report_path = folder / "decontamination_report.json"
    train_names = [name for name in registry if registry[name]["role"] == "train"]
    paths = {name: folder / f"{name}_decontaminated.jsonl" for name in train_names}
    if report_path.exists() and all(p.exists() for p in paths.values()):
        report = json.loads(report_path.read_text())
        if report["input_digest"] != signature:
            raise ValueError("Decontamination cache signature does not match its directory")
        clean = dict(data)
        for name, path in paths.items():
            clean[name] = read_jsonl(path)
            if digest(clean[name]) != report["output_digests"][name]:
                raise ValueError(f"Decontamination cache checksum failed: {name}")
        return clean, report
    clean, reports, near_misses = decontaminate(
        data,
        registry,
        ngram_size=ngram_size,
        coverage_threshold=coverage_threshold,
        near_miss_min=near_miss_min,
    )
    report = {
        **settings,
        "input_digest": signature,
        "cache_directory": str(folder.relative_to(root)),
        "datasets": reports,
        "near_misses": near_misses,
        "output_digests": {name: digest(clean[name]) for name in train_names},
    }
    for name, path in paths.items():
        write_jsonl(path, clean[name])
    write_json(report_path, report)
    return clean, report


def print_pair(training_problem, match):
    print(
        f"  Eval: {match['eval_set']}/{match['eval_id']} | coverage={match['coverage']:.4f} "
        f"| method={match['method']}"
    )
    print(f"  {'TRAINING PROBLEM':<70} | EVALUATION PROBLEM")
    left = textwrap.wrap(training_problem, width=70)
    right = textwrap.wrap(match["eval_problem"], width=70)
    for a, b in zip_longest(left, right, fillvalue=""):
        print(f"  {a:<70} | {b}")


def print_decontamination_report(report):
    print(
        f"Decontamination: {report['ngram_size']}-grams, coverage >= "
        f"{report['coverage_threshold']}, normalization v{report['normalization_version']}"
    )
    print(f"Outputs: {report['cache_directory']}/")
    for name, dataset in report["datasets"].items():
        print(
            f"\n{name}: "
            + str(
                {
                    k: dataset[k]
                    for k in [
                        "original_count",
                        "removed_count",
                        "remaining_count",
                        "near_miss_count",
                    ]
                }
            )
        )
        print("Removed training rows matching each eval set (sets can overlap):")
        for eval_set, count in dataset["removal_counts_by_eval_set"].items():
            print(f"  {eval_set:<15} {count:>5}")
    s1 = report["datasets"].get("s1k_1.1", {})
    for eval_set in ["aime_2025", "aime_2026"]:
        pairs = [
            (row, match)
            for row in s1.get("removed", [])
            for match in row["matches"]
            if match["eval_set"] == eval_set
        ]
        count = len({str(row["id"]) for row, _ in pairs})
        print(f"\ns1K-1.1 sanity check: {eval_set} removed {count} rows (expected 0).")
        if pairs:
            print("WARNING: s1K-1.1 predates this contest. Inspect these overlap decisions:")
            for row, match in pairs:
                print(f"  Training ID: {row['id']}")
                print_pair(row["problem"], match)
    borderline = sorted(
        [(name, row) for name, ds in report["datasets"].items() for row in ds["removed"]],
        key=lambda item: (item[1]["best_coverage"], item[0], str(item[1]["id"])),
    )[:10]
    print("\n10 removed rows with the lowest best-pair coverage:")
    for name, row in borderline:
        print(f"\n{name}/{row['id']} (best coverage={row['best_coverage']:.4f})")
        print_pair(row["problem"], row["matches"][0])
    print(
        f"\nNear misses kept for manual review: {len(report['near_misses'])} "
        "(see near_misses in decontamination_report.json)."
    )
