"""One rule-based scorer and metric implementation for every model stage."""

import math
from collections import defaultdict

from common.math_text import is_hangul, last_boxed, without_latex


def score_response(response, gold, timeout_seconds=5):
    from math_verify import LatexExtractionConfig, parse, verify

    extracted = last_boxed(response)
    if extracted is None or not extracted.strip():
        return extracted, False
    reference = str(gold).strip()
    # Gold columns contain final answers, not explanations; support already-delimited gold.
    if reference.startswith("$") and reference.endswith("$"):
        reference = reference.strip("$")
    config = [LatexExtractionConfig()]
    gold_parsed = parse(
        "$" + reference + "$", extraction_config=config, parsing_timeout=timeout_seconds
    )
    prediction = parse(
        "$" + extracted + "$", extraction_config=config, parsing_timeout=timeout_seconds
    )
    return extracted, bool(verify(gold_parsed, prediction, timeout_seconds=timeout_seconds))


def pass_at_k(n, correct, k):
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= correct <= n and 1 <= k <= n")
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod((n - correct - i) / (n - i) for i in range(k))


def korean_ratio(response):
    letters = [c for c in without_latex(response) if c.isalpha()]
    return sum(is_hangul(c) for c in letters) / len(letters) if letters else 0.0


def mean(values):
    return sum(values) / len(values) if values else None


def metrics(records, n, ks):
    if not records:
        raise ValueError("Cannot evaluate an empty dataset")
    groups = defaultdict(list)
    for row in records:
        groups[str(row["id"])].append(row)
    for identifier, group in groups.items():
        if len(group) != n or {r["sample_index"] for r in group} != set(range(n)):
            raise ValueError(f"Incomplete or duplicate generations for {identifier}")
    return {
        "problems": len(groups),
        "samples_per_problem": n,
        "responses": len(records),
        f"avg@{n}": mean([int(r["correct"]) for r in records]),
        "pass@k": {
            str(k): mean(
                [pass_at_k(n, sum(r["correct"] for r in group), k) for group in groups.values()]
            )
            for k in ks
            if k <= n
        },
        "response_length_tokens": {
            "all": mean([r["token_count"] for r in records]),
            "correct": mean([r["token_count"] for r in records if r["correct"]]),
            "incorrect": mean([r["token_count"] for r in records if not r["correct"]]),
        },
        "korean_response_ratio": mean([r["korean_response_ratio"] for r in records]),
    }
