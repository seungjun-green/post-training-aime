"""Post-translation hard checks and advisory checks, without hiding source text."""

import re
from collections import Counter

from common.math_text import escaped, is_hangul, last_boxed, protected_spans

CHECKS_VERSION = 2
# Do not consume English suffixes: 7th -> 7학년 preserves the numeric literal.
# Superscript/subscript glyphs remain distinct from ordinary digits.
NUMBERS = re.compile(
    r"[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+\-]?\d+)?|[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]+"
)
SYMBOLS = re.compile(r"[+*/^=<>≤≥≠≈×÷±∓√{}]")


def literal_sequence(text):
    """Ordered, nonoverlapping LaTeX/code spans, including bare commands/boxes."""
    spans = protected_spans(text)
    for match in re.finditer(r"\\[A-Za-z]+\*?", text):
        end = match.end()
        while True:
            start = end
            while start < len(text) and text[start] in " \t":
                start += 1
            if start == len(text) or text[start] != "{":
                break
            depth, stop = 1, start + 1
            while stop < len(text) and depth:
                if not escaped(text, stop):
                    depth += (text[stop] == "{") - (text[stop] == "}")
                stop += 1
            if depth:
                break
            end = stop
        spans.append((match.start(), end))
    merged = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [text[a:b] for a, b in merged]


def hard_checks(source, target):
    if not isinstance(target, str) or not target.strip():
        return ["empty_translation"]
    issues = []
    before, after = Counter(NUMBERS.findall(source)), Counter(NUMBERS.findall(target))
    if before != after:
        missing, added = list((before - after).elements()), list((after - before).elements())
        issues.append(f"numbers_changed: missing={missing[:8]!r}, added={added[:8]!r}")
    before, after = literal_sequence(source), literal_sequence(target)
    if before != after:
        index = next(
            (i for i, (a, b) in enumerate(zip(before, after)) if a != b),
            min(len(before), len(after)),
        )
        issues.append(
            f"latex_or_code_sequence_changed: first_difference={index}, "
            f"source_count={len(before)}, target_count={len(after)}"
        )
    if last_boxed(source) != last_boxed(target):
        issues.append("final_boxed_answer_changed")
    if r"\boxed" in source and last_boxed(source) is None:
        issues.append("source_final_boxed_answer_malformed")
    return issues


def soft_checks(source, target, config):
    warnings = []
    ratio = len(target) / len(source) if source else 0
    if not config["LENGTH_RATIO_MIN"] <= ratio <= config["LENGTH_RATIO_MAX"]:
        warnings.append(f"length_ratio={ratio:.4f}")
    if not any(is_hangul(c) for c in target):
        warnings.append("hangul_missing")
    if Counter(SYMBOLS.findall(source)) != Counter(SYMBOLS.findall(target)):
        warnings.append("undelimited_math_symbols_changed")
    if len(re.split(r"\n\s*\n", source)) != len(re.split(r"\n\s*\n", target)):
        warnings.append("paragraph_count_changed")
    return warnings
