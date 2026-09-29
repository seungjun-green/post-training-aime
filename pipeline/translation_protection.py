"""Conservative, lossless shielding for DeepSeek translation requests.

This checks literal preservation, not mathematical meaning or prose completeness.
Protected occurrences may move to accommodate Korean grammar, but may not change,
disappear, or multiply. Source text is never repaired.
"""

import re
from collections import Counter
from dataclasses import dataclass

from common.io import digest
from common.math_text import escaped, protected_spans

PROTECTION_VERSION = 1
# Include digit-bearing identifiers (S0, 8n, 3x3), decimals and scientific notation.
NUMERIC = re.compile(r"[A-Za-z_]*\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?[A-Za-z_\d]*")
MATH_SYMBOL = re.compile(r"[+*/^=<>≤≥≠≈×÷±∓√{}⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]")
MARKER = re.compile(r"⟪KEEP_[a-f0-9]+_\d+⟫")
COMMAND = re.compile(r"\\[A-Za-z]+\*?")


def literal_spans(text):
    spans = protected_spans(text)
    # Extend undelimited commands across ALL following balanced argument groups,
    # e.g. both arguments of \\frac{m}{n}, not just the numerator.
    for match in COMMAND.finditer(text):
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
    spans.extend((m.start(), m.end()) for m in NUMERIC.finditer(text))
    # Merge overlaps only; adjacent literals retain their own occurrence IDs.
    merged = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def preservation_issues(source, target):
    issues = []
    # A multiset permits natural word-order changes, while retaining multiplicity.
    for label, before, after in [
        (
            "protected_literals_changed",
            Counter(source[a:b] for a, b in literal_spans(source)),
            Counter(target[a:b] for a, b in literal_spans(target)),
        ),
        (
            "numeric_literals_changed",
            Counter(NUMERIC.findall(source)),
            Counter(NUMERIC.findall(target)),
        ),
        (
            "math_symbols_changed",
            Counter(MATH_SYMBOL.findall(source)),
            Counter(MATH_SYMBOL.findall(target)),
        ),
    ]:
        if before != after:
            missing = list((before - after).elements())[:5]
            added = list((after - before).elements())[:5]
            # Keep diagnostics bounded even for code blocks/long equations.
            issues.append(f"{label}: missing={str(missing)[:240]}, added={str(added)[:240]}")
    return issues


@dataclass
class ProtectedText:
    source: str
    masked: str
    replacements: dict[str, str]

    @classmethod
    def from_source(cls, source):
        # Fail closed on reserved syntax rather than guessing which marker is data.
        if "⟪KEEP_" in source:
            raise ValueError("Source contains reserved translation placeholder syntax")
        namespace = digest(source)[:12]
        replacements, pieces, last = {}, [], 0
        for index, (start, end) in enumerate(literal_spans(source)):
            marker = f"⟪KEEP_{namespace}_{index}⟫"
            replacements[marker] = source[start:end]
            pieces.extend([source[last:start], marker])
            last = end
        pieces.append(source[last:])
        return cls(source, "".join(pieces), replacements)

    def restore(self, output):
        found = Counter(MARKER.findall(output))
        if found != Counter(self.replacements.keys()):
            raise ValueError("placeholder_integrity: missing, duplicated or unknown placeholder")
        # Also reject damaged marker syntax alongside otherwise valid markers.
        remainder = MARKER.sub("", output)
        if "KEEP_" in remainder or "⟪" in remainder or "⟫" in remainder:
            raise ValueError("placeholder_integrity: malformed placeholder")
        restored = MARKER.sub(lambda m: self.replacements[m.group()], output)
        issues = preservation_issues(self.source, restored)
        if issues:
            raise ValueError("; ".join(issues))
        return restored
