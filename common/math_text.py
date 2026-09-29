"""Brace-aware boxed answers and protected LaTeX/code spans."""

import re


def escaped(text, index):
    count = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        count += 1
        index -= 1
    return count % 2 == 1


def last_boxed(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    match = matches[-1]
    depth = 1
    for i in range(match.end(), len(text)):
        if escaped(text, i):
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[match.end() : i]
    return None  # A malformed final box must not fall back to an earlier answer.


PROTECTED = re.compile(
    r"```[\s\S]*?```|`[^`\n]*`|\[asy\][\s\S]*?\[/asy\]"
    r"|\\begin\{(?P<env>[^}]+)\}[\s\S]*?\\end\{(?P=env)\}"
    r"|(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$"
    r"|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)"
    r"|(?<!\\)\$(?!\$)[\s\S]*?(?<!\\)\$"
)


def protected_spans(text):
    spans = [(m.start(), m.end()) for m in PROTECTED.finditer(text)]
    # Also protect undelimited commands with brace groups (notably boxed answers).
    for match in re.finditer(r"\\[A-Za-z]+\*?\s*\{", text):
        depth = 1
        for i in range(match.end(), len(text)):
            if escaped(text, i):
                continue
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), i + 1))
                    break
    return spans


def without_latex(text):
    chars = list(text)
    for start, end in protected_spans(text):
        chars[start:end] = " " * (end - start)
    return re.sub(r"\\[A-Za-z]+", " ", "".join(chars))


def is_hangul(char):
    return (
        "\uac00" <= char <= "\ud7a3" or "\u1100" <= char <= "\u11ff" or "\u3130" <= char <= "\u318f"
    )
