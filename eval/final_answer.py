"""Gold-independent final-answer extraction for base-model pool annotation only."""
import re

from common.math_text import last_boxed

# Keep benchmark scoring unchanged; this policy is explicitly versioned for the pool.
VERSION = 'explicit-final-v3-other'
MATH_SPAN = re.compile(r'\$\$([\s\S]*?)\$\$|\$([^$\n]+)\$|\\\[([\s\S]*?)\\\]|\\\((.*?)\\\)')
CONCLUSION = re.compile(r'\b(?:therefore|thus|hence|so|in conclusion)\b|\b(?:the\s+)?final answer\b|\banswer\s*:', re.I)


def strip_degrees(value, problem):
    # A degree label is not an arbitrary unit: only strip it for explicitly angular questions.
    angular = re.search(r'angle|degree|∠|\\angle|\\circ', problem, re.I)
    if angular:
        value = re.sub(r'(?:\^\s*\{?\s*\\circ\s*\}?|°|\\text\s*\{\s*degrees?\s*\}|\bdegrees?\b)\s*$', '', value).strip()
    return value


def clean_math(value):
    value = value.strip().strip('$').strip()
    value = re.sub(r'^\\[\[(]|\\[\])]$', '', value).strip()
    return value


def math_only(value, allow_other=False):
    value = clean_math(value).rstrip('.').strip()
    if not value or len(value) > 400 or re.search(r'[\n;]|\b(?:or|and)\b', value):
        return None
    # Permit a small unit vocabulary in LaTeX labels, without accepting arbitrary prose.
    # This affects lexical validation only; the original expression goes to Math-Verify.
    lexical = re.sub(r'\\(?:text|mathrm)\s*\{[~\s]*(?:mm|cm|km|m|mg|kg|g|s|ft|in|yd)[~\s]*\}', '', value)
    if allow_other:
        from eval.other_answers import special_candidate
        special = special_candidate(value)
        if special is not None:
            return special
    words = re.sub(r'\\[A-Za-z]+', '', lexical)
    if re.search(r'[A-Za-z]{2,}', words):
        return None
    if not re.fullmatch(r'[A-Za-z0-9_α-ωΑ-Ω\s\\{}()[\]+*/^=<>.,!|%°−-]+', lexical):
        return None
    if not re.search(r'[A-Za-z0-9α-ωΑ-Ω]', value):
        return None
    return value


def sentence_answer(sentence, allow_other=False):
    """Extract one claimed value, without looking at any reference answer."""
    if re.search(r"\b(?:not|wrong|incorrect|cannot|isn't)\b", sentence, re.I):
        return None
    emphasized = list(re.finditer(r'\*\*([^*]+)\*\*', sentence))
    if len(emphasized) == 1:
        emphasis = emphasized[0]
        outside = sentence[:emphasis.start()] + sentence[emphasis.end():]
        if not re.search(r'\d|\bor\b|\band\b', outside):
            candidate = math_only(emphasis.group(1), allow_other)
            if candidate:
                return candidate
    sentence = sentence.replace('**', '').strip()
    markers = list(re.finditer(r'(?:\banswer(?:\s*\([^)]*\))?\s*(?:is|=|:)|\b(?:is|are|was|equals)\s+)', sentence, re.I))
    direct = [m for m in markers if re.match(r'(?:approximately\s+|about\s+|exactly\s+)?(?:[+-]?\d|\$|\\|\()', sentence[m.end():].strip(), re.I)]
    if len(direct) > 1:
        return None
    markers = direct or markers
    if markers:
        sentence = sentence[markers[-1].end():].lstrip(': \n')
        sentence = re.sub(r'^(?:approximately|about|exactly)\s+', '', sentence, flags=re.I)
    spans = list(MATH_SPAN.finditer(sentence))
    if len(spans) == 1:
        span = spans[0]
        # Reject alternative answers/additional numbers outside the math span.
        outside = sentence[:span.start()] + sentence[span.end():]
        if re.search(r'\d|\bor\b', outside):
            return None
        return math_only(next(g for g in span.groups() if g is not None), allow_other)
    if len(spans) > 1:
        return None
    # Explicit answer marker, or grammatical statement "there are 2.1 pints".
    if not markers:
        return None
    tail = sentence
    # Prefer an entire expression before resorting to a numeric value plus prose/units.
    whole = math_only(tail, allow_other)
    if whole:
        return whole
    number = re.match(r'([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+)?)(?=\s|[.,;!]|$)', tail)
    if not number:
        return None
    rest = tail[number.end():]
    if re.search(r'\bor\b|\band\b|[+*/=<>]', rest):
        return None
    return number.group(1)


def extract_final(response, problem='', answer_type=None):
    allow_other = answer_type == 'other'
    boxed = last_boxed(response)
    if boxed is not None:
        value = clean_math(boxed)
        method = 'boxed'
    elif re.search(r'\\boxed\s*\{', response):
        return '', {'method': 'malformed_box', 'raw_answer': ''}
    else:
        # Code is never treated as a final answer; keep explicit prose before code.
        prose = re.sub(r'```[\s\S]*?(?:```|$)', '', response)
        anchors = list(CONCLUSION.finditer(prose))
        if allow_other and not anchors:
            anchors = list(re.finditer(r'\b(?:the\s+)?answer\s+is\b', prose, re.I))
        value, method = None, 'missing'
        if anchors:
            tail = prose[anchors[-1].start():].strip()
            tail = re.sub(r'(:|\bis)\s*\n\s*\n', r'\1 ', tail)
            # First sentence of the last conclusion, preserving decimal points and LaTeX.
            sentence = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n\s*\n', tail, maxsplit=1)[0]
            value = sentence_answer(sentence, allow_other)
            method = 'explicit_final' if value else 'ambiguous_final'
        else:
            lines = [line.strip() for line in prose.splitlines() if line.strip()]
            if lines:
                line = lines[-1]
                span = MATH_SPAN.fullmatch(line)
                if span:
                    line = next(g for g in span.groups() if g is not None)
                value = math_only(line, allow_other)
                method = 'final_math_line' if value else 'missing'
        if value is None:
            return '', {'method': method, 'raw_answer': ''}
    normalized = strip_degrees(value, problem)
    return normalized, {'method': method, 'raw_answer': value,
                        'degree_label_removed': normalized != value}



FACTOR_TASK = re.compile(r'\b(?:factoriz(?:e|ation|ing)|factoris(?:e|ation|ing)|factor\s+(?:completely|the|this)|factored form)\b', re.I)
EQUATION_TASK = re.compile(r'\b(?:form|find|construct|write|determine)\b[^.?!\n]{0,150}\bequation\b', re.I)


def normalize_radix(value, problem):
    """Convert a complete numeral only when the question establishes radix context."""
    context = re.search(r'\bbinary\b|\bbase\b|[)}\d]\s*_\s*\{?\d+', problem, re.I)
    if not context:
        return value
    compact = re.sub(r'\s+|\\left|\\right', '', value)
    match = re.fullmatch(r'(?:\(([0-9A-Za-z]+)\)|\{([0-9A-Za-z]+)\}|([0-9A-Za-z]+))_(?:\{(\d+)\}|(\d+))', compact)
    if not match:
        return value
    digits = next(x for x in match.groups()[:3] if x is not None)
    base = int(match.group(4) or match.group(5))
    if not 2 <= base <= 36 or len(digits) > 128:
        raise ValueError('Unsupported radix numeral')
    return str(int(digits, base))  # Invalid digits must fail, never fall back to a subscript parse.


def parse_math(value, timeout=5):
    from math_verify import LatexExtractionConfig, parse
    return parse('$' + value + '$', extraction_config=[LatexExtractionConfig()],
                 fallback_mode='no_fallback', parsing_timeout=timeout)


def polynomial_equations_equal(gold, answer, allow_rename=False):
    """Compare bounded rational univariate polynomials up to a nonzero constant.

    Never cancel variable factors, divide rational equations, or accept a bare RHS.
    Returns None outside this deliberately narrow normalization domain.
    """
    import sympy as sp
    if not isinstance(gold, sp.Equality) or not isinstance(answer, sp.Equality):
        return None
    polys = []
    symbols = []
    for equation in (gold, answer):
        free = equation.free_symbols
        if len(free) != 1 or sp.count_ops(equation) > 100:
            return None
        symbol = next(iter(free))
        # Check each side BEFORE subtraction, so rational terms cannot disappear.
        if not all(side.is_polynomial(symbol) for side in (equation.lhs, equation.rhs)):
            return None
        try:
            poly = sp.Poly(equation.lhs-equation.rhs, symbol, domain=sp.QQ)
        except (sp.PolynomialError, sp.CoercionFailed):
            return None
        if not 1 <= poly.degree() <= 12:
            return None
        polys.append(poly.monic().all_coeffs())
        symbols.append(symbol)
    if symbols[0] != symbols[1] and not allow_rename:
        return False
    return polys[0] == polys[1]


def factor_form_valid(expression):
    """Require complete polynomial factors over Q, not just algebraic equivalence.

    None means the requested form is outside the supported bounded univariate case.
    """
    import sympy as sp
    if not isinstance(expression, sp.Expr) or len(expression.free_symbols) != 1 or sp.count_ops(expression) > 100:
        return None
    symbol = next(iter(expression.free_symbols))
    try:
        poly = sp.Poly(expression, symbol, domain=sp.QQ)
        if not 1 <= poly.degree() <= 12:
            return None
        for term in sp.Mul.make_args(expression):
            base, exponent = term.as_base_exp()
            if not base.has(symbol):
                continue
            if exponent.is_Integer is not True or exponent <= 0:
                return False
            factor = sp.Poly(base, symbol, domain=sp.QQ)
            if not factor.is_irreducible:
                return False
    except (sp.PolynomialError, sp.CoercionFailed):
        return None
    return True


def score_final(response, gold, problem='', timeout=5, answer_type=None):
    from math_verify import verify
    from math_verify.utils import timeout as bounded
    import sympy as sp
    answer, audit = extract_final(response, problem, answer_type)
    audit['policy_version'] = VERSION
    if not answer:
        return answer, False, audit
    reference = strip_degrees(str(gold).strip().strip('$'), problem)
    # Only other answers receive the new typed normalization. Other types retain v2 behavior.
    if answer_type == 'other':
        from eval.other_answers import OtherFormatError, normalize_pair
        try:
            canonical, reference, answer, details = normalize_pair(answer, reference, problem, timeout)
            audit.update(details)
        except OtherFormatError as error:
            audit.update(method='unsupported_other_format', failure=str(error))
            return '', False, audit
    else:
        try:
            normalized = normalize_radix(answer, problem)
            reference = normalize_radix(reference, problem)
        except ValueError:
            audit['method'] = 'invalid_radix'
            return '', False, audit
        audit['radix_normalized'] = normalized != answer
        answer = canonical = normalized
    if math_only(canonical) is None:
        audit['method'] = 'invalid_math'
        return '', False, audit
    a, g = parse_math(canonical, timeout), parse_math(reference, timeout)
    audit.update(parsed=bool(a), gold_parsed=bool(g))
    if not a:
        audit['method'] = 'unparseable_final'
        return '', False, audit
    if not g:
        audit['failure'] = 'unparseable_gold'
        return answer, False, audit
    if FACTOR_TASK.search(problem):
        form = bounded(timeout)(factor_form_valid)(a[0])
        audit['factor_form_valid'] = form
        if form is not True:
            audit['failure'] = 'unfactored_or_unsupported_form'
            return answer, False, audit
    if isinstance(g[0], sp.Equality) and EQUATION_TASK.search(problem) and not isinstance(a[0], sp.Equality):
        audit['failure'] = 'expected_complete_equation'
        return answer, False, audit
    equation_match = bounded(timeout)(polynomial_equations_equal)(g[0], a[0], bool(EQUATION_TASK.search(problem)))
    if equation_match is not None:
        audit['comparison'] = 'rational_polynomial_equation'
        return answer, bool(equation_match), audit
    return answer, bool(verify(g, a, timeout_seconds=timeout)), audit
