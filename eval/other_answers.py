"""Narrow format normalization for answer_type=other; Math-Verify still compares values."""
import ast
import re


class OtherFormatError(ValueError):
    pass


NUMBER = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)'
COLON = re.compile(rf'\s*({NUMBER})\s*:\s*({NUMBER})\s*')
CLOCK = re.compile(r'(\d{1,2})\s*:\s*(\d{2})(?:\s*([ap])\.?m\.?)?', re.I)
CLOCK_HOUR = re.compile(r'(\d{1,2})\s*([ap])\.?m\.?', re.I)
RADIX = re.compile(r'(?:\(([0-9A-Za-z]+)\)|\{([0-9A-Za-z]+)\}|([0-9A-Za-z]+))_(?:\{(\d+)\}|(\d+))')
RATIO_CONTEXT = re.compile(r'\bratio\b|\bproportion\b|비율|비례', re.I)
CLOCK_CONTEXT = re.compile(r'\bclock\b|\bwhat time\b|\bat what time\b|\btime of day\b|\ba\.?m\.?\b|\bp\.?m\.?\b|시각|몇\s*시', re.I)


def tidy(value):
    value = value.strip().strip('$').strip().rstrip('.').strip()
    value = value.replace('：', ':').replace('−', '-').replace('％', '%')
    value = re.sub(r'\\(?:left|right)\b', '', value)
    value = re.sub(r'\\(?:,|;|!|quad\b|qquad\b)', ' ', value)
    value = re.sub(r'\\+%', '%', value)
    return value


def special_candidate(value):
    """Recognize entire final answers only, never numbers found inside reasoning."""
    value = tidy(value)
    if CLOCK.fullmatch(value) or CLOCK_HOUR.fullmatch(value) or COLON.fullmatch(value):
        return value
    if re.fullmatch(r"\[\s*(['\"])" + NUMBER + r"\1\s*\]", value):
        return value
    return None


def scalar_value(value, timeout):
    from eval.final_answer import parse_math
    import sympy as sp
    parsed = parse_math(value, timeout)
    if len(parsed) != 1 or not isinstance(parsed[0], sp.Expr) or parsed[0].free_symbols:
        raise OtherFormatError('not_a_numeric_scalar')
    if parsed[0].is_finite is not True:
        raise OtherFormatError('nonfinite_value')
    # Decimal literals are exact decimal rationals in an equality chain.
    # This avoids direction-dependent comparison of unevaluated Float sums.
    parsed[0] = parsed[0].xreplace({f: sp.Rational(str(f)) for f in parsed[0].atoms(sp.Float)})
    return parsed


def numeric_chain(value, timeout):
    """Only collapse a numeric equality chain after verifying EVERY equality."""
    from math_verify import verify
    # Symbolic equations stay intact. Approximate/inequality statements are not chains.
    if '=' not in value or any(op in value for op in ['<', '>', r'\le', r'\ge', r'\approx']):
        return value
    parts = value.split('=')
    if len(parts) > 8:
        raise OtherFormatError('equality_chain_too_long')
    parsed = []
    for part in parts:
        try:
            parsed.append(scalar_value(part.strip(), timeout))
        except OtherFormatError:
            # Preserve symbolic equations; do not discard unknown left-hand sides.
            if re.search(r'[A-Za-z]', re.sub(r'\\[A-Za-z]+', '', part)):
                return value
            raise
    if not all(verify(parsed[0], p, timeout_seconds=timeout) for p in parsed[1:]):
        raise OtherFormatError('inconsistent_numeric_equality')
    return parts[-1].strip()


def normalize(value, problem, kind, timeout):
    value = tidy(value)
    if len(value) > 400:
        raise OtherFormatError('answer_too_long')
    if re.match(r"\[\s*['\"]", value):
        try:
            unpacked = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise OtherFormatError('invalid_list_wrapper') from None
        if (not isinstance(unpacked, list) or len(unpacked) != 1 or
                not isinstance(unpacked[0], str) or not re.fullmatch(NUMBER, unpacked[0].strip())):
            raise OtherFormatError('not_a_single_numeric_string')
        value = unpacked[0].strip()
    if kind == 'clock':
        match = CLOCK.fullmatch(value)
        hour_only = CLOCK_HOUR.fullmatch(value)
        if match:
            hour, minute = int(match[1]), int(match[2])
            meridiem = match[3]
        elif hour_only:
            hour, minute, meridiem = int(hour_only[1]), 0, hour_only[2]
        else:
            raise OtherFormatError('expected_clock_time')
        if not 0 <= minute < 60:
            raise OtherFormatError('invalid_clock_minute')
        if meridiem:
            if not 1 <= hour <= 12:
                raise OtherFormatError('invalid_12_hour_clock')
            hour = hour % 12 + (12 if meridiem.lower() == 'p' else 0)
        elif not 0 <= hour <= 23:
            raise OtherFormatError('invalid_24_hour_clock')
        return str(hour*60+minute), f'{hour:02d}:{minute:02d}'
    if ':' in value:
        if kind != 'ratio':
            raise OtherFormatError('ambiguous_colon_without_context')
        match = COLON.fullmatch(value)
        if not match:
            raise OtherFormatError('unsupported_ratio_format')
        if float(match[2]) == 0:
            raise OtherFormatError('zero_ratio_denominator')
        value = r'\frac{' + match[1] + '}{' + match[2] + '}'
    compact = re.sub(r'\s+', '', value)
    radix = RADIX.fullmatch(compact)
    if radix:
        digits = next(v for v in radix.groups()[:3] if v is not None)
        base = int(radix[4] or radix[5])
        if not 2 <= base <= 36 or len(digits) > 128:
            raise OtherFormatError('unsupported_radix')
        try:
            value = str(int(digits, base))
        except ValueError:
            raise OtherFormatError('invalid_radix_digit') from None
    # Only a WHOLE numeric percentage is converted. A bare 20 stays 20, not 20%.
    percent = re.fullmatch(rf'({NUMBER})\s*%', value)
    if percent:
        value = r'\frac{' + percent[1] + '}{100}'
    if re.search(r'\^\s*\d+(?:\.\d+){2,}', value):
        raise OtherFormatError('ambiguous_unbraced_exponent')
    value = numeric_chain(value, timeout)
    return value, value


def normalize_pair(answer, reference, problem, timeout):
    """Determine colon semantics from question context, not matching answer values."""
    has_colon = ':' in tidy(answer) or ':' in tidy(reference)
    clock = bool(CLOCK_CONTEXT.search(problem))
    ratio = bool(RATIO_CONTEXT.search(problem))
    kind = 'clock' if clock and not ratio and has_colon else 'ratio' if ratio and not clock and has_colon else 'numeric_or_expression'
    a, display = normalize(answer, problem, kind, timeout)
    g, _ = normalize(reference, problem, kind, timeout)
    return a, g, display, {'other_kind':kind, 'normalized_answer':a, 'normalized_gold':g}
