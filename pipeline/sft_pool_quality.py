"""Conservative pre-generation QA plus hash-guarded, mathematically reviewed fixes."""
import hashlib
import re
from collections import Counter
from functools import lru_cache

from eval.final_answer import FACTOR_TASK, factor_form_valid, normalize_radix, parse_math, strip_degrees


def review_row(row, registry, spec):
    identifier = str(row[spec['id_column']])
    text, gold = row[spec['problem_column']], str(row[spec['answer_column']])
    action = registry['reviewed_rows'].get(identifier)
    if action:
        if (hashlib.sha256(text.encode()).hexdigest() != action['problem_sha256'] or gold != action['expected_gold']):
            raise ValueError(f'Reviewed source changed for {identifier}; do not apply a stale correction')
        if action['action'] == 'quarantine':
            return None, dict(action, id=identifier, original=row)
    reasons = []
    if not action and row.get('sub_source') in registry.get('quarantine_unreviewed_sub_sources', []):
        reasons.append('unreviewed_source_subset')
    if re.search(r'\d+\^\d+\d\^\d+', text):
        reasons.append('ambiguous_concatenated_powers')
    if registry.get('screen_reference_parseability'):
        reference_issue = reference_quality(action['corrected_gold'] if action else gold, text)
        if reference_issue:
            reasons.append(reference_issue)
    if re.search(r'\[img\]|!\[[^\]]*\]\(|<img\b|https?://\S+\.(?:png|jpg|jpeg|svg|gif)', text, re.I):
        reasons.append('image_reference')
    if re.search(r'(?:as shown|shown below|following diagram|this figure|in the (?:figure|diagram))', text, re.I):
        reasons.append('diagram_requires_review')
    if re.search(r'\b(?:prove|show that|demonstrate that)\b', text, re.I):
        reasons.append('proof_requires_review')
    if len(re.findall(r'\([a-d]\)|(?m:^\s*[1-9][.)]\s)|\((?:i|ii|iii)\)', text)) >= 2:
        reasons.append('multipart_requires_review')
    if len(re.findall(r'\([A-E]\)|(?m:^\s*[A-E][.)]\s)', text)) >= 2:
        reasons.append('multiple_choice')
    if re.search(r'(?:find|determine|calculate)\b[^.?!\n]{0,100}\b(?:maximum and minimum|minimum and maximum|both)\b', text, re.I):
        reasons.append('multiple_required_values')
    if text.count('?') > 1:
        reasons.append('multiple_questions_requires_review')
    if re.search(r'\band\s*\$0+\$|\b(?:solution|proof)\s*:', text, re.I):
        reasons.append('malformed_or_solution_fragment')
    if reasons:
        return None, {'id': identifier, 'action': 'quarantine', 'reason': reasons[0],
                      'all_reasons': reasons, 'original': row}
    repaired = dict(row)
    repaired['gold_answer_original'] = row.get('gold_answer_original', gold)
    repaired['quality_action'] = 'gold_corrected' if action else 'unchanged'
    if action:
        repaired[spec['answer_column']] = action['corrected_gold']
        return repaired, dict(action, id=identifier, original=row)
    return repaired, None


def clean_rows(rows, registry, spec):
    clean, audit = [], []
    for index, row in enumerate(rows):
        kept, decision = review_row(row, registry, spec)
        if decision:
            audit.append(dict(decision, source_index=index))
        if kept:
            clean.append(kept)
    summary = {'input_rows': len(rows), 'eligible_rows': len(clean),
               'quarantined_rows': sum(x['action'] == 'quarantine' for x in audit),
               'corrected_rows': sum(x['action'] == 'correct' for x in audit),
               'reasons': dict(Counter(x['reason'] for x in audit)),
               'policy_version': registry['version'],
               'quarantined_sub_sources': dict(Counter(x['original'].get('sub_source', 'unknown') for x in audit if x['action'] == 'quarantine')),
               'limitation': 'Rule-based screening and reviewed corrections; not an exhaustive semantic gold audit.'}
    assert summary['input_rows'] == summary['eligible_rows'] + summary['quarantined_rows']
    return clean, audit, summary


@lru_cache(maxsize=32768)
def reference_quality(gold, problem):
    """Check reference parseability and supported answer forms before GPU work."""
    from math_verify.utils import timeout
    try:
        parsed = parse_math(normalize_radix(strip_degrees(str(gold), problem), problem))
        if not parsed:
            return 'unparseable_reference'
        if FACTOR_TASK.search(problem) and timeout(5)(factor_form_valid)(parsed[0]) is not True:
            return 'factorization_reference_requires_review'
    except Exception:
        return 'reference_check_failed'
    return None
