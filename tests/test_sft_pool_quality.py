import hashlib
import itertools
import math
from pathlib import Path

import pytest
import yaml

from eval.final_answer import extract_final, score_final
from pipeline.sft_pool_quality import clean_rows, review_row

ROOT = Path(__file__).resolve().parents[1]
SPEC = {'id_column': 'id', 'problem_column': 'problem', 'answer_column': 'gold_answer'}


@pytest.mark.parametrize('text,answer', [
    ('So, there are approximately 2.1 pints in one liter.', '2.1'),
    ('So, he drank **3** gallons.', '3'),
    ('So the value of $x^2+4y^2$ is 48.', '48'),
    ('Therefore, the value of $d$ is $1$.', '1'),
    ('So, there are 360 integers in which digit 1 is left of digit 6.', '360'),
    ('Answer: 42', '42'),
    ('Final answer is:\n\n42', '42'),
    (r'Work first.'+'\n'+r'\[x = \frac{1}{2}\]', r'x = \frac{1}{2}'),
])
def test_explicit_final_answers(text, answer):
    assert extract_final(text)[0] == answer
    assert score_final(text, answer)[1]


@pytest.mark.parametrize('text', [
    'An intermediate calculation gives 42. Therefore the answer is 7.',
    r'\boxed{7} Later text says the answer is 42.',
    'An intermediate calculation gives 42. I cannot determine the final answer.',
    'Therefore the answer is either 7 or 42.',
    'Final answer is $7$ or $42$.',
    'Thus **42** is not the answer.',
    r'\boxed{7} followed by malformed \boxed{42',
    '```python\nanswer = 42\n```',
])
def test_no_gold_hunting_or_ambiguous_rescue(text):
    assert not score_final(text, '42')[1]


def test_degrees_require_angular_context():
    text = r'\boxed{30 °}'
    assert score_final(text, '30', 'Find angle ABC.')[1]
    assert not score_final(text, '30', 'Compute a number.')[1]


def sample(problem, gold='42', identifier='example'):
    return {'id': identifier, 'problem': problem, 'gold_answer': gold, 'domain': 'math'}


def test_hash_guarded_review_and_original_preservation():
    row = sample('Compute 6 times 7.', '41')
    registry = {'version': 'test', 'reviewed_rows': {'example': {
        'problem_sha256': hashlib.sha256(row['problem'].encode()).hexdigest(),
        'expected_gold': '41', 'action': 'correct', 'corrected_gold': '42', 'reason': 'reviewed'}}}
    repaired, audit = review_row(row, registry, SPEC)
    assert repaired['gold_answer'] == '42' and repaired['gold_answer_original'] == '41'
    assert row['gold_answer'] == '41' and repaired['problem'] == row['problem']
    assert audit['original'] == row
    with pytest.raises(ValueError, match='stale correction'):
        review_row({**row, 'problem': 'Changed question'}, registry, SPEC)


def test_quality_quarantine_counts():
    rows = [sample('Compute 6 times 7.', identifier='a'),
            sample('Use [img]https://example.com/a.png[/img]', identifier='b'),
            sample('Prove the theorem.', identifier='c'),
            sample('(a) Find x. (b) Find y.', identifier='d')]
    clean, audit, report = clean_rows(rows, {'version': 'test', 'reviewed_rows': {}}, SPEC)
    assert len(clean) == 1 and len(audit) == 3
    assert report['input_rows'] == report['eligible_rows'] + report['quarantined_rows']


def test_reviewed_gold_fixes_are_mathematically_checked():
    registry = yaml.safe_load((ROOT / 'configs/sft_pool_quality.yaml').read_text())
    corrected = [x['corrected_gold'] for x in registry['reviewed_rows'].values() if x['action'] == 'correct']
    assert len(corrected) == 6
    assert sum(math.comb(2024, k) for k in range(65)) % 2027 == 1089
    # Verify cycle and pyramid formulas directly against small labeled colorings.
    for m, n in [(3, 3), (3, 4), (4, 3), (4, 4)]:
        cycles = sum(all(colors[i] != colors[(i+1) % n] for i in range(n))
                     for colors in itertools.product(range(m), repeat=n))
        assert cycles == (m-1)**n + (-1)**n*(m-1)
        pyramids = sum(all(colors[i] != colors[(i+1) % n] and colors[i] != colors[-1] for i in range(n))
                       for colors in itertools.product(range(m), repeat=n+1))
        assert pyramids == m*((m-2)**n + (-1)**n*(m-2))
    import sympy as sp
    t = sp.symbols('t', positive=True)
    f = sp.sqrt(t) + 6/sp.sqrt(t)
    assert sp.simplify(sp.diff(f, t) - (t-6)/(2*t**sp.Rational(3, 2))) == 0
    assert f.subs(t, 7) == 13/sp.sqrt(7)


def test_prepare_screens_before_selection_and_exports_quality_audit(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import huggingface_hub
    from datasets import Dataset

    from pipeline import sample_sft_pool as sampling
    cfg = yaml.safe_load((ROOT / 'configs/sft_pool_sampling.yaml').read_text())
    original = [sample(f'Compute 6 times 7 for case {i}.', identifier=f'fixture-{i}') for i in range(55)]
    original[3]['problem'] = 'Find the value in the following diagram.'
    ds = Dataset.from_list(original)
    monkeypatch.setattr(sampling, 'load_source', lambda *args: ds)
    class Api:
        def __init__(self, **kwargs):
            pass
        def dataset_info(self, *args, **kwargs):
            return SimpleNamespace(sha=cfg['data']['revision'])
        def model_info(self, *args, **kwargs):
            return SimpleNamespace(sha=cfg['model']['revision'])
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    prepared, manifest = sampling.prepare(cfg, ROOT, tmp_path, None)
    assert len(prepared) == 54
    assert 'fixture-3' not in prepared['id']
    assert prepared[0]['gold_answer_original'] == '42'
    assert len(manifest['smoke_indices']) == 50
    summary = json.loads((tmp_path / 'quality_summary.json').read_text())
    assert summary['quarantined_rows'] == 1
    again, repeat = sampling.prepare(cfg, ROOT, tmp_path, None)
    assert again.to_list() == prepared.to_list() and repeat == manifest


@pytest.mark.parametrize('answer,expected', [
    ('42x^2-46x+12=0', True), ('0=42x^2-46x+12', True),
    ('21x^2-23x+6=0', True), ('0', False),
    ('42x^2-46x+11=0', False), ('x*(21x^2-23x+6)=0', False),
])
def test_complete_equations_and_no_added_roots(answer, expected):
    extracted, correct, _ = score_final('\\boxed{' + answer + '}', '21y^2-23y+6=0',
                                         'Form a new quadratic equation with the given roots.')
    assert correct == expected
    assert extracted == answer


def test_no_arbitrary_variable_renaming_or_equation_rhs_loss():
    assert not score_final(r'\boxed{y=2}', 'x=2', 'Give the relation.')[1]
    assert score_final(r'\boxed{x=2}', '2', 'Find x.')[1]
    assert extract_final(r'\boxed{y=x+1}')[0] == 'y=x+1'
    assert extract_final(r'\boxed{x^2=4}')[0] == 'x^2=4'


@pytest.mark.parametrize('answer,expected', [
    ('x^7+2x^5+x+2', False),
    ('(x^2+x+1)(x^5-x^4+2x^3-x^2-x+2)', True),
    ('1*(x^7+2x^5+x+2)', False),
    ('(x^2+x+1)(x^5-x^4+2x^3-x^2-x+3)', False),
])
def test_factorization_requires_form_and_equivalence(answer, expected):
    assert score_final('\\boxed{' + answer + '}',
                       '(x^2+x+1)(x^5-x^4+2x^3-x^2-x+2)',
                       'Factorization: x^7+2x^5+x+2.')[1] == expected


def test_incomplete_factorization_and_repeated_factors():
    assert not score_final(r'\boxed{(x^2-1)(x+2)}', '(x-1)(x+1)(x+2)', 'Factor completely.')[1]
    assert score_final(r'\boxed{(x+1)^2}', '(x+1)^2', 'Factor completely.')[1]


@pytest.mark.parametrize('answer,expected', [
    ('10010_2', True), ('(10010)_{2}', True), ('18', True),
    ('10010', False), ('10011_2', False), ('102_2', False),
])
def test_binary_normalization(answer, expected):
    result = score_final('\\boxed{' + answer + '}', '(10010)_{2}',
                         'Find $(11101)_{2}-(1011)_{2}$.')
    assert result[1] == expected
    if answer == '102_2':
        assert result[0] == ''


def test_radix_context_does_not_rewrite_indexed_numbers():
    from eval.final_answer import normalize_radix
    assert normalize_radix('10010_2', 'Compute the indexed expression.') == '10010_2'


@pytest.mark.parametrize('answer', ['))))', '缛فلة𝟭𝟭𝟭𝟭', '这是答案', '!!!', r'\frac{}{}'])
def test_junk_is_empty_extraction(answer):
    extracted, correct, _ = score_final('\\boxed{' + answer + '}', '42')
    assert extracted == '' and not correct


def test_source_wide_review_rules():
    registry = {'version': 'test', 'reviewed_rows': {},
                'quarantine_unreviewed_sub_sources': ['synthetic_math'],
                'screen_reference_parseability': True}
    rows = [dict(sample('Compute 6 times 7.', identifier='synthetic'), sub_source='synthetic_math'),
            sample('Compute 6 times 7.', '))))', 'bad_gold'),
            sample('Factor completely.', '(x^2-1)(x+2)', 'bad_form'),
            sample('Compute 6 times 7.', identifier='valid')]
    clean, audit, report = clean_rows(rows, registry, SPEC)
    assert [r['id'] for r in clean] == ['valid']
    assert {r['reason'] for r in audit} == {'unreviewed_source_subset', 'unparseable_reference', 'factorization_reference_requires_review'}


def test_second_smoke_gold_and_question_counterexamples():
    import sympy as sp
    assert all(a*b == 4*(a+b) for a,b in [(5,20),(6,12)])
    for m,n,p,q in [(2,1,6,4),(3,2,5,3)]:
        assert (m+p,n+p,m+q,n+q) == (8,7,6,5)
    c = sp.symbols('c')
    area_squared = (1-c*c)*(1-c)**2/4
    assert sp.factor(sp.diff(area_squared,c)) == -(c-1)**2*(2*c+1)/2
    assert area_squared.subs(c,-sp.Rational(1,2)) == sp.Rational(27,64)
    # P is the midpoint of A,B: their signed triangle area vanishes.
    a,b = sp.Point(1,0), sp.Point(sp.Rational(1,2),sp.sqrt(3)/2)
    p = sp.Segment(a,b).midpoint
    assert sp.det(sp.Matrix([[*(b-a)], [*(p-a)]])) == 0


def test_replay_keeps_eligible_ids_and_reports_replacements():
    from pipeline.sample_sft_pool import select_smoke_rows
    rows = [sample(f'Problem {i}', identifier=str(i)) for i in range(60)]
    spec = {'rows': 50, 'seed': 42, 'preferred_ids': ['0','1','quarantined']}
    selection, report = select_smoke_rows(rows, spec)
    assert len(selection) == len(set(selection)) == 50
    assert {0,1} <= set(selection)
    assert report['replayed_ids'] == ['0','1']
    assert report['excluded_previous_ids'] == ['quarantined']
    assert len(report['new_ids']) == 48
    assert select_smoke_rows(rows, spec) == (selection, report)


def test_regrade_preserves_generation_and_handles_prior_correction(tmp_path):
    import json
    from pipeline.sample_sft_pool import regrade_smoke
    root = tmp_path / 'code'
    (root / 'configs').mkdir(parents=True)
    rows = []
    for i in range(50):
        row = sample('Compute 6 times 7.', identifier=str(i))
        row.update(responses=[r'\boxed{42}']*8, extracted_answers=['42']*8,
                   correct=[True]*8, response_tokens=[6]*8, finish_reasons=['stop']*8,
                   num_correct=8)
        rows.append(row)
    rows[0].update(gold_answer_original='41', quality_action='gold_corrected')
    registry = {'version':'test', 'reviewed_rows': {'0': {
        'action':'correct', 'expected_gold':'41', 'corrected_gold':'42',
        'problem_sha256':hashlib.sha256(rows[0]['problem'].encode()).hexdigest(),
        'reason':'incorrect_gold'}}}
    (root / 'configs/sft_pool_quality.yaml').write_text(yaml.safe_dump(registry))
    path = tmp_path / 'input.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out = regrade_smoke(path, root, tmp_path / 'out', {'data':SPEC, 'sampling':{'verify_timeout_seconds':5}})
    regraded = [json.loads(s) for s in (out/'smoke_results_regraded.jsonl').read_text().splitlines()]
    assert regraded[0]['gold_answer_original'] == '41'
    for before, after in zip(rows,regraded):
        for field in ['responses','response_tokens','finish_reasons']:
            assert before[field] == after[field]
    assert json.loads((out/'summary.json').read_text())['grading_error_count'] == 0
    changes = [json.loads(s) for s in (out/'grading_changes.jsonl').read_text().splitlines()]
    assert changes[0]['old_gold'] == '42'


@pytest.mark.parametrize('answer', [r'52\, \text{cm}^2', r'52 \mathrm{~cm}^2'])
def test_valid_latex_unit_labels_remain_parseable(answer):
    extracted, correct, _ = score_final('\\boxed{' + answer + '}', '52', 'Find the area difference in square cm.')
    assert extracted and correct


@pytest.mark.parametrize('problem,gold,reason', [
    ('Compute 6 times 7.', '42', None),
    ('Use [img]https://example.com/a.png[/img]', '42', 'image_reference'),
    ('Find the area in the following diagram.', '42', 'diagram_requires_review'),
    ('Prove the theorem.', '42', 'proof_requires_review'),
    ('(a) Find x. (b) Find y.', '42', 'multipart_requires_review'),
    ('Find the number. (A) 41 (B) 42', '42', 'multiple_choice'),
    ('Find both values.', '42', 'multiple_required_values'),
    ('Compute 2^34^5.', '42', 'ambiguous_concatenated_powers'),
    ('Compute 6 times 7.', '))))', 'unparseable_reference'),
    ('Factor completely.', '(x^2-1)(x+2)', 'factorization_reference_requires_review'),
])
def test_current_policy_filters_content_equally_for_synthetic_rows(problem, gold, reason):
    registry = yaml.safe_load((ROOT / 'configs/sft_pool_quality.yaml').read_text())
    assert registry['quarantine_unreviewed_sub_sources'] == []
    for source in ['synthetic_math', 'algebra']:
        row = dict(sample(problem, gold), sub_source=source)
        kept, audit = review_row(row, registry, SPEC)
        if reason is None:
            assert kept is not None and audit is None
        else:
            assert kept is None and audit['reason'] == reason


def test_manual_quarantine_still_applies_to_synthetic_rows():
    registry = yaml.safe_load((ROOT / 'configs/sft_pool_quality.yaml').read_text())
    row = dict(sample('Compute 6 times 7.'), sub_source='synthetic_math')
    registry['reviewed_rows'][row['id']] = {
        'problem_sha256': hashlib.sha256(row['problem'].encode()).hexdigest(),
        'expected_gold': row['gold_answer'], 'action': 'quarantine', 'reason': 'reviewed_problem'}
    kept, audit = review_row(row, registry, SPEC)
    assert kept is None and audit['reason'] == 'reviewed_problem'
