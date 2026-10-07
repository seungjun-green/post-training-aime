import itertools
import json
from pathlib import Path

import pytest
import yaml

from pipeline.sft_pool import (
    ReferenceIndex,
    adapt,
    build_pool,
    grams,
    iter_jsonl,
    normalize,
    parse_gold,
    rejection_reasons,
    s1_math,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / 'configs/sft_pool.yaml').read_text())


def row(source, identifier, problem, answer='42', **meta):
    return {'id': identifier, 'dataset_source': source, 'problem': problem,
            'gold_answer': answer, 'metadata': meta, 'sub_source': 'synthetic_fixture'}


def test_normalize_latex():
    assert normalize(r'$\displaystyle x^2+3x$') == normalize('x ^ {2} + 3 x')
    assert normalize(r'$\left(\dfrac{1}{2}\right)$') == normalize(r'\frac { 1 } { 2 }')
    assert normalize(r'\mathrm{x}\, + \quad 2') == normalize('x+2')
    assert normalize('Value 12') != normalize('Value 13')
    assert normalize(normalize(r'$x^2+3x$')) == normalize(r'$x^2+3x$')


def test_reference_injected_embedded_short_and_number_variant():
    text = 'Find the sum of all positive integers less than 100 satisfying these conditions'
    idx = ReferenceIndex([{'id': 'm500', 'set': 'math_500', 'problem': text},
                          {'id': 'short', 'set': 'test', 'problem': 'Compute 7 plus 8'}], 8)
    assert list(idx.matches(text))[0]['coverage'] == 1
    assert list(idx.matches('Context before. ' + text + ' End of question.'))[0]['coverage'] == 1
    assert list(idx.matches('Context. Compute 7 plus 8.'))[0]['coverage'] == 1
    changed = list(idx.matches(text.replace('100', '200')))
    assert changed and changed[0]['number_variant']


def test_distinct_and_no_aggregated_overlap():
    refs = [{'id': 'a', 'set': 'test', 'problem': 'one two three four five six'},
            {'id': 'b', 'set': 'test', 'problem': 'red blue green white black pink'}]
    idx = ReferenceIndex(refs, 2)
    matches = list(idx.matches('one two three. red blue green. one two three'))
    assert [m['coverage'] for m in matches] == [0.4, 0.4]
    assert len(grams('a b a b a b', 2)) == 2


def test_gold_no_parser_fallback_or_prose():
    for answer in ['hello', 'The answer is 42', '2, 3', 'A', 'True', '', r'\text{no solution}']:
        assert parse_gold(answer) is None
    for answer in ['42', r'\frac{1}{2}', r'\sqrt{2}', 'x^2+3*x']:
        assert parse_gold(answer) is not None


def test_s1_domain_and_gold_metadata():
    cfg = config()
    raw = {'question': 'Find the value.', 'solution': 'An entire reference explanation.',
           'cot_type': 'math', 'source_type': 'KbsdJames/Omni-MATH',
           'metadata': repr({'domain': ['Mathematics -> Algebra'], 'answer': r'\frac{1}{2}',
                             'messages': [{'content': 'secret trace'}]}), 'thinking_trajectories': ['trace']}
    item = adapt('s1', raw, 0, cfg['sources']['s1'], 'default')
    assert s1_math(item)
    assert item['gold_answer'] == r'\frac{1}{2}'
    assert 'trace' not in json.dumps(item)
    for source in ['Hothan/OlympiadBench/Open-ended/Physics', 'daman1209arora/jeebench/phy']:
        item['sub_source'] = source
        assert not s1_math(item)
    item['sub_source'] = 'TIGER-Lab/TheoremQA/float'
    item['metadata'] = {'cot_type': 'math'}
    assert not s1_math(item)


def test_filters():
    cfg = config()
    assert 'dataset_invalid' in rejection_reasons(row('numina', 'x', 'Find a value', problem_is_valid='Incomplete'), cfg)
    assert 'proof_heuristic' in rejection_reasons(row('math', 'x', 'Show that the result holds'), cfg)
    assert 'multiple_choice' in rejection_reasons(row('math', 'x', 'Choose (A) 3 (B) 4'), cfg)
    assert 'multipart_heuristic' in rejection_reasons(row('math', 'x', '(a) Find x. (b) Find y.'), cfg)


def test_pipeline_and_counts(tmp_path):
    text = 'Find the sum of all positive integers less than 100 satisfying these conditions'
    items = [row('math', 'm1', 'Calculate the value of the very special expression below'),
             row('math', 'm2', text), row('math', 'm3', 'Context before ' + text + ' after'),
             row('math', 'm4', 'Compute 7 plus 8'),
             row('math', 'm5', 'Completely separate RL question about a special geometric quantity'),
             row('math', 'm6', text.replace('100', '200'), '43'),
             row('numina', 'n1', 'Calculate the value of the very special expression below')]
    report = build_pool(items, [{'id': 'eval1', 'set': 'math_500', 'problem': text},
                               {'id': 'eval2', 'set': 'short', 'problem': 'Compute 7 plus 8'}],
                        [{'id': 'rl1', 'set': 'dapo', 'problem': items[4]['problem']}], config(), tmp_path)
    pool = list(iter_jsonl(tmp_path / 'pool.jsonl'))
    assert {x['id'] for x in pool} == {'m1', 'm6'}
    removed = list(iter_jsonl(tmp_path / 'removed.jsonl'))
    assert next(x for x in removed if x['id'] == 'n1')['duplicate_of'] == 'm1'
    assert any(x['id'] == 'm6' for x in iter_jsonl(tmp_path / 'eval_near_misses.jsonl'))
    assert sum(c['loaded'] for c in report['counts'].values()) == len(pool) + len(removed)
    # Deterministic bytes for every output file.
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    build_pool(items, [{'id': 'eval1', 'set': 'math_500', 'problem': text},
                       {'id': 'eval2', 'set': 'short', 'problem': 'Compute 7 plus 8'}],
               [{'id': 'rl1', 'set': 'dapo', 'problem': items[4]['problem']}], config(), tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_exact_conflict_and_equivalence(tmp_path):
    text = 'Evaluate this single expression'
    items = [row('math', 'a', text, '1'), row('numina', 'b', text, '2')]
    build_pool(items, [], [], config(), tmp_path)
    assert not list(iter_jsonl(tmp_path / 'pool.jsonl'))
    assert len(list(iter_jsonl(tmp_path / 'removed.jsonl'))) == 2
    items[0]['gold_answer'], items[1]['gold_answer'] = '0.5', r'\frac{1}{2}'
    build_pool(items, [], [], config(), tmp_path)
    assert [x['id'] for x in iter_jsonl(tmp_path / 'pool.jsonl')] == ['a']


def test_prefix_join_matches_bruteforce(tmp_path):
    cfg = config()
    cfg.update(dedup_ngram=1, dedup_jaccard=0.5)
    texts = [' '.join(x) for size in [2, 3, 4] for x in itertools.combinations(['apple', 'orange', 'pear', 'plum', 'lime'], size)]
    items = [row('math', str(i), text) for i, text in enumerate(texts)]
    build_pool(items, [], [], cfg, tmp_path)
    expected = list(range(len(items)))
    def root(i):
        while expected[i] != i:
            i = expected[i]
        return i
    for i, a in enumerate(texts):
        for j, b in enumerate(texts[:i]):
            x, y = set(a.split()), set(b.split())
            if len(x & y) / len(x | y) >= .5:
                expected[root(i)] = root(j)
    assert len(list(iter_jsonl(tmp_path / 'pool.jsonl'))) == len({root(i) for i in range(len(items))})


def test_real_math500_injection(tmp_path):
    path = ROOT / 'data/sources/math_500.jsonl'
    if not path.exists():
        pytest.skip('Local cached MATH-500 not present')
    original = next(iter_jsonl(path))
    problem = original['problem']
    index = ReferenceIndex([{'id': original['id'], 'set': 'math_500', 'problem': problem}], 8)
    assert max(x['coverage'] for x in index.matches(problem)) == 1
    assert max(x['coverage'] for x in index.matches('Additional context. ' + problem + ' End.')) == 1


def test_colab_embedded_code_and_cells(tmp_path):
    import base64
    import io
    import zipfile

    import nbformat

    path = ROOT / 'notebooks/build_clean_sft_pool.ipynb'
    if not path.exists():
        pytest.skip('Notebook container validation runs in the repository')
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook>', 'exec')
    boot = nb.cells[2].source
    import ast
    tree = ast.parse(boot)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))
    encoded = ast.literal_eval(assignment.value)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as bundle:
        for name in bundle.namelist():
            # The bundled test file includes this test, which safely skips checking
            # a notebook file when executed in the isolated Colab bundle.
            assert bundle.read(name) == (ROOT / name).read_bytes()
    assert 'userdata.get("HF_TOKEN")' in nb.cells[1].source
    assert 'Seungjun/clean-math-sft-pool' in nb.cells[3].source


def test_upload_rejects_changed_files_before_network(tmp_path):
    from common.io import write_json
    from pipeline.sft_pool_hub import sha256_file, upload

    pool = tmp_path / 'pool.jsonl'
    pool.write_text('{}\n')
    write_json(tmp_path / 'COMPLETE.json', {'rows': 1, 'files': {'pool.jsonl': sha256_file(pool)}})
    pool.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match='changed since completion'):
        upload(tmp_path, 'Seungjun/clean-math-sft-pool', token=None)
