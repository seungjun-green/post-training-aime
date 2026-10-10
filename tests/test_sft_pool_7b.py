import ast
import base64
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from datasets import Dataset

from common.io import append_jsonl, write_json
from pipeline import sample_sft_pool_7b as seven

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = yaml.safe_load((ROOT / 'configs/sft_pool_sampling_7b.yaml').read_text())
    cfg['export']['shard_rows'] = 3
    return cfg


def source(scores=range(9)):
    return Dataset.from_list([{
        'id': f'row-{i}', 'problem': 'Compute 6 times 7.', 'gold_answer': '42',
        'answer_type': 'integer', 'responses': ['3B original response'] * 8,
        'extracted_answers': ['42' if j < score else '0' for j in range(8)],
        'correct': [j < score for j in range(8)], 'response_tokens': list(range(8)),
        'finish_reasons': ['stop'] * 8, 'num_correct': score,
        'metadata': {'source': 'original'}, 'nullable': None,
    } for i, score in enumerate(scores)])


def outputs():
    return [{'text': r'\boxed{42}' if i < 5 else r'\boxed{0}',
             'token_count': 10 + i, 'finish_reason': 'length' if i == 7 else 'stop'} for i in range(8)]


def manifest(ds):
    return {'signature': 'test', 'dataset_revision': 'original-revision',
            'selected_indices': seven.select_rows(ds), 'source_rows': len(ds)}


def complete_run(tmp_path, ds):
    m = manifest(ds)
    annotation, errors = seven.annotate(outputs(), '42')
    for i in reversed(m['selected_indices']):
        append_jsonl(tmp_path / 'responses_7b.jsonl', {
            'source_index': i, 'id': ds[i]['id'], 'annotation': annotation,
            'grading_errors': errors, 'grading_audit': []})
    out = seven.export(ds, m, config(), tmp_path)
    return m, out


def test_exact_selection_and_invalid_source():
    assert seven.select_rows(source()) == [0, 1, 2]
    assert seven.select_rows(source([8, 2, 4, 0, 1])) == [1, 3, 4]
    with pytest.raises(ValueError, match='already has 7B'):
        seven.select_rows(source().add_column('7B_num_correct', [None] * 9))
    with pytest.raises(ValueError, match='Invalid original'):
        seven.select_rows(source([-1]))
    with pytest.raises(ValueError, match='Missing 3B'):
        seven.select_rows(source().remove_columns('responses'))


def test_export_preserves_all_originals_and_uses_null_for_unselected(tmp_path):
    ds = source()
    m, out = complete_run(tmp_path, ds)
    seven.validate_export(ds, m, out)
    rows = [row for f in sorted((out / 'data').glob('*.parquet')) for row in pq.read_table(f).to_pylist()]
    assert len(rows) == len(ds)
    for i, row in enumerate(rows):
        assert {k: row[k] for k in ds.column_names} == ds[i]
        if i < 3:
            assert row['7B_num_correct'] == 5
            assert row['7B_finish_reasons'][-1] == 'length'
            assert all(len(row[k]) == 8 for k in seven.NEW_COLUMNS[:-1])
        else:
            assert all(row[k] is None for k in seven.NEW_COLUMNS)
    assert json.loads((out / 'summary.json').read_text())['responses'] == 24


def test_no_matching_rows_and_incomplete_generation(tmp_path):
    ds = source([3, 8])
    m, out = complete_run(tmp_path, ds)
    seven.validate_export(ds, m, out)
    assert json.loads((out / 'summary.json').read_text())['responses'] == 0
    with pytest.raises(ValueError, match='Incomplete'):
        seven.export(source(), manifest(source()), config(), tmp_path)


def test_reject_journal_rows_outside_selection(tmp_path):
    ds = source()
    annotation, errors = seven.annotate(outputs(), '42')
    append_jsonl(tmp_path / 'responses_7b.jsonl', {
        'source_index': 8, 'id': 'row-8', 'annotation': annotation, 'grading_errors': errors})
    with pytest.raises(ValueError, match='outside'):
        seven.journal_index(tmp_path, ds, [0, 1, 2])


def test_upload_validation_catches_changed_3b_even_with_updated_checksum(tmp_path):
    ds = source()
    m, out = complete_run(tmp_path, ds)
    p = out / 'data/train-00000.parquet'
    table = pq.read_table(p)
    rows = table.to_pylist()
    rows[0]['responses'][0] = 'modified original'
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), p)
    complete = json.loads((out / 'COMPLETE.json').read_text())
    complete['files']['data/train-00000.parquet'] = seven.hash_file(p)
    write_json(out / 'COMPLETE.json', complete)
    with pytest.raises(ValueError, match='Original row/3B'):
        seven.validate_export(ds, m, out)


def test_generation_selection_and_resume(tmp_path, monkeypatch):
    ds = source([0, 3, 1, 8, 2])
    m = manifest(ds)
    monkeypatch.setattr(seven, 'prepare', lambda *args: (ds, m))
    tokenizer = SimpleNamespace(chat_template='native', encode=lambda *args, **kw: [1, 2],
                                apply_chat_template=lambda messages, **kw: messages[1]['content'])
    monkeypatch.setattr(seven, 'make_engine', lambda *args: SimpleNamespace(tokenizer=tokenizer))
    generated = []
    def fake_generate(engine, jobs, execution, *, prompt_renderer):
        assert prompt_renderer is seven.render_prompt
        for job in reversed(list(jobs)):
            generated.append(job['source_index'])
            assert job['n'] == 8
            yield job, 'prompt', outputs()
    import eval.batched_generation
    monkeypatch.setattr(eval.batched_generation, 'generate_problems', fake_generate)
    seven.generate(config(), ROOT, tmp_path, None)
    assert set(generated) == {0, 2, 4}
    with (tmp_path / 'responses_7b.jsonl').open('ab') as f:
        f.write(b'{"interrupted')
    seven.generate(config(), ROOT, tmp_path, None)
    assert len(generated) == 3
    seven.validate_export(ds, m, tmp_path / 'full')


def test_generation_passes_other_answer_type_to_the_existing_grader(tmp_path, monkeypatch):
    row = source([0])[0]
    row.update(problem='What time does the clock show?', gold_answer='12:00', answer_type='other')
    ds = Dataset.from_list([row])
    monkeypatch.setattr(seven, 'prepare', lambda *args: (ds, manifest(ds)))
    tokenizer = SimpleNamespace(chat_template='native', encode=lambda *args, **kw: [1, 2],
                                apply_chat_template=lambda messages, **kw: 'clock prompt')
    monkeypatch.setattr(seven, 'make_engine', lambda *args: SimpleNamespace(tokenizer=tokenizer))
    def fake_generate(engine, jobs, execution, **kwargs):
        for job in jobs:
            yield job, 'clock prompt', [{**r, 'text': r'\boxed{12:00}'} for r in outputs()]
    import eval.batched_generation
    monkeypatch.setattr(eval.batched_generation, 'generate_problems', fake_generate)
    seven.generate(config(), ROOT, tmp_path, None)
    result = pq.read_table(tmp_path / 'full/data/train-00000.parquet').to_pylist()[0]
    assert result['7B_correct'] == [True] * 8 and result['7B_num_correct'] == 8
    assert result['num_correct'] == 0


def test_instruct_prompt_stops_and_native_context_budget():
    captured = []
    tokenizer = SimpleNamespace(chat_template='native', eos_token_id=151645,
        apply_chat_template=lambda messages, **kw: captured.append((messages, kw)) or 'formatted',
        encode=lambda token, **kw: [{'<|im_end|>': 151645, '<|endoftext|>': 151643}[token]])
    assert seven.render_prompt(tokenizer, 'Question') == 'formatted'
    assert captured[0][0] == [{'role': 'system', 'content': seven.SYSTEM_PROMPT},
                              {'role': 'user', 'content': 'Question'}]
    assert captured[0][1]['add_generation_prompt'] is True
    assert seven.stop_ids(tokenizer) == [151645, 151643]
    tokenizer.eos_token_id = 151643
    with pytest.raises(ValueError, match='EOS'):
        seven.stop_ids(tokenizer)
    assert seven.completion_budget(100, config()) == 3072
    assert seven.completion_budget(2000, config()) == 2096
    with pytest.raises(ValueError, match='context'):
        seven.completion_budget(4096, config())


def test_prepare_pins_source_and_rejects_setting_changes(tmp_path, monkeypatch):
    import datasets
    import huggingface_hub
    calls = []
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **kw: SimpleNamespace(
        dataset_info=lambda *args, **kw: calls.append('resolve') or SimpleNamespace(sha='first-commit')))
    def load(*args, **kwargs):
        assert kwargs['revision'] == 'first-commit'
        return source()
    monkeypatch.setattr(datasets, 'load_dataset', load)
    _, m = seven.prepare(config(), ROOT, tmp_path, None)
    _, resumed = seven.prepare(config(), ROOT, tmp_path, None)
    assert resumed == m and calls == ['resolve']
    cfg = config()
    cfg['sampling']['seed'] += 1
    with pytest.raises(ValueError, match='Code/settings'):
        seven.prepare(cfg, ROOT, tmp_path, None)


def test_atomic_upload_guard_verification_and_lost_receipt_recovery(tmp_path, monkeypatch):
    import huggingface_hub
    ds = source()
    m, out = complete_run(tmp_path, ds)
    monkeypatch.setattr(seven, 'prepare', lambda *args: (ds, m))
    remote = tmp_path / 'remote'
    remote.mkdir()
    old_card = tmp_path / 'original.md'
    old_card.write_text('---\nlicense: mit\ndataset_info:\n  features: []\n---\nOriginal 3B dataset.\n')
    state = {'sha': 'unrelated-commit', 'commits': 0}
    class Api:
        def __init__(self, **kwargs): pass
        def dataset_info(self, repo): return SimpleNamespace(sha=state['sha'])
        def create_commit(self, repo, **kwargs):
            assert kwargs['parent_commit'] == 'original-revision'
            state['commits'] += 1
            for op in kwargs['operations']:
                path = remote / op.path_in_repo
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(Path(op.path_or_fileobj).read_bytes())
            state['sha'] = 'new-commit'
            return SimpleNamespace(oid=state['sha'])
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    def download(repo, name, **kwargs):
        path = old_card if kwargs['revision'] == 'original-revision' and name == 'README.md' else remote / name
        if not path.exists(): raise FileNotFoundError(name)
        return str(path)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    with pytest.raises(ValueError, match='HF changed'):
        seven.upload(config(), ROOT, tmp_path, None)
    assert state['commits'] == 0
    state['sha'] = 'original-revision'
    assert seven.upload(config(), ROOT, tmp_path, None)['verified']
    assert state['commits'] == 1
    card = (remote / 'README.md').read_text()
    assert 'Original 3B dataset.' in card and 'license: mit' in card
    assert 'annotated-7b/test/data/*.parquet' in card and 'dataset_info' not in card
    (tmp_path / 'upload_receipt.json').unlink()
    assert seven.upload(config(), ROOT, tmp_path, None)['verified']
    assert state['commits'] == 1


def test_notebook_embeds_current_runtime_and_full_upload_flow():
    nb = nbformat.read(ROOT / 'notebooks/sample_sft_pool_math7b_8.ipynb', as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code': compile(cell.source, '<notebook>', 'exec')
    code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    assert 'userdata.get("HF_TOKEN")' in code and 'UPLOAD_TO_HF = True' in code
    assert "['--mode', 'full']" in code and "['--mode', 'upload']" in code
    tree = ast.parse(code)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(call.args[0])))) as archive:
        for name in archive.namelist(): assert archive.read(name) == (ROOT / name).read_bytes()
        assert 'pipeline/sample_sft_pool_7b.py' in archive.namelist()
