import ast
import base64
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pyarrow.parquet as pq
import pytest
import yaml
from datasets import Dataset, load_dataset

from common.io import append_jsonl, write_json
from pipeline import sample_sft_pool as sampling

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / 'configs/sft_pool_sampling.yaml').read_text())


def outputs():
    return [{'text': r'Work. \boxed{42}' if i % 2 == 0 else r'Work. \boxed{0}',
             'token_count': i + 5, 'finish_reason': 'length' if i == 7 else 'stop'} for i in range(8)]


def fixture_data(count=60):
    return Dataset.from_list([{'id': f'id-{i}', 'problem': f'Compute a quantity for {i}.',
                               'gold_answer': '42', 'domain': 'algebra', 'nullable': None,
                               'metadata': {'level': i}, 'synthetic': False} for i in range(count)])


def runtime_files(path):
    write_json(path / 'runtime.json', {'gpu': 'test-only fake engine'})
    write_json(path / 'prompt.json', {'prompt': 'test'})


def test_annotation_types_grading_and_missing_boxes():
    result, errors = sampling.annotate(outputs(), '42')
    assert result['num_correct'] == 4
    assert result['correct'] == [True, False] * 4
    assert result['finish_reasons'][-1] == 'length'
    assert result['response_tokens'] == list(range(5, 13))
    assert errors == [None] * 8
    generated = outputs()
    generated[0]['text'] = 'Answer is 42 without a box'
    result, _ = sampling.annotate(generated, '42')
    assert result['extracted_answers'][0] == '' and result['correct'][0] is False
    generated[1]['text'] = r'\boxed{41} then correction \boxed{42}'
    result, _ = sampling.annotate(generated, '42')
    assert result['extracted_answers'][1] == '42' and result['correct'][1]


def test_math_verify_equivalence_and_grade_exception(monkeypatch):
    generated = outputs()
    generated[0]['text'] = r'\boxed{\frac{1}{2}}'
    result, _ = sampling.annotate(generated, '0.5')
    assert result['correct'][0]
    def fail(*args, **kwargs):
        raise TimeoutError('test')
    monkeypatch.setattr(sampling, 'score_final', fail)
    result, errors = sampling.annotate(outputs(), '42')
    assert result['num_correct'] == 0 and errors == ['TimeoutError'] * 8


def test_rejects_incomplete_or_bad_vectors():
    with pytest.raises(ValueError):
        sampling.annotate(outputs()[:7], '42')
    bad = outputs()
    bad[0]['finish_reason'] = 'abort'
    with pytest.raises(ValueError):
        sampling.annotate(bad, '42')
    row, _ = sampling.annotate(outputs(), '42')
    row['num_correct'] = 8
    with pytest.raises(ValueError):
        sampling.validate_annotation(row)


def test_random_selection_and_seed_stability():
    a = sampling.select_smoke(28905, 42)
    assert len(a) == len(set(a)) == 50
    assert a == sampling.select_smoke(28905, 42)
    assert a != sampling.select_smoke(28905, 43)
    assert sampling.sample_seed(42, 'a') == sampling.sample_seed(42, 'a')
    assert sampling.sample_seed(42, 'a') != sampling.sample_seed(42, 'b')


def test_checkpoint_repair_and_corruption(tmp_path):
    ds, cfg = fixture_data(), config()
    annotation, errors = sampling.annotate(outputs(), '42')
    record = {'source_index': 0, 'id': 'id-0', 'annotation': annotation, 'grading_errors': errors}
    path = tmp_path / 'responses.jsonl'
    append_jsonl(path, record)
    valid = path.read_bytes()
    with path.open('ab') as f:
        f.write(b'{"incomplete')
    assert sampling.scan_journal(path, ds, cfg['data']) == {0: 0}
    assert path.read_bytes() == valid
    append_jsonl(path, record)
    with pytest.raises(ValueError, match='Duplicate'):
        sampling.scan_journal(path, ds, cfg['data'])
    path.write_bytes(valid + b'bad\n')
    with pytest.raises(ValueError, match='Corrupt'):
        sampling.scan_journal(path, ds, cfg['data'])


@pytest.fixture
def generated_run(tmp_path, monkeypatch):
    ds, cfg = fixture_data(), config()
    cfg['data']['expected_rows'] = len(ds)
    cfg['export']['shard_rows'] = 17
    manifest = {'signature': 'test-signature', 'dataset_revision': 'old-revision',
                'smoke_indices': sampling.select_smoke(len(ds), 42), 'config': cfg}
    runtime_files(tmp_path)
    monkeypatch.setattr(sampling, 'prepare', lambda *args: (ds, manifest))
    generated = []
    class Tokenizer:
        chat_template = 'test'
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]['content']
        def encode(self, prompt, **kwargs):
            return list(range(len(prompt.split())))
    engine = SimpleNamespace(tokenizer=Tokenizer(), config={**cfg['sampling'], **cfg['engine']})
    monkeypatch.setattr(sampling, 'make_engine', lambda *args: engine)
    def fake_generate(engine, jobs, execution):
        for job in jobs:
            generated.append(job['source_index'])
            yield job, 'prompt', outputs()
    import eval.batched_generation
    monkeypatch.setattr(eval.batched_generation, 'generate_problems', fake_generate)
    sampling.generate(cfg, ROOT, tmp_path, None, 'smoke')
    assert len(generated) == 50
    sampling.generate(cfg, ROOT, tmp_path, None, 'smoke')
    assert len(generated) == 50  # Resuming does not generate twice.
    sampling.generate(cfg, ROOT, tmp_path, None, 'full')
    assert len(generated) == len(ds) and len(set(generated)) == len(ds)
    return ds, cfg, manifest, tmp_path


def test_smoke_full_resume_export_and_originals(generated_run):
    ds, cfg, manifest, path = generated_run
    smoke = [json.loads(line) for line in (path / 'smoke/smoke_results.jsonl').read_text().splitlines()]
    assert [r['id'] for r in smoke] == [ds[i]['id'] for i in manifest['smoke_indices']]
    files = sorted((path / 'full/data').glob('*.parquet'))
    merged = [row for file in files for row in pq.read_table(file).to_pylist()]
    assert len(merged) == len(ds)
    for i, row in enumerate(merged):
        assert {k: row[k] for k in ds.column_names} == ds[i]
        assert set(row) == set(ds.column_names) | set(sampling.COLUMNS)
        sampling.validate_annotation(row)
    loaded = load_dataset('parquet', data_files=[str(p) for p in files], split='train', cache_dir=str(path / 'hf-cache'))
    assert loaded[0]['responses'] == merged[0]['responses']
    summary = json.loads((path / 'full/summary.json').read_text())
    assert summary['responses'] == len(ds)*8
    assert summary['num_correct_histogram']['4'] == len(ds)
    assert summary['truncated_fraction'] == 1/8


def test_incomplete_export_is_blocked(tmp_path):
    with pytest.raises(ValueError, match='Incomplete generation'):
        sampling.export(fixture_data(), config(), {}, tmp_path, [0], 'full')


def test_upload_atomicity_preservation_guard_and_retry(generated_run, monkeypatch):
    import huggingface_hub
    ds, cfg, manifest, path = generated_run
    remote = path / 'mock_remote'
    remote.mkdir()
    source_card = remote / 'source_README.md'
    source_card.write_text('---\nlicense: mit\ndataset_info:\n  features: []\n---\nOriginal dataset documentation.\n')
    state = {'sha': 'old-revision', 'commits': 0}
    class Api:
        def __init__(self, **kwargs):
            pass
        def dataset_info(self, repo):
            return SimpleNamespace(sha=state['sha'])
        def create_commit(self, repo, **kwargs):
            assert kwargs['parent_commit'] == state['sha']
            state['commits'] += 1
            for operation in kwargs['operations']:
                destination = remote / operation.path_in_repo
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(Path(operation.path_or_fileobj).read_bytes())
            state['sha'] = 'uploaded-revision'
            return SimpleNamespace(oid=state['sha'])
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    def download(repo, name, **kwargs):
        return str(source_card if name == 'README.md' and kwargs['revision'] == 'old-revision' else remote / name)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    state['sha'] = 'someone-elses-commit'
    with pytest.raises(ValueError, match='changed since preparation'):
        sampling.upload(cfg, ROOT, path, None)
    assert state['commits'] == 0
    state['sha'] = 'old-revision'
    receipt = sampling.upload(cfg, ROOT, path, None)
    assert receipt['verified'] and state['commits'] == 1
    card = (remote / 'README.md').read_text()
    assert 'Original dataset documentation.' in card and 'license: mit' in card
    assert 'annotated/data/*.parquet' in card
    assert 'dataset_info' not in card  # Stale feature schema replaced by Parquet inference.
    receipt['verified'] = False  # Resume interrupted post-commit verification.
    write_json(path / 'upload_receipt.json', receipt)
    assert sampling.upload(cfg, ROOT, path, None)['verified']
    assert state['commits'] == 1


def test_notebook_smoke_gates_and_embedded_files():
    nb = nbformat.read(ROOT / 'notebooks/sample_clean_sft_pool_qwen3b_8.ipynb', as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook>', 'exec')
    settings = nb.cells[1].source
    assert 'RUN_FULL = False' in settings and 'UPLOAD_FULL = False' in settings
    assert 'userdata.get("HF_TOKEN")' in nb.cells[3].source
    assert any("['--mode', 'smoke']" in c.source for c in nb.cells if c.cell_type == "code")
    tree = ast.parse(nb.cells[3].source)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    bundle = zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(call.args[0]))))
    for name in bundle.namelist():
        assert bundle.read(name) == (ROOT / name).read_bytes()


def test_qwen_base_eos_and_chat_end_stop_ids():
    class Tokenizer:
        eos_token_id = 151643
        def encode(self, token, **kwargs):
            return {'<|endoftext|>': [151643], '<|im_end|>': [151645]}[token]
    assert sampling.qwen_stop_ids(Tokenizer()) == [151643, 151645]
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = 151645
    with pytest.raises(ValueError, match='Unexpected EOS'):
        sampling.qwen_stop_ids(tokenizer)


def test_latest_cleaned_source_pins_once_and_discovers_rows(tmp_path, monkeypatch):
    import huggingface_hub
    cfg = config()
    cfg['data'].update(revision='main', expected_rows=None, require_text_cleanup=True)
    source = fixture_data(55)
    loaded = []
    checks = []
    head = ['a' * 40]
    class Api:
        def __init__(self, **kwargs): pass
        def dataset_info(self, *args, **kwargs):
            assert kwargs['revision'] == 'main'
            return SimpleNamespace(sha=head[0])
        def model_info(self, *args, **kwargs):
            return SimpleNamespace(sha=cfg['model']['revision'])
    def load(cfg, revision, token):
        loaded.append(revision)
        return source
    def cleanup(spec, revision, token):
        checks.append(revision)
        return {'retained_rows':55}
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    monkeypatch.setattr(sampling, 'load_source', load)
    monkeypatch.setattr(sampling, 'cleaned_source_report', cleanup)
    _, first = sampling.prepare(cfg, ROOT, tmp_path, None)
    assert first['source_rows'] == 55 and first['dataset_revision'] == 'a'*40
    head[0] = 'b'*40
    _, again = sampling.prepare(cfg, ROOT, tmp_path, None)
    assert first == again
    assert loaded == checks == ['a'*40, 'a'*40]
    monkeypatch.setattr(sampling, 'cleaned_source_report', lambda *args: {'retained_rows':54})
    with pytest.raises(ValueError, match='Cleanup report row count'):
        sampling.prepare(cfg, ROOT, tmp_path, None)


def test_load_source_dynamic_or_explicit_count(monkeypatch):
    import datasets
    cfg = config()
    monkeypatch.setattr(datasets, 'load_dataset', lambda *args, **kwargs: fixture_data(55))
    cfg['data']['expected_rows'] = None
    assert len(sampling.load_source(cfg, 'a'*40, None)) == 55
    cfg['data']['expected_rows'] = 56
    with pytest.raises(ValueError, match='Expected 56 rows'):
        sampling.load_source(cfg, 'a'*40, None)


def test_cleanup_publication_required_before_smoke(tmp_path, monkeypatch):
    import huggingface_hub
    card = tmp_path/'README.md'
    report = tmp_path/'summary.json'
    cfg = config()
    requested = []
    def download(repo, filename, **kwargs):
        requested.append((filename, kwargs['revision']))
        return str(card if filename == 'README.md' else report)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    card.write_text('---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/train.parquet\n---\nOriginal dataset')
    with pytest.raises(ValueError, match='Finish the cleanup-and-upload notebook'):
        sampling.cleaned_source_report(cfg['data'], 'a'*40, None)
    card.write_text(card.read_text().replace('data/train.parquet', 'cleaning/text-only/run/train-cleaned.parquet'))
    data = {'policy_version':'text-only-exclusions-v1', 'repo_id':cfg['data']['repo'],
            'source_config':'default','retained_rows':55}
    report.write_text(json.dumps(data))
    assert sampling.cleaned_source_report(cfg['data'], 'a'*40, None) == data
    assert requested[-1] == ('cleaning/text-only/run/summary.json', 'a'*40)
