import json
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import yaml
from test_transfer_eval import Tokenizer, tiny_sources

from common.io import read_jsonl, write_json, write_jsonl
from eval.export_transfer_results import FORMAT, build_export, publish_export
from eval.run_transfer_eval import run_model
from eval.run_transfer_runpod import exclusive
from eval.transfer_data import prepare_data
from eval.transfer_models import prepare_models


@pytest.fixture
def completed(tmp_path):
    config, loader = tiny_sources()
    docs, manifest = prepare_data(config, tmp_path, loader)
    run = prepare_models(config, tmp_path, tmp_path, tmp_path, ['base'])[0]

    def generate(engine, jobs, execution, prompt_renderer):
        for job in jobs:
            doc = next(d for d in docs if d['id'] == job['key'])
            text = 'HELLO' if doc['benchmark'] == 'ifeval' else 'The answer is ' + doc['answer'] + '.'
            yield job, prompt_renderer(engine.tokenizer, doc['prompt']), [
                {'text': text, 'token_count': 10, 'finish_reason': 'stop'}]

    run_model(run, config, docs, manifest, tmp_path, generator=generate,
              engine_factory=lambda *a: SimpleNamespace(tokenizer=Tokenizer()),
              metadata_factory=lambda: {'packages': {'test': '1'}, 'hardware': {}}, fingerprint=lambda _: 'weights')
    return config, tmp_path


def test_export_parquet_roundtrip_tables_card_and_deterministic_resume(completed):
    config, root = completed
    # Nearby secrets, models and logs must never enter the allowlisted export.
    (root / 'secret.env').write_text('DO_NOT_UPLOAD')
    folder = build_export(config, root, 'Seungjun/test-results', ['base'])
    marker = json.loads((folder / 'transfer_export.json').read_text())
    snapshot = folder / 'runs' / marker['run_id']
    rows = [row for p in sorted((snapshot / 'data/base').glob('*.parquet')) for row in pq.read_table(p).to_pylist()]
    assert len(rows) == 4
    assert len({r['example_id'] for r in rows}) == 4
    for row in rows:
        original = read_jsonl(root / 'models/base/full' / row['benchmark'] / 'generations.jsonl')[0]
        assert row['response'] == original['text']
        assert row['prompt'] == original['prompt']
        assert row['formatted_prompt'] == original['formatted_prompt']
        assert json.loads(row['score_json']) == original['score']
        if row['benchmark'] == 'ifeval':
            assert row['correct'] is None
            assert json.loads(row['document_json'])['kwargs'] == original['kwargs']
        else:
            assert row['correct'] is True
    readme = (folder / 'README.md').read_text()
    metadata = yaml.safe_load(readme.split('---', 2)[1])
    assert metadata['configs'][0]['data_files'][0]['path'] == f'runs/{marker["run_id"]}/data/base/*.parquet'
    for heading in ['Overall results', 'ARC-Challenge', 'MMLU-Pro subjects', 'Non-math BBH', 'IFEval']:
        assert heading in readme
    assert '| 100 |' in readme and '4 inference records' in readme.replace('*', '')
    assert (snapshot / 'tables/comparison.csv').read_text() == (root / 'comparison.csv').read_text()
    assert build_export(config, root, 'Seungjun/test-results', ['base']) == folder
    assert not any('secret' in p.name or 'console' in p.name for p in folder.rglob('*'))


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'score', 'document', 'response', 'smoke'])
def test_export_rejects_incomplete_or_changed_results(completed, change):
    config, root = completed
    path = root / 'models/base/full/arc_challenge/generations.jsonl'
    rows = read_jsonl(path)
    if change == 'missing':
        (root / 'models/base/full/results.json').unlink()
    elif change == 'duplicate':
        write_jsonl(path, rows + rows)
    elif change == 'score':
        rows[0]['score']['correct'] = False
        write_jsonl(path, rows)
    elif change == 'document':
        rows[0]['prompt'] = 'Changed prompt'
        write_jsonl(path, rows)
    elif change == 'response':
        rows[0]['text'] = 'Changed response'
        write_jsonl(path, rows)
    else:
        result = json.loads((root / 'models/base/full/results.json').read_text())
        result['identity']['mode'] = 'smoke'
        write_json(root / 'models/base/full/results.json', result)
    with pytest.raises(ValueError):
        build_export(config, root, 'Seungjun/test-results', ['base'])
    assert not list((root / 'hf_exports').glob('*/README.md'))


def test_export_blocks_active_coordinator_and_missing_selected_models(completed):
    config, root = completed
    with exclusive(root / 'runpod_coordinator.lock'):
        with pytest.raises(RuntimeError, match='Another evaluation process'):
            build_export(config, root, 'Seungjun/test-results', ['base'])
    with pytest.raises(ValueError, match='must finish'):
        build_export(config, root, 'Seungjun/test-results')


class FakeHub:
    def __init__(self, root, files=None, private=True):
        self.root, self.files, self.private = root, files or [], private
        self.calls = []

    def create_repo(self, **kwargs):
        self.calls.append(('create_repo', kwargs))

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=self.private, sha='parent-sha')

    def list_repo_files(self, **kwargs):
        return self.files

    def hf_hub_download(self, **kwargs):
        return str(self.root / kwargs['filename'])

    def create_commit(self, **kwargs):
        self.calls.append(('commit', kwargs))
        return SimpleNamespace(commit_url='https://huggingface.co/datasets/Seungjun/test-results/commit/test')


def test_upload_only_allowlisted_files_in_one_dataset_commit(completed):
    config, root = completed
    folder = build_export(config, root, 'Seungjun/test-results', ['base'])
    (folder / '.env').write_text('token=SECRET')
    (folder / 'console.log').write_text('do not upload')
    api = FakeHub(folder, files=['.gitattributes'])
    publish_export(folder, 'Seungjun/test-results', private=True, token='fake-write-token', api=api)
    assert api.calls[0] == ('create_repo', {'repo_id': 'Seungjun/test-results', 'repo_type': 'dataset',
                                         'private': True, 'exist_ok': True})
    commit = api.calls[1][1]
    assert commit['repo_type'] == 'dataset' and commit['parent_commit'] == 'parent-sha'
    names = [op.path_in_repo for op in commit['operations']]
    assert 'README.md' in names and 'transfer_export.json' in names
    assert len([name for name in names if name.endswith('.parquet')]) == 4
    assert not {'.env', 'console.log'} & set(names)
    assert 'delete_patterns' not in commit
    # Re-upload is permitted for a repo created by this exporter, without deleting older runs.
    api = FakeHub(folder, files=['.gitattributes', 'transfer_export.json', 'runs/older/results.json'])
    publish_export(folder, 'Seungjun/test-results', token='fake', api=api)
    assert len(api.calls) == 2


@pytest.mark.parametrize('kind', ['unrelated', 'visibility', 'token', 'changed_file', 'wrong_format'])
def test_upload_rejects_unsafe_or_invalid_destination_and_payload(completed, kind):
    config, root = completed
    folder = build_export(config, root, 'Seungjun/test-results', ['base'])
    files = ['README.md'] if kind == 'unrelated' else []
    api = FakeHub(folder, files=files, private=kind != 'visibility')
    token = None if kind == 'token' else 'fake'
    if kind == 'changed_file':
        next(folder.rglob('comparison.csv')).write_text('modified')
    if kind == 'wrong_format':
        remote = root / 'remote'
        write_json(remote / 'transfer_export.json', {'format': FORMAT + '-other'})
        api = FakeHub(remote, files=['transfer_export.json'])
    with pytest.raises(ValueError):
        publish_export(folder, 'Seungjun/test-results', token=token, api=api)
    assert not any(name == 'commit' for name, _ in api.calls)
