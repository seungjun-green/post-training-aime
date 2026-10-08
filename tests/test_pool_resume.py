import ast
import base64
import io
import json
from pathlib import Path
import zipfile

import nbformat
import pytest

from common.pool_resume import resolve_run_config, show_checkpoint_status

ROOT = Path(__file__).resolve().parents[1]


def saved_run(path):
    path.mkdir(exist_ok=True)
    manifest = {'config': {'sampling': {'seed': 42, 'max_new_tokens': 8192}},
                'dataset_revision': 'pinned', 'quality': {'eligible_rows': 3}}
    (path / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_auto_restores_saved_settings_and_never_mutates_manifest(tmp_path, capsys):
    manifest = saved_run(tmp_path)
    before = (tmp_path / 'manifest.json').read_bytes()
    cfg = resolve_run_config({'sampling': {'seed': 99}}, tmp_path)
    assert cfg == manifest['config']
    assert (tmp_path / 'manifest.json').read_bytes() == before
    assert 'Resuming saved run' in capsys.readouterr().out


def test_resume_requires_existing_and_new_never_overwrites(tmp_path):
    saved_run(tmp_path / 'original-run')
    with pytest.raises(ValueError, match='original-run'):
        resolve_run_config({}, tmp_path / 'misspelled-run', 'resume')
    with pytest.raises(ValueError, match='empty run directory'):
        resolve_run_config({}, tmp_path / 'original-run', 'new')
    assert resolve_run_config({'new': True}, tmp_path / 'fresh', 'auto') == {'new': True}
    assert not (tmp_path / 'fresh').exists()


def test_orphan_journal_cannot_be_reinterpreted_as_new_run(tmp_path):
    (tmp_path / 'responses.jsonl').write_text('{"source_index":0}\n')
    with pytest.raises(ValueError, match='no manifest'):
        resolve_run_config({}, tmp_path)


def test_status_counts_out_of_order_and_leaves_interrupted_tail_untouched(tmp_path):
    saved_run(tmp_path)
    path = tmp_path / 'responses.jsonl'
    path.write_bytes(b'{"source_index":2}\n{"source_index":0}\n{"source_index":')
    before = path.read_bytes()
    assert show_checkpoint_status(tmp_path) == {'saved': 2, 'total': 3, 'remaining': 1}
    assert path.read_bytes() == before


def test_status_rejects_duplicate_and_interior_corruption(tmp_path):
    saved_run(tmp_path)
    path = tmp_path / 'responses.jsonl'
    path.write_text('{"source_index":0}\n{"source_index":0}\n')
    with pytest.raises(ValueError, match='Duplicate'):
        show_checkpoint_status(tmp_path)
    path.write_text('broken\n{"source_index":0}\n')
    with pytest.raises(ValueError, match='Corrupt'):
        show_checkpoint_status(tmp_path)


def test_colab_resume_notebook_is_self_contained():
    nb = nbformat.read(ROOT / 'notebooks/full_run_clean_sft_pool_qwen3b_8.ipynb', as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook>', 'exec')
    assert "RUN_MODE = 'auto'" in nb.cells[1].source
    prepare = nb.cells[5].source
    assert prepare.index('resolve_run_config') < prepare.index('CONFIG_PATH.write_text')
    assert 'show_checkpoint_status' in prepare
    tree = ast.parse(nb.cells[3].source)
    encoded = next(n.args[0].value for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as z:
        assert 'common/pool_resume.py' in z.namelist()
        for name in z.namelist():
            assert z.read(name) == (ROOT / name).read_bytes()


@pytest.fixture
def recovery_run(tmp_path, monkeypatch):
    import yaml
    from pipeline import sample_sft_pool as sampling
    rows = [{'id': f'row-{i}'} for i in range(3)]
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump({'data': {'id_column': 'id'}}))
    monkeypatch.setattr(sampling, 'prepare', lambda *args: (rows, {}))
    def entry(i):
        return {'source_index': i, 'id': f'row-{i}', 'annotation': {
            'responses': ['answer'] * 8, 'extracted_answers': ['42'] * 8,
            'correct': [True] * 8, 'response_tokens': [10] * 8,
            'finish_reasons': ['stop'] * 8, 'num_correct': 8}}
    return config_path, tmp_path / 'responses.jsonl', entry


@pytest.mark.parametrize('broken', [b'broken\n', b'{"source_index":\n', b'\x00\x00\xff\n'])
def test_recovery_keeps_valid_rows_after_corruption_and_full_backup(recovery_run, broken):
    from common.pool_resume import recover_checkpoint
    config_path, path, entry = recovery_run
    data = json.dumps(entry(2)).encode() + b'\n' + broken + json.dumps(entry(0)).encode()
    path.write_bytes(data)
    report = recover_checkpoint(config_path, path.parent)
    assert Path(report['backup']).read_bytes() == data
    assert report['unreadable_records'] == 1
    assert report['saved_problems'] == 2 and report['remaining_problems'] == 1
    kept = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert kept == [entry(2), entry(0)]
    assert path.read_bytes().endswith(b'\n')
    assert recover_checkpoint(config_path, path.parent) is None


def test_recovery_does_not_hide_source_mismatch(recovery_run):
    from common.pool_resume import recover_checkpoint
    config_path, path, entry = recovery_run
    wrong = entry(1)
    wrong['id'] = 'wrong-source'
    data = b'broken\n' + json.dumps(wrong).encode() + b'\n'
    path.write_bytes(data)
    with pytest.raises(ValueError, match='ID mismatch'):
        recover_checkpoint(config_path, path.parent)
    assert path.read_bytes() == data
    assert not list(path.parent.glob('*.tmp'))


def test_recovery_stops_if_live_writer_changes_journal(recovery_run, monkeypatch):
    from common.pool_resume import recover_checkpoint
    from pipeline import sample_sft_pool as sampling
    config_path, path, entry = recovery_run
    path.write_bytes(b'broken\n' + json.dumps(entry(0)).encode() + b'\n')
    original = sampling.prepare
    def concurrently_append(*args):
        with path.open('ab') as f:
            f.write(json.dumps(entry(1)).encode() + b'\n')
        return original(*args)
    monkeypatch.setattr(sampling, 'prepare', concurrently_append)
    with pytest.raises(RuntimeError, match='changed during validation'):
        recover_checkpoint(config_path, path.parent)
    assert path.read_bytes().startswith(b'broken\n')
    assert json.loads(path.read_bytes().splitlines()[-1]) == entry(1)


def test_generation_entry_recovers_before_scanning(recovery_run, monkeypatch):
    from common.pool_resume import generate_with_recovery
    from common import pool_progress
    from pipeline import sample_sft_pool as sampling
    config_path, path, entry = recovery_run
    original = json.dumps(entry(0)).encode() + b'\nbroken\n' + json.dumps(entry(2)).encode() + b'\n'
    path.write_bytes(original)
    monkeypatch.setattr(pool_progress, 'check_resume', lambda *args: None)
    def generate(cfg, root, run_dir, token, mode):
        index = sampling.scan_journal(path, [{'id': f'row-{i}'} for i in range(3)], cfg['data'])
        assert set(index) == {0, 2}
        assert mode == 'full'
        assert next((path.parent / 'checkpoint_backups').glob('responses-*.jsonl')).read_bytes() == original
        return 'generation reached'
    monkeypatch.setattr(sampling, 'generate', generate)
    assert generate_with_recovery(config_path, path.parent) == 'generation reached'


def test_generation_cell_cannot_bypass_recovery():
    nb = nbformat.read(ROOT / 'notebooks/full_run_clean_sft_pool_qwen3b_8.ipynb', as_version=4)
    assert "'common.pool_resume', '--generate'" in nb.cells[7].source
    assert 'pool_progress.run_pool_logged' in nb.cells[7].source
    assert "RUN_NAME = 'qwen25-3b-fresh-run-002'" in nb.cells[1].source
    assert "RUN_MODE = 'auto'" in nb.cells[1].source
