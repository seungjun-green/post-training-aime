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
    assert "RUN_MODE = 'new'" in nb.cells[1].source
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
