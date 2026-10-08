import sys
from types import SimpleNamespace

import pytest

from common.pool_progress import PoolProgress, run_pool_logged


def test_panel_collapses_output_and_keeps_redacted_diagnostics(tmp_path, monkeypatch):
    created, updated = [], []
    monkeypatch.setitem(sys.modules, 'IPython', SimpleNamespace(get_ipython=lambda: SimpleNamespace(kernel=True)))
    def display(data, **kwargs):
        created.append(data['text/plain'])
        def update(data, **kwargs):
            assert kwargs == {'raw': True}
            updated.append(data['text/plain'])
        return SimpleNamespace(update=update)
    monkeypatch.setitem(sys.modules, 'IPython.display', SimpleNamespace(display=display))
    monkeypatch.setenv('HF_TOKEN', 'private-token')
    lines = ['INFO enormous config private-token', 'WARNING diagnostic',
             'full: 100/200 rows already complete', 'full: 101/200 rows; index=5; correct=2/8']
    path = tmp_path / 'full.log'
    run_pool_logged([sys.executable, '-c', f'print({chr(10).join(lines)!r})'], cwd=tmp_path, log_path=path)
    assert len(created) == 1
    assert '101/200' in updated[-1] and 'Complete' in updated[-1]
    assert 'enormous config' not in updated[-1]
    assert 'WARNING diagnostic' in updated[-1]
    assert 'enormous config [REDACTED]' in path.read_text()
    assert 'private-token' not in path.read_text()


def test_failure_retains_actual_error_and_does_not_complete(tmp_path, capsys):
    with pytest.raises(RuntimeError, match='CUDA failed'):
        run_pool_logged([sys.executable, '-c', "print('CUDA failed'); exit(1)"],
                        cwd=tmp_path, log_path=tmp_path / 'failure.log')
    assert 'Stopped / failed' in capsys.readouterr().out


def test_resume_rate_excludes_previously_completed_rows(tmp_path, capsys):
    now = [0]
    p = PoolProgress(tmp_path / 'log', clock=lambda: now[0])
    p.consume('full: 100/200 rows already complete')
    now[0] = 60
    p.consume('full: 110/200 rows; index=20; correct=1/8')
    visible = capsys.readouterr().out
    assert '10.0 problems/min' in visible
    assert '00:09:00' in visible


def test_four_gpu_status_and_completion(tmp_path, capsys):
    p = PoolProgress(tmp_path / 'log')
    p.consume("GPU 0: {'done': 3, 'total': 10} | GPU 1: {'done': 5, 'total': 10} | GPU 2: starting / generating | GPU 3: finished")
    p.render(force=True)
    assert p.rows['GPU 0'] == (3, 10)
    assert p.rows['GPU 2'] == 'starting / generating'
    p.consume('GPU 0: finished | GPU 1: finished | GPU 2: finished | GPU 3: finished')
    assert p.rows['GPU 0'] == (10, 10)
    assert len(p.rows) == 4


def test_resume_diagnostic_identifies_settings_without_modifying_manifest(tmp_path):
    import json
    from pathlib import Path
    import yaml
    from common.io import digest
    from common.pool_progress import check_resume
    from pipeline.sample_sft_pool import code_hash
    cfg = {'engine': {'max_num_seqs': 16}}
    checksum = code_hash(Path(__file__).resolve().parents[1])
    manifest = {'config': cfg, 'code_hash': checksum,
                'signature': digest({'config': cfg, 'code_hash': checksum})}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    check_resume(config, tmp_path)
    cfg['engine']['max_num_seqs'] = 64
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='engine.max_num_seqs: saved=16; requested=64'):
        check_resume(config, tmp_path)
    assert path.read_bytes() == before
