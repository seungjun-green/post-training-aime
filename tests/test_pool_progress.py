import sys
from types import SimpleNamespace

import pytest

from common.pool_progress import PoolProgress, run_pool_logged


@pytest.fixture
def bars(monkeypatch):
    created = []
    class Bar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.n = kwargs.get('initial', 0)
            self.total = kwargs.get('total')
            self.closed = False
            self.postfix = {}
            created.append(self)
        def update(self, delta):
            self.n += delta
        def close(self):
            self.closed = True
        def refresh(self):
            pass
        def set_description_str(self, text, **kwargs):
            self.description = text
        def set_postfix(self, postfix, **kwargs):
            self.postfix = postfix
    monkeypatch.setitem(sys.modules, 'tqdm.auto', SimpleNamespace(tqdm=Bar))
    return created


def test_real_bar_tracks_counts_and_keeps_redacted_diagnostics(tmp_path, monkeypatch, bars, capsys):
    monkeypatch.setenv('HF_TOKEN', 'private-token')
    lines = ['INFO enormous config private-token', 'WARNING diagnostic',
             'full: 100/200 rows already complete', 'full: 101/200 rows; index=5; correct=2/8']
    path = tmp_path / 'full.log'
    run_pool_logged([sys.executable, '-c', f'print({chr(10).join(lines)!r})'], cwd=tmp_path, log_path=path)
    progress = bars[-1]
    assert progress.total == 200 and progress.n == 101
    assert progress.kwargs['initial'] == 100
    assert progress.postfix['last correct'] == '2/8'
    assert progress.postfix['log warnings'] == 1
    assert all(bar.closed for bar in bars)
    assert 'enormous config [REDACTED]' in path.read_text()
    assert 'private-token' not in path.read_text()
    assert "text/plain" not in capsys.readouterr().out


def test_failure_retains_error_and_partial_bar(tmp_path, capsys, bars):
    with pytest.raises(RuntimeError, match='CUDA failed'):
        run_pool_logged([sys.executable, '-c', "print('full: 3/10 rows already complete'); print('full: 4/10 rows; index=4; correct=1/8'); print('CUDA failed'); exit(1)"],
                        cwd=tmp_path, log_path=tmp_path / 'failure.log')
    assert 'Stopped / failed' in capsys.readouterr().out
    assert bars[-1].n == 4 and bars[-1].closed


def test_resume_starts_timing_after_warmup(tmp_path, bars):
    p = PoolProgress(tmp_path / 'log')
    p.consume('full: 100/200 rows already complete')
    p.consume('Capturing CUDA graphs')
    assert len(bars) == 1  # Only the startup indicator exists during compilation.
    p.consume('Generation active: 4 problems pending; GPU response limit 16')
    assert len(bars) == 2 and bars[-1].kwargs['initial'] == 100
    p.consume('full: 110/200 rows; index=20; correct=1/8')
    assert bars[-1].n == 110
    assert '{remaining}' in bars[-1].kwargs['bar_format']
    p.finish('Complete')


def test_four_gpu_bars_and_completion(tmp_path, bars):
    p = PoolProgress(tmp_path / 'log')
    p.consume("GPU 0: {'done': 3, 'total': 10} | GPU 1: {'done': 5, 'total': 10} | GPU 2: starting / generating | GPU 3: finished")
    assert len(p.bars) == 4
    assert p.bars['GPU 0'].n == 3
    p.consume("GPU 0: finished | GPU 1: finished | GPU 2: {'done': 4, 'total': 10} | GPU 3: finished")
    assert p.bars['GPU 0'].n == 10
    assert p.bars['GPU 2'].total == 10
    assert p.bars['GPU 2'].n == 4
    p.finish('Complete')
    assert all(bar.closed for bar in bars)


def test_text_tqdm_fallback_renders_progress_and_eta(tmp_path, capsys, monkeypatch):
    from tqdm.std import tqdm
    monkeypatch.setitem(sys.modules, 'tqdm.auto', SimpleNamespace(tqdm=tqdm))
    p = PoolProgress(tmp_path / 'log')
    p.consume('full: 100/200 rows already complete')
    p.consume('full: 101/200 rows; index=1; correct=0/8')
    p.finish('Complete')
    output = capsys.readouterr().out
    assert '101/200 problems' in output
    assert 'remaining' in output and 'elapsed' in output
    assert 'text/plain' not in output


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
