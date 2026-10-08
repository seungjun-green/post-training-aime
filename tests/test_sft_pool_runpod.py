import ast
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from datasets import Dataset
import nbformat
import pyarrow.parquet as pq
import pytest
import yaml

from common.io import append_jsonl, write_json
from pipeline import sample_sft_pool as base
from pipeline import sample_sft_pool_runpod as parallel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT / 'configs/sft_pool_sampling.yaml').read_text())
    ds = Dataset.from_list([{'id': f'row-{i}', 'problem': 'Compute 2+2.', 'gold_answer': '4'} for i in range(11)])
    manifest = {'signature': 'test', 'model_revision': 'pinned'}
    ds.save_to_disk(str(tmp_path / 'prepared'))
    write_json(tmp_path / 'manifest.json', manifest)
    parallel.protocol(cfg, ROOT, tmp_path, 4)
    seen = []
    def make_engine(cfg, manifest, directory):
        write_json(directory / 'runtime.json', {'gpu': 'fake'})
        write_json(directory / 'prompt.json', {'template': 'fake'})
        tokenizer = SimpleNamespace(chat_template='fake',
            apply_chat_template=lambda *a, **kw: 'test prompt',
            encode=lambda *a, **kw: [1, 2])
        return SimpleNamespace(tokenizer=tokenizer, config={**cfg['engine'], **cfg['sampling']})
    monkeypatch.setattr(base, 'make_engine', make_engine)
    import eval.batched_generation
    def generate(engine, jobs, execution):
        for job in jobs:
            seen.append(job)
            yield job, '', [{'text': r'\boxed{4}', 'finish_reason': 'stop', 'token_count': 4}] * 8
    monkeypatch.setattr(eval.batched_generation, 'generate_problems', generate)
    return cfg, ds, manifest, tmp_path, seen


def test_workers_resume_merge_and_provenance(run):
    cfg, ds, manifest, path, seen = run
    for rank in range(4):
        parallel.worker(cfg, ROOT, path, rank, 4)
    assert sorted(j['source_index'] for j in seen) == list(range(len(ds)))
    assert all(j['seed'] == base.sample_seed(cfg['sampling']['seed'], j['key']) for j in seen)
    # Recover an interrupted tail, preserving completed problems without regeneration.
    with (path / 'workers/2/responses.jsonl').open('ab') as f:
        f.write(b'{"source_index":')
    for rank in range(4):
        parallel.worker(cfg, ROOT, path, rank, 4)
    assert len(seen) == len(ds)
    out = parallel.merge(ds, cfg, manifest, path, 4)
    rows = [r for p in sorted((out / 'data').glob('*.parquet')) for r in pq.read_table(p).to_pylist()]
    assert [r['id'] for r in rows] == list(ds['id'])
    assert all(r['num_correct'] == 8 for r in rows)
    provenance = json.loads((out / 'provenance.json').read_text())
    assert provenance['parallel']['workers'] == 4
    complete = json.loads((out / 'COMPLETE.json').read_text())
    assert all(base.hash_file(out / p) == h for p, h in complete['files'].items())


def test_merge_rejects_missing_and_wrong_owner(run):
    cfg, ds, manifest, path, _ = run
    for rank in range(3):
        parallel.worker(cfg, ROOT, path, rank, 4)
    with pytest.raises(ValueError, match='incomplete'):
        parallel.merge(ds, cfg, manifest, path, 4)
    parallel.worker(cfg, ROOT, path, 3, 4)
    row = json.loads((path / 'workers/0/responses.jsonl').read_text().splitlines()[0])
    append_jsonl(path / 'workers/1/responses.jsonl', row)
    with pytest.raises(ValueError, match='misassigned'):
        parallel.merge(ds, cfg, manifest, path, 4)
    assert not (path / 'full/COMPLETE.json').exists()


def test_locks_and_protocol(run):
    cfg, ds, manifest, path, _ = run
    with parallel.exclusive(path / 'workers/0/worker.lock'):
        with pytest.raises(RuntimeError, match='Another process'):
            parallel.worker(cfg, ROOT, path, 0, 4)
    with pytest.raises(ValueError, match='changed'):
        parallel.protocol(cfg, ROOT, path, 3)


def test_runtime_mismatch_blocks_merge(run):
    cfg, ds, manifest, path, _ = run
    for rank in range(4):
        parallel.worker(cfg, ROOT, path, rank, 4)
    write_json(path / 'workers/2/runtime.json', {'gpu': 'different'})
    with pytest.raises(ValueError, match='disagree'):
        parallel.merge(ds, cfg, manifest, path, 4)


def test_launch_assigns_devices_and_cleans_up_on_failure(tmp_path, monkeypatch):
    calls, signals = [], []
    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(calls)
            calls.append((command, kwargs))
        def poll(self):
            return 1 if self.pid == 102 else None
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(parallel.subprocess, 'Popen', Process)
    monkeypatch.setattr(parallel.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(RuntimeError, match='Worker 2 failed'):
        parallel.launch(tmp_path / 'config.yaml', ROOT, tmp_path, ['0', '1', '2', '3'])
    assert [k['env']['CUDA_VISIBLE_DEVICES'] for _, k in calls] == ['0', '1', '2', '3']
    assert all(k['start_new_session'] for _, k in calls)
    assert {pid for pid, sig in signals} == {100, 101, 102, 103}
    assert all(k['stdout'].closed for _, k in calls)


def test_notebook_compiles_and_bundle_is_current():
    path = ROOT / 'notebooks/full_run_clean_sft_pool_qwen3b_8_runpod_4gpu.ipynb'
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, str(path), 'exec')
            assert 'google.colab' not in cell.source
    tree = ast.parse(nb.cells[3].source)
    encoded = next(n.args[0].value for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as z:
        assert 'pipeline/sample_sft_pool_runpod.py' in z.namelist()
        for name in z.namelist():
            assert z.read(name) == (ROOT / name).read_bytes(), name
    assert "'pipeline.sample_sft_pool_runpod'" in nb.cells[5].source
    assert "'--gpus'" in nb.cells[5].source


def test_partial_worker_resume_only_generates_remaining_rows(run):
    cfg, ds, manifest, path, seen = run
    parallel.worker(cfg, ROOT, path, 0, 4)
    journal = path / 'workers/0/responses.jsonl'
    first = journal.read_text().splitlines()[0]
    journal.write_text(first + '\n')
    seen.clear()
    parallel.worker(cfg, ROOT, path, 0, 4)
    assert [job['source_index'] for job in seen] == [4, 8]
    assert set(base.scan_journal(journal, ds, cfg['data'])) == {0, 4, 8}
