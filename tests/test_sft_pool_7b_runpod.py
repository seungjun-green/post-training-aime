import ast
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import nbformat
import pyarrow.parquet as pq
import pytest
import yaml
from datasets import Dataset

from common.io import append_jsonl, write_json
from pipeline import sample_sft_pool_7b as seven
from pipeline import sample_sft_pool_7b_runpod as parallel
from pipeline import sample_sft_pool_runpod as launcher

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT / 'configs/sft_pool_sampling_7b.yaml').read_text())
    cfg = parallel.parallel_config(cfg, ROOT, 4)
    scores = [0, 3, 1, 8, 2, 0, 4, 1, 5, 2, 6, 7]
    ds = Dataset.from_list([{'id': str(i), 'problem': 'Compute 6 times 7.', 'gold_answer': '42',
         'responses': ['old response'] * 8, 'extracted_answers': ['42'] * 8,
         'correct': [j < score for j in range(8)], 'response_tokens': [10] * 8,
         'finish_reasons': ['stop'] * 8, 'num_correct': score}
        for i, score in enumerate(scores)])
    manifest = {'signature': 'test', 'config': cfg, 'selected_indices': seven.select_rows(ds),
                'source_rows': len(ds)}
    ds.save_to_disk(str(tmp_path / 'prepared'))
    write_json(tmp_path / 'manifest.json', manifest)
    seen = []
    def engine(cfg, manifest, directory):
        write_json(directory / 'runtime.json', {'gpu': 'fake RTX PRO 6000'})
        write_json(directory / 'prompt.json', {'template': 'instruct'})
        tokenizer = SimpleNamespace(chat_template='instruct', encode=lambda *a, **kw: [1, 2],
                                    apply_chat_template=lambda *a, **kw: 'prompt')
        return SimpleNamespace(tokenizer=tokenizer)
    monkeypatch.setattr(seven, 'make_engine', engine)
    def generate(engine, jobs, execution, *, prompt_renderer):
        assert prompt_renderer is seven.render_prompt
        for job in reversed(list(jobs)):
            seen.append(job)
            yield job, '', [{'text': r'\boxed{42}', 'token_count': 5, 'finish_reason': 'stop'}] * 8
    import eval.batched_generation
    monkeypatch.setattr(eval.batched_generation, 'generate_problems', generate)
    return cfg, ds, manifest, tmp_path, seen


def test_four_workers_cover_only_selected_rows_resume_and_merge(run):
    cfg, ds, manifest, path, seen = run
    for rank in range(4):
        parallel.worker(cfg, ROOT, path, rank, 4)
    expected = manifest['selected_indices']
    assert sorted(j['source_index'] for j in seen) == expected
    assert len({j['source_index'] for j in seen}) == len(expected)
    assert all(j['n'] == 8 for j in seen)
    for rank in range(4):
        journal = path / f'workers/{rank}/responses_7b.jsonl'
        saved = [json.loads(line)['source_index'] for line in journal.read_text().splitlines()]
        assert set(saved) == set(expected[rank::4])
    # Partial final append is repaired, without re-generating completed rows.
    with (path / 'workers/0/responses_7b.jsonl').open('ab') as f:
        f.write(b'{"unfinished')
    for rank in range(4):
        parallel.worker(cfg, ROOT, path, rank, 4)
    assert len(seen) == len(expected)
    out = parallel.merge(ds, cfg, manifest, path, 4)
    seven.validate_export(ds, manifest, out)
    rows = [r for p in sorted((out / 'data').glob('*.parquet')) for r in pq.read_table(p).to_pylist()]
    assert [r['id'] for r in rows] == list(ds['id'])
    for i, row in enumerate(rows):
        assert {k: row[k] for k in ds.column_names} == ds[i]
        assert row['7B_num_correct'] == (8 if i in expected else None)
    provenance = json.loads((out / 'provenance.json').read_text())
    assert provenance['manifest']['config']['parallel']['workers'] == 4


def test_three_workers_cover_selected_rows_resume_and_preserve_nulls(run):
    cfg, ds, manifest, path, seen = run
    cfg = parallel.parallel_config(cfg, ROOT, 3)
    manifest['config'] = cfg
    write_json(path / 'manifest.json', manifest)
    for rank in range(3):
        parallel.worker(cfg, ROOT, path, rank, 3)
    assert sorted(j['source_index'] for j in seen) == manifest['selected_indices']
    assert not (path / 'workers/3').exists()
    journal = path / 'workers/1/responses_7b.jsonl'
    saved = journal.read_text().splitlines()
    missing = json.loads(saved[-1])['source_index']
    journal.write_text(saved[0] + '\n')
    seen.clear()
    for rank in range(3):
        parallel.worker(cfg, ROOT, path, rank, 3)
    assert [j['source_index'] for j in seen] == [missing]
    out = parallel.merge(ds, cfg, manifest, path, 3)
    seven.validate_export(ds, manifest, out)
    rows = [r for f in sorted((out / 'data').glob('*.parquet')) for r in pq.read_table(f).to_pylist()]
    for row in rows:
        if row['num_correct'] >= 3:
            assert all(row[key] is None for key in seven.NEW_COLUMNS)
        else:
            assert row['7B_num_correct'] == 8


def test_partial_resume_generates_only_missing_worker_rows(run):
    cfg, ds, m, path, seen = run
    parallel.worker(cfg, ROOT, path, 0, 4)
    journal = path / 'workers/0/responses_7b.jsonl'
    first = journal.read_text().splitlines()[0]
    kept = json.loads(first)['source_index']
    journal.write_text(first + '\n')
    seen.clear()
    parallel.worker(cfg, ROOT, path, 0, 4)
    assert [j['source_index'] for j in seen] == [i for i in m['selected_indices'][::4] if i != kept]


def test_missing_or_misassigned_worker_rows_block_export(run):
    cfg, ds, m, path, _ = run
    for rank in range(3):
        parallel.worker(cfg, ROOT, path, rank, 4)
    with pytest.raises(ValueError, match='incomplete'):
        parallel.merge(ds, cfg, m, path, 4)
    parallel.worker(cfg, ROOT, path, 3, 4)
    row = json.loads((path / 'workers/0/responses_7b.jsonl').read_text().splitlines()[0])
    append_jsonl(path / 'workers/1/responses_7b.jsonl', row)
    with pytest.raises(ValueError, match='outside'):
        parallel.merge(ds, cfg, m, path, 4)
    assert not (path / 'full/COMPLETE.json').exists()


def test_worker_locks_runtime_and_config_guard(run):
    cfg, ds, m, path, _ = run
    with parallel.exclusive(path / 'workers/0/worker.lock'):
        with pytest.raises(RuntimeError, match='Another process'):
            parallel.worker(cfg, ROOT, path, 0, 4)
    with pytest.raises(ValueError, match='disagree'):
        parallel.worker({**cfg, 'changed': True}, ROOT, path, 0, 4)
    for rank in range(4):
        parallel.worker(cfg, ROOT, path, rank, 4)
    write_json(path / 'workers/2/runtime.json', {'gpu': 'different'})
    with pytest.raises(ValueError, match='disagree'):
        parallel.merge(ds, cfg, m, path, 4)


@pytest.mark.parametrize('count', [0, 1, 2, 3])
def test_empty_worker_partitions_need_no_gpu_records(run, count):
    cfg, ds, m, path, seen = run
    # Restrict the source itself, keeping the full selection contract valid.
    indices = m['selected_indices'][:count] + [i for i in range(len(ds)) if i not in m['selected_indices']]
    ds = ds.select(indices)
    m.update(source_rows=len(ds), selected_indices=seven.select_rows(ds))
    ds.save_to_disk(str(path / 'prepared'))
    write_json(path / 'manifest.json', m)
    for rank in range(4):
        parallel.worker(cfg, ROOT, path, rank, 4)
    out = parallel.merge(ds, cfg, m, path, 4)
    seven.validate_export(ds, m, out)
    assert len(seen) == count


@pytest.mark.parametrize('workers', [3, 4])
def test_7b_launch_uses_distinct_devices_and_stops_all_on_failure(tmp_path, monkeypatch, workers):
    calls, killed = [], []
    class Process:
        def __init__(self, cmd, **kw):
            self.pid = 100 + len(calls)
            calls.append((cmd, kw))
        def poll(self): return 1 if self.pid == 101 else None
        def wait(self, **kw): return 0
    monkeypatch.setattr(launcher.subprocess, 'Popen', Process)
    monkeypatch.setattr(launcher.os, 'killpg', lambda pid, sig: killed.append(pid))
    with pytest.raises(RuntimeError, match='Worker 1 failed'):
        parallel.launch(tmp_path / 'cfg.yaml', ROOT, tmp_path, [str(i) for i in range(workers)],
                        module='pipeline.sample_sft_pool_7b_runpod')
    assert all(cmd[2] == 'pipeline.sample_sft_pool_7b_runpod' for cmd, _ in calls)
    assert [kw['env']['CUDA_VISIBLE_DEVICES'] for _, kw in calls] == [str(i) for i in range(workers)]
    assert set(killed) == set(range(100, 100 + workers))
    assert all(kw['stdout'].closed for _, kw in calls)


@pytest.mark.parametrize('workers', [3, 4])
def test_runpod_notebook_is_self_contained_and_has_no_colab_dependencies(workers):
    nb = nbformat.read(ROOT / f'notebooks/sample_sft_pool_math7b_8_runpod_{workers}gpu.ipynb', as_version=4)
    nbformat.validate(nb)
    code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    for c in nb.cells:
        if c.cell_type == 'code': compile(c.source, '<runpod>', 'exec')
    assert 'google.colab' not in code and 'userdata.get' not in code and '/content/' not in code
    assert "getpass.getpass('Hugging Face write token: ')" in code
    assert "'pipeline.sample_sft_pool_7b_runpod'" in code
    assert 'UPLOAD_TO_HF = True' in code and 'MAX_NUM_SEQS = 64' in code
    assert f'GPU_IDS = {[str(i) for i in range(workers)]!r}' in code
    assert f'torch.cuda.device_count() == {workers}' in code
    assert f'runpod{workers}-v1' in code
    assert "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'" in code
    assert "'/workspace/" in code and "['--mode', 'full']" in code and "['--mode', 'upload']" in code
    tree = ast.parse(code)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(call.args[0])))) as z:
        for name in z.namelist(): assert z.read(name) == (ROOT / name).read_bytes(), name
        assert 'pipeline/sample_sft_pool_7b_runpod.py' in z.namelist()
