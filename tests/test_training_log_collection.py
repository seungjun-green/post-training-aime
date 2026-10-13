import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import nbformat

from scripts.collect_qwen_training_logs import EXPERIMENTS, RUNS, collect

ROOT = Path(__file__).resolve().parents[1]


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_collection_keeps_histories_and_eval_but_excludes_large_artifacts(tmp_path):
    drive = tmp_path / 'drive'
    for key, folder, name, kind in RUNS:
        root = (drive / EXPERIMENTS / folder).resolve()
        logs = root / ('logs/stage1' if kind == 'sft' else 'logs') / name
        records = [{'step': 1, 'loss': 0.5, 'policy_metrics': {'entropy': 1.2}},
                   {'step': 2, 'loss': 0.4, 'mean_length': 100}]
        put(logs / 'steps.jsonl', '\n'.join(json.dumps(r) for r in records) + '\n')
        put(logs / 'run_manifest.json', json.dumps({'key': key}))
        put(logs / 'rollouts/001/sampling.json', 'EXCLUDED')
        put(logs / 'rollouts/001/batch_001.jsonl', 'EXCLUDED')
        put(logs / 'secret.env', 'EXCLUDED')
        put(logs / 'training_console.log', 'EXCLUDED')
        cp = root / ('checkpoints/stage1' if kind == 'sft' else 'checkpoints') / name / 'epoch_1'
        put(cp / 'trainer_state.json', '{"global_step": 2}')
        put(cp / 'model.safetensors', 'EXCLUDED')
        put(cp / 'optimizer.pt', 'EXCLUDED')
        put(root / 'inputs' / name / 'training.yaml', 'epochs: 5')
        put(root / 'inputs' / name / 'dataset.jsonl', 'EXCLUDED')
        put(root / 'eval' / name / 'epoch_summary.csv', 'epoch,accuracy\n1,50\n')
        put(root / 'eval' / name / 'epoch_1/full/results/test.json', '{"accuracy": 50}')
        put(root / 'eval' / name / 'epoch_1/full/results/test_generations.jsonl', 'EXCLUDED')
    parent = drive / 'LG-AIME-DAPO-Qwen25-3B-MiniBatch300/logs/dapo_qwen25_3b_base/steps.jsonl'
    parent.write_text('{"step": 40}\n{"step": 41}\n')
    original = parent.read_bytes()
    path, manifest = collect(drive, tmp_path / 'exports', notify=lambda _: None)
    assert manifest['status'] == 'complete'
    assert len(manifest['runs']) == 9
    assert all(r['steps_collected'] for r in manifest['runs'])
    assert 'policy_metrics.entropy' in manifest['runs'][0]['metric_fields']
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert {'README.md', 'run_inventory.csv', 'collection_manifest.json'} <= set(archive.namelist())
        for item in manifest['files']:
            raw = archive.read(item['archive_path'])
            assert raw == Path(item['source']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == item['sha256']
            assert b'EXCLUDED' not in raw
        assert archive.read('drive/' + parent.relative_to(drive).as_posix()) == original
        assert b'Original steps' in archive.read('README.md')
    assert parent.read_bytes() == original


def test_missing_malformed_oversize_and_symlink_are_reported(tmp_path):
    drive = tmp_path / 'drive'
    logs = drive / EXPERIMENTS / 'base-rl-native-eos/logs/dapo_qwen25_3b_pretrained_base'
    put(logs / 'steps.jsonl', '{"step": 1}\n{"step": 1}\nBROKEN\n')
    put(logs / 'run_manifest.json', 'x' * 2000)
    outside = put(tmp_path / 'outside.json', 'EXCLUDED')
    (logs / 'train_metrics.json').symlink_to(outside)
    path, report = collect(drive, tmp_path / 'exports', max_file_mb=0.001, notify=lambda _: None)
    assert report['status'] == 'partial'
    assert report['missing'] and report['skipped']
    assert report['runs'][0]['malformed_lines'] == 1
    assert report['runs'][0]['duplicate_step_records'] == 1
    with zipfile.ZipFile(path) as archive:
        assert not any('train_metrics.json' in n for n in archive.namelist())


def test_notebook_bundle_and_cpu_workflow(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from build_training_log_collection_notebook import FILES, NAME, cells
    nb = nbformat.read(ROOT / 'notebooks' / NAME, as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells] == [c.source for c in cells()]
    sources = [c.source for c in nb.cells if c.cell_type == 'code']
    for source in sources:
        compile(source, NAME, 'exec')
    boot = next(s for s in sources if 'BUNDLE =' in s)
    node = next(n for n in ast.parse(boot).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(node.value)))) as archive:
        assert set(archive.namelist()) == set(FILES)
        for name in FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()
    assert 'files.download(str(ZIP_PATH))' in sources[-1]
    assert 'shutil.copy2(ZIP_PATH, drive_copy)' in '\n'.join(sources)
    assert 'pip install' not in '\n'.join(sources)
