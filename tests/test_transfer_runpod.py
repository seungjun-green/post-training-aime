import ast
import base64
import io
import json
import signal
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from eval import run_transfer_runpod as parallel
from eval import runpod_transfer_storage as storage
from eval.run_transfer_eval import bundle_files, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/qwen_transfer_eval.yaml'


def test_drive_plan_matches_evaluator_paths_and_manifests(tmp_path):
    config = load_config(CONFIG)
    plan = storage.drive_copy_plan(config, tmp_path)
    assert len(plan) == 7
    assert {p['key'] for p in plan} == set(config['models']) - {'base', 'instruct'}
    for item in plan:
        local = Path(item['destination'])
        entry = config['models'][item['key']]
        assert local.parts[-2:] == Path(entry['path']).parts[-2:]
        train_root = local.parents[3 if entry['kind'] == 'sft' else 2]
        expected = train_root / ('logs/stage1' if entry['kind'] == 'sft' else 'logs') / local.parent.name / 'run_manifest.json'
        assert Path(item['manifest_destination']) == expected
        for source, dest in [('source', 'destination'), ('manifest_source', 'manifest_destination')]:
            assert item[source] == 'gdrive:' + Path(item[dest]).relative_to(tmp_path / 'models').as_posix()
    by_key = {p['key']: p for p in plan}
    assert 'LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix/checkpoints/' in by_key['instruct_rl']['source']
    assert by_key['rejection_sft']['source'].endswith('/epoch_4')
    assert by_key['distill_sft']['source'].endswith('/epoch_5')
    assert storage.drive_copy_plan(config, tmp_path, selected=['base', 'instruct']) == []
    with pytest.raises(ValueError):
        storage.drive_copy_plan(config, tmp_path, remote='gdrive:arbitrary/folder')
    with pytest.raises(ValueError):
        storage.drive_copy_plan(config, tmp_path, selected=['base', 'base'])


def test_copy_only_inference_inputs_and_space_preflight(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    calls = []
    checks = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[1] == 'size':
            return SimpleNamespace(stdout=json.dumps({'bytes': 1234}))
        assert checks, 'Storage must be checked before copying'
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(storage, 'check_space', lambda path, **kw: checks.append(kw))
    plan = storage.copy_checkpoints(config, tmp_path, 'gdrive:', selected=['distill_sft', 'instruct_rl'], runner=runner)
    assert [c[1] for c in calls] == ['size', 'size', 'copy', 'copyto', 'copy', 'copyto']
    assert checks == [{'additional_bytes': 2468}]
    assert all('--checksum' in c for c in calls[2:])
    assert all('sync' not in c and 'delete' not in c for c in calls)
    rules = storage.CHECKPOINT_FILTERS
    assert rules[-1] == '- **'
    for name in ['optimizer', 'scheduler', 'rng_state', 'training_args']:
        assert not any(name in r for r in rules[:-1])
    for name in ['model*.safetensors', 'chat_template.jinja', 'stage1_checkpoint.json', 'dapo_checkpoint.json']:
        assert '+ /' + name in rules
    assert calls[2][2:4] == [plan[0]['source'], plan[0]['destination']]


def test_small_container_disk_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda path: SimpleNamespace(
        total=30 * storage.GIB, free=29 * storage.GIB))
    with pytest.raises(RuntimeError, match='30 GB container disk is insufficient'):
        storage.check_space(tmp_path)
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda path: SimpleNamespace(
        total=200 * storage.GIB, free=180 * storage.GIB))
    assert storage.check_space(tmp_path, additional_bytes=40 * storage.GIB)['required_gib'] == 100


def test_gpu_inventory_rejects_duplicate_alias_and_wrong_hardware(monkeypatch):
    text = '\n'.join(f'{i}, GPU-{i}, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.1' for i in range(4))
    monkeypatch.setattr(storage.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=text))
    assert len(storage.gpu_inventory(['0', '1', '2', '3'])) == 4
    assert len(storage.gpu_inventory(['0', '1'])) == 2
    assert len(storage.gpu_inventory(['GPU-0'])) == 1
    with pytest.raises(ValueError, match='duplicate physical'):
        storage.gpu_inventory(['0', 'GPU-0', '2', '3'])
    with pytest.raises(ValueError, match='one to four distinct'):
        storage.gpu_inventory([])
    with pytest.raises(ValueError, match='duplicate physical'):
        storage.gpu_inventory(['0', 'GPU-0'])
    text = text.replace('RTX PRO 6000', 'RTX 4090')
    with pytest.raises(ValueError, match='Expected a 96 GB'):
        storage.gpu_inventory(['0', '1', '2', '3'])


def test_incremental_progress_and_partial_lines(tmp_path):
    path = tmp_path / 'console.log'
    path.write_text('old log\n')
    state = {'log_path': path, 'offset': path.stat().st_size, 'done': 0}
    totals = {'arc_challenge': 4, 'mmlu_pro': 8, 'bbh_nonmath': 6, 'ifeval': 2}
    with path.open('a') as f:
        f.write('irrelevant output\narc_challenge: 4/4 problems\nmmlu_pro: 3/8 prob')
    parallel.consume_progress(state, totals)
    assert state['done'] == 4
    with path.open('a') as f:
        f.write('lems\n')
    parallel.consume_progress(state, totals)
    assert state['done'] == 7 and state['benchmark'] == 'mmlu_pro'
    parallel.consume_progress(state, totals)
    assert state['done'] == 7


def fake_launcher(monkeypatch, fail_key=None, interrupt=False):
    processes, killed, max_active = [], [], [0]

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 100000 + len(processes)
            self.key = command[command.index('--worker') + 1]
            self.command, self.kwargs = command, kwargs
            self.waited = False
            self.polls = 0
            active = [p for p in processes if not p.waited]
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] not in [p.kwargs['env']['CUDA_VISIBLE_DEVICES'] for p in active]
            processes.append(self)
            max_active[0] = max(max_active[0], len(active) + 1)
            kwargs['stdout'].write('arc_challenge: 2/2 problems\n')
            kwargs['stdout'].flush()

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                return None
            if interrupt:
                raise KeyboardInterrupt
            return 1 if self.key == fail_key else 0

        def wait(self, timeout=None):
            self.waited = True
            return 0

    monkeypatch.setattr(parallel.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    return Process, processes, killed, max_active


@pytest.mark.parametrize('gpu_count', [2, 4])
def test_nine_model_queue_uses_selected_independent_gpus(tmp_path, monkeypatch, gpu_count):
    factory, processes, killed, max_active = fake_launcher(monkeypatch)
    runs = [{'key': f'm{i}', 'label': f'Model {i}'} for i in range(9)]
    complete = []
    result = parallel.launch(CONFIG, tmp_path, 'experiments', 'instruct', runs, list(map(str, range(gpu_count))),
                             {'arc_challenge': 2}, popen=factory, poll_seconds=0,
                             on_complete=lambda: complete.append(True))
    assert result == [r['key'] for r in runs]
    assert len(complete) == 9 and max_active[0] == gpu_count
    for process in processes:
        assert process.kwargs['start_new_session']
        assert process.kwargs['env']['VLLM_WORKER_MULTIPROC_METHOD'] == 'spawn'
        assert process.waited and process.kwargs['stdout'].closed
        assert (process.pid, signal.SIGKILL) in killed
    status = json.loads((tmp_path / 'runpod_status.json').read_text())
    assert status['done'] == status['total'] == 18 and not status['active'] and not status['pending']


@pytest.mark.parametrize('interrupt', [False, True])
@pytest.mark.parametrize('gpu_count', [2, 4])
def test_failure_or_interrupt_cleans_all_active_worker_groups(tmp_path, monkeypatch, interrupt, gpu_count):
    factory, processes, killed, _ = fake_launcher(monkeypatch, fail_key='m0', interrupt=interrupt)
    runs = [{'key': f'm{i}', 'label': f'Model {i}'} for i in range(9)]
    with pytest.raises(KeyboardInterrupt if interrupt else RuntimeError):
        parallel.launch(CONFIG, tmp_path, 'experiments', 'instruct', runs, list(map(str, range(gpu_count))),
                        {'arc_challenge': 2}, smoke=True, popen=factory, poll_seconds=0)
    assert len(processes) == gpu_count  # No extra jobs start after failure.
    assert all(p.waited and p.kwargs['stdout'].closed for p in processes)
    assert all((p.pid, signal.SIGTERM) in killed and (p.pid, signal.SIGKILL) in killed for p in processes)
    assert all('--smoke' in p.command for p in processes)
    assert all('runpod_smoke_console.log' in p.kwargs['stdout'].name for p in processes)


def test_exclusive_locks_and_immutable_parallel_protocol(tmp_path):
    path = parallel.worker_lock(tmp_path, 'base')
    with parallel.exclusive(path):
        with pytest.raises(RuntimeError, match='Another evaluation process'):
            with parallel.exclusive(path):
                pytest.fail('Lock should not be acquired twice')
    with parallel.exclusive(path):
        pass
    config = load_config(CONFIG)
    parallel.pin_parallel_protocol(tmp_path, config)
    parallel.pin_parallel_protocol(tmp_path, config)
    config['generation']['temperature'] = 0.5
    with pytest.raises(ValueError, match='protocol changed'):
        parallel.pin_parallel_protocol(tmp_path, config)


@pytest.mark.parametrize('gpu_count', [2, 4])
def test_self_contained_notebook_and_isolated_cli(tmp_path, monkeypatch, gpu_count):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from build_qwen_transfer_runpod_notebook import EXTRA_FILES, FOUR_GPU_NAME, NAME, cells
    notebook = nbformat.read(ROOT / 'notebooks' / (NAME if gpu_count == 2 else FOUR_GPU_NAME), as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells(gpu_count)]
    sources = [c.source for c in notebook.cells if c.cell_type == 'code']
    for source in sources:
        compile(source, 'runpod_notebook', 'exec')
    all_code = '\n'.join(sources)
    assert 'google.colab' not in all_code and 'drive.mount' not in all_code
    for flag in ['COPY_FROM_DRIVE', 'RUN_SMOKE', 'RUN_EVAL', 'BACKUP_TO_DRIVE']:
        assert flag + ' = False' in all_code
    settings = {}
    exec(sources[0], settings)
    assert settings['GPU_IDS'] == list(map(str, range(gpu_count)))
    assert str(settings['STORAGE_ROOT']) == ('/root/qwen-transfer' if gpu_count == 2 else '/workspace/qwen-transfer')
    source = next(s for s in sources if 'BUNDLE = ' in s)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(node.value)))) as archive:
        assert set(archive.namelist()) == set(bundle_files() + EXTRA_FILES)
        for name in archive.namelist():
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(tmp_path)
    assert 'UPLOAD_TO_HF = True' in all_code and 'HF_PRIVATE = True' in all_code
    assert 'eval.export_transfer_results' in sources[-1]
    for module in ['eval.run_transfer_runpod', 'eval.run_transfer_eval', 'eval.export_transfer_results']:
        result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=tmp_path,
                                check=True, capture_output=True, text=True)
        assert '--output-root' in result.stdout
