"""Run nine transfer evaluations on a dynamic queue of one to four independent GPUs."""

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from contextlib import ExitStack, contextmanager
from pathlib import Path

from common.io import digest, write_json
from eval.run_transfer_eval import ROOT, comparison, load_config, run_model
from eval.runpod_transfer_storage import check_space, gpu_inventory
from eval.transfer_data import prepare_data, smoke_documents
from eval.transfer_models import prepare_models


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another evaluation process owns {path}') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def worker_lock(output_root, key):
    return Path(output_root) / 'models' / key / 'runpod_worker.lock'


def pin_parallel_protocol(output_root, config):
    # Device assignment can change across resumes; per-model manifests bind the
    # actual hardware/runtime and generation protocol. Queue code is separately pinned.
    record = {'version': 'model-queue-v1', 'config_digest': digest(config),
              'driver_digest': digest(Path(__file__).read_text()),
              'storage_helper_digest': digest((ROOT / 'eval/runpod_transfer_storage.py').read_text()),
              'tensor_parallel_size': 1, 'max_concurrent_models': 4}
    path = Path(output_root) / 'runpod_protocol.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('RunPod launcher/protocol changed; use a new OUTPUT_ROOT')
    write_json(path, record)


def consume_progress(state, totals):
    """Read only new log bytes. Workers never share stdout or result files."""
    with state['log_path'].open('r', errors='replace') as stream:
        stream.seek(state['offset'])
        text = state.get('tail', '') + stream.read()
        state['offset'] = stream.tell()
    lines = text.split('\n')
    state['tail'] = lines.pop()
    for line in lines:
        match = re.fullmatch(r'([\w-]+): (\d+)/(\d+) problems\s*', line)
        if match and match[1] in totals and int(match[3]) == totals[match[1]]:
            before = 0
            for benchmark, count in totals.items():
                if benchmark == match[1]:
                    break
                before += count
            state['done'] = max(state['done'], before + int(match[2]))
            state['benchmark'] = match[1]


def stop_processes(states):
    # Each child owns a session containing its vLLM workers, so cancellation also
    # stops descendants. Sending to exited leaders cleans up surviving children.
    for state in states:
        try:
            os.killpg(state['process'].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for state in states:
        process = state['process']
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        state['log'].close()


def launch(config_path, output_root, experiment_root, instruct_rl_root, runs, gpu_ids, totals,
           *, smoke=False, on_complete=None, popen=subprocess.Popen, poll_seconds=1):
    queue, active = deque(runs), {}
    completed = []
    per_model = sum(totals.values())
    status_path = Path(output_root) / ('runpod_smoke_status.json' if smoke else 'runpod_status.json')
    last_status = 0
    try:
        while queue or active:
            for gpu in gpu_ids:
                if gpu in active or not queue:
                    continue
                run = queue.popleft()
                log_path = Path(output_root) / 'models' / run['key'] / 'logs' / (
                    'runpod_smoke_console.log' if smoke else 'runpod_eval_console.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                offset = log_path.stat().st_size if log_path.exists() else 0
                log = log_path.open('a')
                command = [sys.executable, '-m', 'eval.run_transfer_runpod', '--worker', run['key'],
                           '--config', str(config_path), '--output-root', str(output_root),
                           '--experiment-root', str(experiment_root), '--instruct-rl-root', str(instruct_rl_root)]
                if smoke:
                    command.append('--smoke')
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONUNBUFFERED': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
                       'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'}
                try:
                    process = popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active[gpu] = {'run': run, 'process': process, 'log': log, 'log_path': log_path,
                               'offset': offset, 'done': 0, 'benchmark': 'loading'}
                print(f'GPU {gpu}: starting {run["label"]}', flush=True)
            finished = []
            for gpu, state in active.items():
                consume_progress(state, totals)
                code = state['process'].poll()
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f'{state["run"]["key"]} failed on GPU {gpu} (exit {code}). '
                                       f'See {state["log_path"]}. Completed responses are preserved.')
                stop_processes([state])
                completed.append(state['run']['key'])
                finished.append(gpu)
                print(f'GPU {gpu}: completed {state["run"]["label"]}', flush=True)
                if on_complete:
                    on_complete()
            for gpu in finished:
                del active[gpu]
            if finished or time.monotonic() - last_status >= 15:
                done = len(completed) * per_model + sum(s['done'] for s in active.values())
                status = {'completed': completed, 'pending': [r['key'] for r in queue],
                          'done': done, 'total': len(runs) * per_model,
                          'active': {g: {'model': s['run']['key'], 'done': s['done'],
                                        'total': per_model, 'benchmark': s['benchmark']} for g, s in active.items()}}
                write_json(status_path, status)
                print(f'transfer: {done}/{status["total"]} problems', flush=True)
                print(' | '.join(f'GPU {g}: {s["run"]["key"]} {s["benchmark"]} '
                                 f'{s["done"]}/{per_model}' for g, s in active.items()), flush=True)
                last_status = time.monotonic()
            if active:
                time.sleep(poll_seconds)
    finally:
        stop_processes(list(active.values()))
    return completed


def prefetch_hf(config, runs):
    from huggingface_hub import snapshot_download
    for run in runs:
        if run['revision']:
            print('Prefetch pinned model:', run['model'], flush=True)
            snapshot_download(run['model'], revision=run['revision'], token=os.getenv('HF_TOKEN'),
                              allow_patterns=['*.safetensors', '*.json', '*.jinja', '*.txt', '*.model'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/qwen_transfer_eval.yaml'))
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--instruct-rl-root', required=True)
    parser.add_argument('--storage-root', default='/root/qwen-transfer')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--models', help='Comma-separated subset; default all nine')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker')
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output_root)
    if args.worker:
        if args.worker not in cfg['models']:
            parser.error('Unknown model key')
        with exclusive(worker_lock(output, args.worker)):
            runs = prepare_models(cfg, args.experiment_root, args.instruct_rl_root, output, [args.worker])
            docs, manifest = prepare_data(cfg, output)
            run_model(runs[0], cfg, docs, manifest, output, smoke=args.smoke)
        return

    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    gpu_ids = args.gpus.split(',')
    print('GPU inventory:', gpu_inventory(gpu_ids), flush=True)
    print('Storage:', check_space(args.storage_root), flush=True)
    selected = args.models.split(',') if args.models else list(cfg['models'])
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(cfg['models']):
        parser.error('Select unique known model keys')
    with exclusive(output / 'runpod_coordinator.lock'):
        pin_parallel_protocol(output, cfg)
        # Detect orphaned workers before rewriting selection/data or starting work.
        with ExitStack() as stack:
            for key in selected:
                stack.enter_context(exclusive(worker_lock(output, key)))
            runs = prepare_models(cfg, args.experiment_root, args.instruct_rl_root, output, selected)
            docs, manifest = prepare_data(cfg, output)
        prefetch_hf(cfg, runs)
        totals = manifest['counts']
        if args.smoke:
            smoke = smoke_documents(docs)
            totals = {name: sum(d['benchmark'] == name for d in smoke) for name in totals}
        print(f'{len(runs)} models; {sum(totals.values())} prompts/model; '
              f'{len(gpu_ids)} single-GPU workers', flush=True)
        if args.prepare_only:
            return
        launch(Path(args.config).resolve(), output, args.experiment_root, args.instruct_rl_root,
               runs, gpu_ids, totals, smoke=args.smoke,
               on_complete=None if args.smoke else lambda: comparison(cfg, output))


if __name__ == '__main__':
    main()
