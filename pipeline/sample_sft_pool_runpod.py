"""Four independent single-GPU samplers; parent alone prepares, merges and uploads."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml
from common.io import append_jsonl, digest, write_json
from pipeline import sample_sft_pool as base


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process owns {path}; stop it before restarting') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def assignment(size, rank, workers):
    if not 0 <= rank < workers:
        raise ValueError('Invalid worker rank')
    return list(range(rank, size, workers))


def protocol(cfg, root, run_dir, workers):
    value = {'workers': workers, 'partition': 'source-index-modulo-v1',
             'base_signature': digest({'config': cfg, 'code_hash': base.code_hash(root)}),
             'driver_hash': base.hash_file(Path(__file__))}
    path = run_dir / 'parallel_protocol.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Parallel code/settings changed; use a new RUN_NAME')
    write_json(path, value)


def worker(cfg, root, run_dir, rank, workers):
    from datasets import load_from_disk
    from common.english_prompts import render_prompt
    from eval.batched_generation import generate_problems
    from eval.engines import check_context
    directory = run_dir / 'workers' / str(rank)
    with exclusive(directory / 'worker.lock'):
        ds = load_from_disk(str(run_dir / 'prepared'))
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        selected = assignment(len(ds), rank, workers)
        journal = directory / 'responses.jsonl'
        completed = base.scan_journal(journal, ds, cfg['data'])
        if set(completed) - set(selected):
            raise ValueError('Worker checkpoint contains another worker’s rows')
        pending = [i for i in selected if i not in completed]
        print(f'GPU worker {rank}: {len(completed)}/{len(selected)} saved', flush=True)
        if not pending:
            return
        engine = base.make_engine(cfg, manifest, directory)
        for i in pending:
            prompt = render_prompt(engine.tokenizer, ds[i][cfg['data']['problem_column']])
            check_context(engine.tokenizer.encode(prompt, add_special_tokens=False), engine.config)
        jobs = ({'key': str(ds[i][cfg['data']['id_column']]), 'source_index': i,
                 'problem': ds[i][cfg['data']['problem_column']], 'n': 8,
                 'seed': base.sample_seed(cfg['sampling']['seed'], ds[i][cfg['data']['id_column']])}
                for i in pending)
        done = len(completed)
        for job, _, responses in generate_problems(engine, jobs, cfg['execution']):
            row = ds[job['source_index']]
            audit = []
            annotation, errors = base.annotate(responses, row[cfg['data']['answer_column']],
                cfg['sampling']['verify_timeout_seconds'], problem=row[cfg['data']['problem_column']],
                answer_type=row.get('answer_type'), audit_output=audit)
            append_jsonl(journal, {'source_index': job['source_index'], 'id': job['key'],
                'annotation': annotation, 'grading_errors': errors, 'grading_audit': audit,
                'generation_diagnostics': [{k: r.get(k) for k in ['stop_reason', 'last_token_id']}
                                           for r in responses]})
            done += 1
            write_json(directory / 'progress.json', {'done': done, 'total': len(selected)})
            print(f'worker {rank}: {done}/{len(selected)}; correct={annotation["num_correct"]}/8', flush=True)


def merge(ds, cfg, manifest, run_dir, workers):
    from contextlib import ExitStack
    # Acquire worker locks too: an orphaned worker must never race merge/export.
    with ExitStack() as stack:
        handles, indices = [], []
        for rank in range(workers):
            directory = run_dir / 'workers' / str(rank)
            stack.enter_context(exclusive(directory / 'worker.lock'))
            path = directory / 'responses.jsonl'
            index = base.scan_journal(path, ds, cfg['data'])
            if set(index) != set(assignment(len(ds), rank, workers)):
                raise ValueError(f'Worker {rank} is incomplete or has misassigned rows')
            indices.append(index)
            handles.append(stack.enter_context(path.open('rb')))
        for name in ['runtime.json', 'prompt.json']:
            records = [json.loads((run_dir / 'workers' / str(r) / name).read_text()) for r in range(workers)]
            if any(record != records[0] for record in records):
                raise ValueError(f'Workers disagree on {name}')
            write_json(run_dir / name, records[0])
        temporary = run_dir / 'responses.jsonl.tmp'
        with temporary.open('wb') as output:
            for i in range(len(ds)):
                rank = i % workers
                handles[rank].seek(indices[rank][i])
                output.write(handles[rank].readline())
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(run_dir / 'responses.jsonl')
        result = base.export(ds, cfg, manifest, run_dir, list(range(len(ds))), 'full')
        # Include the parallel protocol in the uploaded, checksummed provenance.
        provenance = json.loads((result / 'provenance.json').read_text())
        provenance['parallel'] = json.loads((run_dir / 'parallel_protocol.json').read_text())
        write_json(result / 'provenance.json', provenance)
        complete = json.loads((result / 'COMPLETE.json').read_text())
        complete['files']['provenance.json'] = base.hash_file(result / 'provenance.json')
        write_json(result / 'COMPLETE.json', complete)
        return result


def launch(cfg_path, root, run_dir, gpu_ids):
    processes, logs = [], []
    try:
        for rank, gpu in enumerate(gpu_ids):
            directory = run_dir / 'workers' / str(rank)
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / 'console.log').open('a')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
            command = [sys.executable, '-m', 'pipeline.sample_sft_pool_runpod',
                       '--config', str(cfg_path), '--run-dir', str(run_dir), '--mode', 'worker',
                       '--rank', str(rank), '--gpus', ','.join(gpu_ids)]
            processes.append(subprocess.Popen(command, cwd=root, env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        last_status = 0
        while True:
            codes = [p.poll() for p in processes]
            failed = next((r for r, c in enumerate(codes) if c not in (None, 0)), None)
            if failed is not None:
                raise RuntimeError(f'Worker {failed} failed ({codes[failed]}). See {run_dir}/workers/{failed}/console.log')
            if all(c == 0 for c in codes):
                break
            if time.monotonic() - last_status >= 30:
                statuses = []
                for r, c in enumerate(codes):
                    path = run_dir / 'workers' / str(r) / 'progress.json'
                    progress = json.loads(path.read_text()) if path.exists() else None
                    statuses.append(f'GPU {gpu_ids[r]}: ' + ('finished' if c == 0 else str(progress or 'starting / generating')))
                print(' | '.join(statuses), flush=True)
                last_status = time.monotonic()
            time.sleep(1)
    finally:
        # Kill whole process groups, including vLLM subprocesses, on failure/interruption.
        for p in processes:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 3
        for p in processes:
            try:
                p.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for p in processes:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--mode', choices=['prepare', 'full', 'upload', 'worker'], required=True)
    args = parser.parse_args()
    gpu_ids = args.gpus.split(',')
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or any(not g.strip() for g in gpu_ids):
        raise ValueError('Select four distinct GPU IDs or UUIDs')
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    cfg = yaml.safe_load(args.config.read_text())
    token = os.environ.get('HF_TOKEN')
    if args.mode == 'worker':
        worker(cfg, root, run_dir, args.rank, len(gpu_ids))
        return
    # Convert termination into cleanup so notebook interruption also stops child groups.
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    with exclusive(run_dir / 'coordinator.lock'):
        protocol(cfg, root, run_dir, len(gpu_ids))
        if args.mode == 'upload':
            base.upload(cfg, root, run_dir, token)
            return
        ds, manifest = base.prepare(cfg, root, run_dir, token)
        if args.mode == 'prepare':
            return
        # Refuse to replace prepared data while any surviving worker is active.
        from contextlib import ExitStack
        with ExitStack() as stack:
            for rank in range(len(gpu_ids)):
                stack.enter_context(exclusive(run_dir / 'workers' / str(rank) / 'worker.lock'))
            ds.save_to_disk(str(run_dir / 'prepared'))
        from huggingface_hub import snapshot_download
        snapshot_download(cfg['model']['repo'], revision=manifest['model_revision'], token=token)
        launch(args.config.resolve(), root, run_dir, gpu_ids)
        merge(ds, cfg, manifest, run_dir, len(gpu_ids))


if __name__ == '__main__':
    main()
