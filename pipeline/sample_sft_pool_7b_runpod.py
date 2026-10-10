"""Independent GPU workers for the Math-7B 3B-score-0/1/2 follow-up."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal

import yaml

from common.io import write_json
from pipeline import sample_sft_pool_7b as seven
from pipeline.sample_sft_pool_runpod import exclusive, launch


def assignment(selected, rank, workers):
    if not 0 <= rank < workers:
        raise ValueError('Invalid worker rank')
    return selected[rank::workers]


def parallel_config(cfg, root, workers):
    # Driver changes are part of the ordinary manifest signature and remote run path.
    return {**cfg, 'parallel': {
        'workers': workers, 'partition': 'selected-position-modulo-v1',
        'driver_hash': seven.hash_file(Path(root) / 'pipeline/sample_sft_pool_7b_runpod.py'),
        'launcher_hash': seven.hash_file(Path(root) / 'pipeline/sample_sft_pool_runpod.py')}}


def worker(cfg, root, run_dir, rank, workers):
    from datasets import load_from_disk
    directory = run_dir / 'workers' / str(rank)
    with exclusive(directory / 'worker.lock'):
        ds = load_from_disk(str(run_dir / 'prepared'))
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        if manifest['config'] != cfg:
            raise ValueError('Worker settings/code disagree with the prepared manifest')
        selected = assignment(manifest['selected_indices'], rank, workers)
        seven.generate_selected(ds, manifest, cfg, directory, selected)


def merge(ds, cfg, manifest, run_dir, workers):
    with ExitStack() as stack:
        indices, handles, active = {}, {}, []
        for rank in range(workers):
            directory = run_dir / 'workers' / str(rank)
            stack.enter_context(exclusive(directory / 'worker.lock'))
            expected = assignment(manifest['selected_indices'], rank, workers)
            index = seven.journal_index(directory, ds, expected)
            if set(index) != set(expected):
                raise ValueError(f'Worker {rank} is incomplete')
            indices[rank] = index
            if expected:
                active.append(rank)
                handles[rank] = stack.enter_context((directory / 'responses_7b.jsonl').open('rb'))
        for name in ['runtime.json', 'prompt.json']:
            records = [json.loads((run_dir / 'workers' / str(r) / name).read_text()) for r in active]
            if records:
                if any(record != records[0] for record in records):
                    raise ValueError(f'Workers disagree on {name}')
                write_json(run_dir / name, records[0])
        temporary = run_dir / 'responses_7b.jsonl.tmp'
        with temporary.open('wb') as output:
            for position, i in enumerate(manifest['selected_indices']):
                rank = position % workers
                handles[rank].seek(indices[rank][i])
                output.write(handles[rank].readline())
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(run_dir / 'responses_7b.jsonl')
        return seven.export(ds, manifest, cfg, run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--mode', choices=['prepare', 'full', 'upload', 'worker'], required=True)
    args = parser.parse_args()
    gpu_ids = [x.strip() for x in args.gpus.split(',')]
    if len(gpu_ids) not in (3, 4) or len(set(gpu_ids)) != len(gpu_ids) or any(not x for x in gpu_ids):
        raise ValueError('Select three or four distinct GPU IDs or UUIDs')
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    cfg = parallel_config(yaml.safe_load(args.config.read_text()), root, len(gpu_ids))
    token = os.environ.get('HF_TOKEN')
    if args.mode == 'worker':
        worker(cfg, root, run_dir, args.rank, len(gpu_ids))
        return
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    with exclusive(run_dir / 'coordinator.lock'):
        if args.mode == 'upload':
            # Do not publish while any orphaned GPU worker still writes checkpoints.
            with ExitStack() as stack:
                for rank in range(len(gpu_ids)):
                    stack.enter_context(exclusive(run_dir / 'workers' / str(rank) / 'worker.lock'))
                seven.upload(cfg, root, run_dir, token)
            return
        ds, manifest = seven.prepare(cfg, root, run_dir, token)
        if args.mode == 'prepare':
            return
        with ExitStack() as stack:
            for rank in range(len(gpu_ids)):
                stack.enter_context(exclusive(run_dir / 'workers' / str(rank) / 'worker.lock'))
            ds.save_to_disk(str(run_dir / 'prepared'))
        if manifest['selected_indices']:
            from huggingface_hub import snapshot_download
            snapshot_download(seven.MODEL, revision=seven.MODEL_REVISION, token=token,
                              allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
            launch(args.config.resolve(), root, run_dir, gpu_ids,
                   module='pipeline.sample_sft_pool_7b_runpod')
        merge(ds, cfg, manifest, run_dir, len(gpu_ids))


if __name__ == '__main__':
    main()
