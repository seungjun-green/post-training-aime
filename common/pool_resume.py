"""Notebook resume controls without changing the sampling/checkpoint protocol."""
import copy
import json
from pathlib import Path


def resolve_run_config(requested, run_dir, mode='auto'):
    """Resume only the named run, restoring its manifest configuration verbatim."""
    if mode not in {'auto', 'resume', 'new'}:
        raise ValueError('RUN_MODE must be auto, resume, or new')
    run_dir = Path(run_dir)
    manifest_path = run_dir / 'manifest.json'
    if mode == 'new' and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('RUN_MODE=new requires an empty run directory. Choose a new RUN_NAME; nothing was overwritten.')
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        config = copy.deepcopy(saved['config'])
        print(f'Resuming saved run: {run_dir.name}')
        print(f"Pinned input: {saved['dataset_revision']}")
        print('Using the saved model, sampling, batching and data settings; settings above apply only to new runs.')
        return config
    if mode == 'resume':
        available = sorted(p.parent.name for p in run_dir.parent.glob('*/manifest.json'))
        names = ', '.join(available[-10:]) or '(none found)'
        raise ValueError(f'No saved manifest for {run_dir.name}. Set RUN_NAME to the original run. Saved runs: {names}')
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Run directory contains files but no manifest. Choose a new RUN_NAME or recover the original manifest; nothing was overwritten.')
    print(f'Starting new run: {run_dir.name}')
    return copy.deepcopy(requested)


def show_checkpoint_status(run_dir):
    """Read-only journal count; generation performs full source/annotation validation."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    total = manifest['quality']['eligible_rows']
    journal = run_dir / 'responses.jsonl'
    seen = set()
    interrupted_tail = False
    if journal.exists():
        with journal.open('rb') as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if not line.endswith(b'\n') and not handle.read(1):
                        interrupted_tail = True
                        break
                    raise ValueError('Corrupt checkpoint journal; generation stopped before modifying it') from None
                index = row['source_index']
                if type(index) is not int or not 0 <= index < total or index in seen:
                    raise ValueError('Duplicate or invalid checkpoint index')
                seen.add(index)
    print(f'Saved problems: {len(seen):,}/{total:,} | Remaining: {total-len(seen):,}')
    print('Generation validates and skips saved problems, then exports and uploads when complete.')
    if interrupted_tail:
        print('An interrupted final checkpoint will be repaired automatically; its problem will be regenerated.')
    return {'saved': len(seen), 'total': total, 'remaining': total-len(seen)}
