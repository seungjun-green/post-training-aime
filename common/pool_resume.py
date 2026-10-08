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


def recover_checkpoint(config_path, run_dir):
    """Back up and remove unreadable JSONL records; validate all survivors before replace.

    Run only after stopping all generation sessions that write to this run directory.
    Valid JSON with bad IDs, duplicate indices or invalid annotations is NOT discarded.
    """
    import os
    import shutil
    import uuid
    import yaml
    from common.io import write_json
    from pipeline import sample_sft_pool as sampling

    run_dir = Path(run_dir)
    journal = run_dir / 'responses.jsonl'
    if not journal.exists():
        return None
    def fingerprint():
        stat = journal.stat()
        return stat.st_size, stat.st_mtime_ns
    initial = fingerprint()
    invalid = []
    with journal.open('rb') as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid.append(offset)
    if not invalid:
        return None
    if fingerprint() != initial:
        raise RuntimeError('Checkpoint changed during inspection. Stop all other generation sessions first.')
    recovery_dir = run_dir / 'checkpoint_backups'
    recovery_dir.mkdir(exist_ok=True)
    tag = uuid.uuid4().hex
    backup = recovery_dir / f'responses-{tag}.jsonl'
    shutil.copyfile(journal, backup)
    backup_hash = sampling.hash_file(backup)
    if fingerprint() != initial or sampling.hash_file(journal) != backup_hash:
        raise RuntimeError('Checkpoint changed while backing up. Original journal was not replaced.')
    temporary = run_dir / f'responses-recovered-{tag}.tmp'
    try:
        with backup.open('rb') as source, temporary.open('wb') as output:
            for line in source:
                try:
                    json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                output.write(line if line.endswith(b'\n') else line + b'\n')
            output.flush()
            os.fsync(output.fileno())
        cfg = yaml.safe_load(Path(config_path).read_text())
        root = Path(__file__).resolve().parents[1]
        dataset, _ = sampling.prepare(cfg, root, run_dir, os.environ.get('HF_TOKEN'))
        recovered = sampling.scan_journal(temporary, dataset, cfg['data'])
        # Do not replace any journal that was appended to while validation ran.
        if fingerprint() != initial or sampling.hash_file(journal) != backup_hash:
            raise RuntimeError('Checkpoint changed during validation. Original journal was not replaced.')
        report = {'backup': str(backup), 'backup_sha256': backup_hash,
                  'unreadable_records': len(invalid), 'unreadable_byte_offsets': invalid,
                  'saved_problems': len(recovered), 'remaining_problems': len(dataset)-len(recovered)}
        # Record the backup location before committing the replacement.
        write_json(recovery_dir / f'recovery-{tag}.json', report)
        temporary.replace(journal)
        print(f'Recovered checkpoint: kept {len(recovered):,} completed problems; removed {len(invalid):,} unreadable records.')
        print(f'Original checkpoint backup: {backup}')
        print('Missing problems will be regenerated. No saved scores or responses were invented.')
        return report
    finally:
        temporary.unlink(missing_ok=True)


def generate_with_recovery(config_path, run_dir):
    """Always recover immediately before generation, including standalone cell reruns."""
    import os
    import yaml
    from common.pool_progress import check_resume
    from pipeline import sample_sft_pool as sampling
    check_resume(config_path, run_dir)
    recover_checkpoint(config_path, run_dir)
    cfg = yaml.safe_load(Path(config_path).read_text())
    return sampling.generate(cfg, Path(__file__).resolve().parents[1], Path(run_dir),
                             os.environ.get('HF_TOKEN'), 'full')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--recover', nargs=2, metavar=('CONFIG', 'RUN_DIR'))
    group.add_argument('--generate', nargs=2, metavar=('CONFIG', 'RUN_DIR'))
    args = parser.parse_args()
    if args.generate:
        generate_with_recovery(*args.generate)
    else:
        recover_checkpoint(*args.recover)
