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



def scan_checkpoint_readonly(path, dataset, spec):
    """Validate and index with the same read-only I/O used by recovery.

    Google Drive is a remote mount. Scanning must not open a healthy journal for
    random writes. Repairs are an explicit backed-up operation before generation.
    """
    from pipeline.sample_sft_pool import validate_annotation
    path = Path(path)
    if not path.exists():
        return {}
    index = {}
    with path.open('rb') as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f'Unreadable checkpoint at byte {offset}, record length {len(line)} bytes: '
                                 f'{type(error).__name__}: {error}. '
                                 'Cause unconfirmed; preserve the journal and run the read-only diagnostic.') from error
            i = entry['source_index']
            if type(i) is not int or not 0 <= i < len(dataset) or i in index:
                raise ValueError('Duplicate or invalid checkpoint index')
            if entry['id'] != str(dataset[i][spec['id_column']]):
                raise ValueError('Checkpoint/source ID mismatch')
            validate_annotation(entry['annotation'])
            index[i] = offset
    return index

def recover_checkpoint(config_path, run_dir):
    """Repair a local snapshot, then verify the replacement read back from Drive."""
    import os
    import shutil
    import tempfile
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
    # Inspect the exact byte snapshot that will be repaired. Do not use a separate
    # preliminary scan of the remote mount to decide whether recovery is needed.
    with tempfile.TemporaryDirectory(prefix='pool-recovery-') as directory:
        snapshot = Path(directory) / 'original.jsonl'
        repaired = Path(directory) / 'repaired.jsonl'
        shutil.copyfile(journal, snapshot)
        original_hash = sampling.hash_file(snapshot)
        if fingerprint() != initial or sampling.hash_file(journal) != original_hash:
            raise RuntimeError('Checkpoint changed during snapshot. Stop other sessions; original was not replaced.')
        invalid, needs_newline = [], False
        with snapshot.open('rb') as source, repaired.open('wb') as output:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                needs_newline = not line.endswith(b'\n')
                try:
                    json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    invalid.append({'offset': offset, 'bytes': len(line), 'error': str(error)})
                    continue
                output.write(line if line.endswith(b'\n') else line + b'\n')
            output.flush()
            os.fsync(output.fileno())
        if not invalid and not needs_newline:
            print('Checkpoint JSON integrity: passed (local snapshot matches Drive).', flush=True)
            return None
        recovery_dir = run_dir / 'checkpoint_backups'
        recovery_dir.mkdir(exist_ok=True)
        tag = uuid.uuid4().hex
        backup = recovery_dir / f'responses-{tag}.jsonl'
        shutil.copyfile(snapshot, backup)
        if sampling.hash_file(backup) != original_hash:
            raise RuntimeError('Backup verification failed; original journal was not replaced.')
        cfg = yaml.safe_load(Path(config_path).read_text())
        root = Path(__file__).resolve().parents[1]
        dataset, _ = sampling.prepare(cfg, root, run_dir, os.environ.get('HF_TOKEN'))
        recovered = scan_checkpoint_readonly(repaired, dataset, cfg['data'])
        repaired_hash = sampling.hash_file(repaired)
        temporary = run_dir / f'responses-recovered-{tag}.tmp'
        try:
            shutil.copyfile(repaired, temporary)
            if sampling.hash_file(temporary) != repaired_hash:
                raise RuntimeError('Staged repair failed Drive readback verification; original was not replaced.')
            if fingerprint() != initial or sampling.hash_file(journal) != original_hash:
                raise RuntimeError('Checkpoint changed during validation. Original journal was not replaced.')
            report = {'backup': str(backup), 'backup_sha256': original_hash,
                      'repaired_sha256': repaired_hash,
                      'unreadable_records': len(invalid),
                      'unreadable_byte_offsets': [r['offset'] for r in invalid], 'errors': invalid,
                      'saved_problems': len(recovered), 'remaining_problems': len(dataset)-len(recovered),
                      'verified': False}
            report_path = recovery_dir / f'recovery-{tag}.json'
            write_json(report_path, report)
            temporary.replace(journal)
            # Validate the actual published journal, not just a temporary file.
            if sampling.hash_file(journal) != repaired_hash:
                raise RuntimeError(f'Repaired journal readback differs from the local repair. Original backup: {backup}')
            if scan_checkpoint_readonly(journal, dataset, cfg['data']) != recovered:
                raise RuntimeError(f'Repaired journal index differs on Drive. Original backup: {backup}')
            report['verified'] = True
            write_json(report_path, report)
            print(f'Recovery VERIFIED: kept {len(recovered):,} completed problems; removed {len(invalid):,} unreadable records.', flush=True)
            print(f'Original checkpoint backup: {backup}', flush=True)
            print(f'Remaining problems to generate: {len(dataset)-len(recovered):,}', flush=True)
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
    # Storage adapter only: retain the existing pinned sampling/grading code and
    # manifest signature. Both generation and export use this read-only scanner.
    original_scan = sampling.scan_journal
    sampling.scan_journal = scan_checkpoint_readonly
    try:
        return sampling.generate(cfg, Path(__file__).resolve().parents[1], Path(run_dir),
                                 os.environ.get('HF_TOKEN'), 'full')
    finally:
        sampling.scan_journal = original_scan


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
