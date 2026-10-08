"""Read-only checkpoint diagnosis. Never edits the Drive source or starts generation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def inspect(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    errors, count, bad = [], 0, 0
    with path.open('rb') as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            digest.update(line)
            count += 1
            try:
                json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                bad += 1
                if len(errors) < 5:
                    errors.append({'offset': offset, 'bytes': len(line),
                                   'error': f'{type(error).__name__}: {error}',
                                   'line_sha256': hashlib.sha256(line).hexdigest()})
    after = path.stat()
    return {'file_bytes': after.st_size, 'lines': count, 'unreadable_lines': bad,
            'sha256': digest.hexdigest(), 'errors': errors,
            'changed_during_read': (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)}


def diagnose(path):
    path = Path(path)
    report = {'drive_first': inspect(path)}
    with tempfile.TemporaryDirectory(prefix='checkpoint-diagnostic-') as directory:
        snapshot = Path(directory) / 'snapshot.jsonl'
        shutil.copyfile(path, snapshot)
        report['local_copy'] = inspect(snapshot)
    report['drive_second'] = inspect(path)
    report['all_reads_identical'] = len({r['sha256'] for r in report.values()}) == 1
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.journal), indent=2), flush=True)
