"""Read-only collection of Qwen experiment logs for the final project write-up.

Standard library only; embedded in the CPU Colab collection notebook.
"""

import csv
import fnmatch
import hashlib
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENTS = 'LG-AIME-Qwen25-3B-Experiments'
RUNS = [
    ('base_rl', 'base-rl-native-eos', 'dapo_qwen25_3b_pretrained_base', 'rl'),
    ('kimi_sft', 'base-sft-v2-short', 'sft_qwen25_3b_base_s1_kimi', 'sft'),
    ('rejection_sft', 'base-rejection-sampling-sft', 'sft_qwen25_3b_base_self_rft', 'sft'),
    ('distill_sft', 'base-7b-distillation-sft', 'sft_qwen25_3b_base_7b_distill', 'sft'),
    ('distill_sft_epochs6_7', 'base-7b-distillation-sft-epoch5-to7',
     'sft_qwen25_3b_base_7b_distill_continue', 'sft'),
    ('rejection_rl', 'base-rejection-sampling-sft-rl', 'dapo_qwen25_3b_rejection_epoch4_sft', 'rl'),
    ('distill_rl', 'base-7b-distillation-sft-rl', 'dapo_qwen25_3b_7b_distill_epoch5_sft', 'rl'),
    ('instruct_rl', '../LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix', 'dapo_qwen25_3b_base', 'rl'),
    ('instruct_rl_original', '../LG-AIME-DAPO-Qwen25-3B-MiniBatch300', 'dapo_qwen25_3b_base', 'rl'),
]
TRAIN_PATTERNS = [
    'steps.jsonl', 'run_manifest.json', 'data_report.json', 'resolved_config.json',
    'config.yaml', 'training_summary.json', 'train_metrics.json', 'attempts.jsonl',
    'attempt_*.json', 'abandoned_steps_*.jsonl', 'epoch_*_train_metrics.json',
]
EVAL_NAMES = {
    'epoch_summary.json', 'epoch_summary.csv', 'summary_table.json', 'summary_table.csv',
    'comparison_table.json', 'comparison_table.csv', 'comparison.json', 'comparison.csv',
    'subject_task_breakdowns.csv', 'metrics.json', 'metrics.csv',
}
NOTES = """# Qwen training log collection

Raw files are preserved under drive/ with their original paths relative to MyDrive.
collection_manifest.json lists source paths, file checksums, missing files, errors and
size exclusions. run_inventory.csv lists available step counts/ranges and metric fields.
No curves have been smoothed or merged, and missing metrics are not reconstructed.

Included: seven trained variants, distillation continuation epochs 6–7, the original
instruct-RL recovery parent, training settings/reports, checkpoint JSON metadata,
and available math evaluation summaries (including per-epoch results).
B0 and I0 have no training histories; their combined math evaluation tables are included
when present. RunPod-only transfer results are NOT on Drive and are not fetched here;
use the already-exported Hugging Face result artifacts separately for those analyses.

Instruct RL: use original steps 1–40, then MemoryFix steps 41 onward. Original steps
after 40 were abandoned. Keep the two runs separate until recovery provenance is checked.
Distillation: original epochs 1–5 and continuation epochs 6–7 have separate run manifests;
do not assume their step counters or learning-rate schedules restart/continue identically.
Abandoned-step files are diagnostic history, not additional successful updates.
RL rollout statistics can repeat across optimizer updates sharing a rollout: use
rollout_first_update/cached_rollout to avoid counting the same generated samples twice.

Excluded: model weights, optimizer/scheduler/RNG states, tokenizer files, dataset
snapshots, raw rollouts, generated evaluation answers, secrets, and console logs.
Only small checkpoint marker/trainer_state JSON files are read from checkpoint folders.
Drive sources are never modified. Files exceeding the selected size limit are reported.
If a run is missing, this ZIP is a partial collection, not proof that it never ran.
"""


def walk_files(root):
    """Avoid raw rollout trees and symlinks while scanning the selected directories."""
    if not root.is_dir() or root.is_symlink():
        return
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = sorted(f for f in folders if f not in {'rollouts', '__pycache__', '.cache'}
                            and not (Path(directory) / f).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if not path.is_symlink():
                yield path


def step_summary(raw):
    fields, steps, records, malformed = set(), [], 0, 0

    def visit(value, prefix=''):
        for key, item in value.items():
            name = f'{prefix}.{key}' if prefix else key
            if isinstance(item, dict):
                visit(item, name)
            else:
                fields.add(name)

    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('Expected object')
        except ValueError:
            malformed += 1
            continue
        records += 1
        visit(record)
        if isinstance(record.get('step'), (int, float)):
            steps.append(record['step'])
    return {'records': records, 'step_min': min(steps) if steps else None,
            'step_max': max(steps) if steps else None, 'duplicate_step_records': len(steps) - len(set(steps)),
            'malformed_lines': malformed, 'metric_fields': sorted(fields)}


def collect(drive_root, output_dir, *, max_file_mb=100, notify=print):
    drive_root = Path(drive_root).resolve()
    if not drive_root.is_dir():
        raise FileNotFoundError(f'Mount Google Drive first: {drive_root}')
    if max_file_mb <= 0:
        raise ValueError('max_file_mb must be positive')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = output_dir / f'qwen_training_logs_{stamp}.zip'
    manifest = {'created_utc': stamp, 'drive_root': str(drive_root), 'files': [], 'runs': [],
                'missing': [], 'skipped': [], 'errors': [], 'max_file_mb': max_file_mb}
    seen = set()
    limit = int(max_file_mb * 1024**2)
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def add(path, *, run=None, steps=False):
            if path in seen:
                return
            seen.add(path)
            try:
                if not path.resolve().is_relative_to(drive_root) or path.is_symlink():
                    manifest['skipped'].append({'path': str(path), 'reason': 'outside root or symlink'})
                    return
                size = path.stat().st_size
                if size > limit:
                    manifest['skipped'].append({'path': str(path), 'reason': 'size limit', 'bytes': size})
                    return
                # Bound reads too, in case a log grows while the notebook is collecting it.
                with path.open('rb') as stream:
                    raw = stream.read(limit + 1)
                if len(raw) > limit:
                    manifest['skipped'].append({'path': str(path), 'reason': 'size limit'})
                    return
                name = 'drive/' + path.relative_to(drive_root).as_posix()
                archive.writestr(name, raw)
                manifest['files'].append({'source': str(path), 'archive_path': name, 'bytes': len(raw),
                                          'sha256': hashlib.sha256(raw).hexdigest()})
                if steps:
                    run.update(step_summary(raw))
                    run['steps_collected'] = True
            except (OSError, UnicodeError, ValueError) as error:
                manifest['errors'].append({'path': str(path), 'error': str(error)})

        def eval_summaries(root):
            for path in walk_files(root):
                if ('smoke' in path.relative_to(root).parts or 'hf_exports' in path.parts):
                    continue
                if (path.name in EVAL_NAMES or (path.suffix == '.json' and path.parent.name == 'results'
                                               and 'full' in path.relative_to(root).parts)):
                    add(path)

        for key, folder, name, kind in RUNS:
            root = (drive_root / EXPERIMENTS / folder).resolve()
            logs = root / ('logs/stage1' if kind == 'sft' else 'logs') / name
            run = {'key': key, 'kind': kind, 'root': str(root), 'steps_collected': False}
            manifest['runs'].append(run)
            for expected in [logs / 'steps.jsonl', logs / 'run_manifest.json']:
                if not expected.is_file():
                    manifest['missing'].append(str(expected))
            for path in walk_files(logs):
                if any(fnmatch.fnmatchcase(path.name, pattern) for pattern in TRAIN_PATTERNS):
                    add(path, run=run, steps=path == logs / 'steps.jsonl')
            for path in walk_files(root / 'inputs' / name):
                if path.name in {'training.yaml', 'manifest.json', 'data_report.json', 'report.json'}:
                    add(path)
            checkpoints = root / ('checkpoints/stage1' if kind == 'sft' else 'checkpoints') / name
            for path in walk_files(checkpoints):
                if path.name in {'trainer_state.json', 'stage1_checkpoint.json', 'dapo_checkpoint.json'}:
                    add(path)
            eval_summaries(root / 'eval')
            notify(f'{key}: ' + (f"{run.get('records', 0):,} step-log records collected"
                                if run['steps_collected'] else 'MISSING / unreadable step log; see report'))

        for folder in ['eval-base-instruct-rl-amc-math-temp0', 'eval-sft-rl-step300-amc-math-temp0']:
            eval_summaries(drive_root / EXPERIMENTS / folder)
        manifest['status'] = ('partial' if manifest['missing'] or manifest['errors'] or manifest['skipped']
                              or any(r.get('malformed_lines', 0) for r in manifest['runs']) else 'complete')
        archive.writestr('README.md', NOTES)
        archive.writestr('collection_manifest.json', json.dumps(manifest, indent=2))
        table = io.StringIO()
        columns = ['key', 'kind', 'root', 'steps_collected', 'records', 'step_min', 'step_max',
                   'duplicate_step_records', 'malformed_lines', 'metric_fields']
        writer = csv.DictWriter(table, fieldnames=columns)
        writer.writeheader()
        for run in manifest['runs']:
            writer.writerow({**run, 'metric_fields': json.dumps(run.get('metric_fields', []))})
        archive.writestr('run_inventory.csv', table.getvalue())
    with zipfile.ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP integrity check failed: {bad}')
    return destination, manifest
