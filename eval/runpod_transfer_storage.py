"""RunPod storage preflight and selective, one-way checkpoint copies from Google Drive."""

import json
import re
import shutil
import subprocess
from pathlib import Path

GIB = 2**30
# Plenty of headroom for Python/CUDA wheels, HF base/instruct weights, compilation,
# data, raw/scored responses, and temporary downloads. No optimizer files are copied.
RESERVE_GIB = 60
CHECKPOINT_FILTERS = [
    '+ /model*.safetensors', '+ /model.safetensors.index.json',
    '+ /config.json', '+ /generation_config.json', '+ /tokenizer*',
    '+ /special_tokens_map.json', '+ /added_tokens.json', '+ /vocab.json',
    '+ /merges.txt', '+ /chat_template.jinja', '+ /chat_templates/**',
    '+ /stage1_checkpoint.json', '+ /dapo_checkpoint.json', '- **',
]


def check_space(path, *, additional_bytes=0, reserve_gib=RESERVE_GIB):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    required = additional_bytes + reserve_gib * GIB
    if usage.free < required:
        raise RuntimeError(f'{path}: {usage.free / GIB:.1f} GiB free; need at least '
                           f'{required / GIB:.1f} GiB for remaining copies and workspace. '
                           'The 30 GB container disk is insufficient. Use a filesystem with '
                           'about 200 GB capacity and point STORAGE_ROOT at it. A container '
                           'disk can be used, but its files do not survive Pod stop/restart.')
    return {'free_gib': usage.free / GIB, 'required_gib': required / GIB,
            'total_gib': usage.total / GIB}


def drive_copy_plan(config, storage_root, remote='gdrive:', selected=None):
    """remote is a named rclone remote whose root is My Drive."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+:', remote):
        raise ValueError('Use a named rclone remote such as gdrive: with its root at My Drive')
    keys = list(config['models']) if selected is None else selected
    if not keys or len(set(keys)) != len(keys) or set(keys) - set(config['models']):
        raise ValueError('Select unique known model keys')
    root = Path(storage_root).resolve() / 'models'
    plan = []
    for key in keys:
        entry = config['models'][key]
        if 'path' not in entry:
            continue
        path = Path(entry['path'])
        if key == 'instruct_rl':
            relative = path.relative_to('/content/drive/MyDrive')
        else:
            relative = Path('LG-AIME-Qwen25-3B-Experiments') / path
        local = root / relative
        is_sft = entry['kind'] == 'sft'
        parent_root = relative.parents[3 if is_sft else 2]
        manifest = parent_root / ('logs/stage1' if is_sft else 'logs') / relative.parent.name / 'run_manifest.json'
        plan.append({'key': key, 'source': remote + relative.as_posix(), 'destination': str(local),
                     'manifest_source': remote + manifest.as_posix(),
                     'manifest_destination': str(root / manifest)})
    return plan


def filter_args():
    return [argument for rule in CHECKPOINT_FILTERS for argument in ['--filter', rule]]


def copy_commands(item, rclone='rclone'):
    return [
        [rclone, 'copy', item['source'], item['destination'], *filter_args(),
         '--checksum', '--transfers', '4', '--checkers', '8', '--stats', '30s'],
        [rclone, 'copyto', item['manifest_source'], item['manifest_destination'], '--checksum'],
    ]


def copy_checkpoints(config, storage_root, remote, *, selected=None, rclone='rclone', runner=subprocess.run):
    plan = drive_copy_plan(config, storage_root, remote, selected)
    # Ask Drive for exact inference-file sizes before spending time copying weights.
    missing_bytes = 0
    for item in plan:
        result = runner([rclone, 'size', item['source'], '--json', *filter_args()],
                        check=True, capture_output=True, text=True)
        remote_size = json.loads(result.stdout)['bytes']
        local = Path(item['destination'])
        # Only count known inference files; optimizer files cannot lower this bound.
        present = sum(p.stat().st_size for p in local.rglob('*') if p.is_file() and (
            p.name.startswith('model') and p.name.endswith('.safetensors')))
        missing_bytes += max(0, remote_size - present)
    print('Storage preflight:', check_space(storage_root, additional_bytes=missing_bytes), flush=True)
    for item in plan:
        print('Copy:', item['key'], '=>', item['destination'], flush=True)
        for command in copy_commands(item, rclone):
            runner(command, check=True)
    return plan


def gpu_inventory(gpu_ids):
    if not 1 <= len(gpu_ids) <= 4 or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Choose one to four distinct GPU IDs/UUIDs')
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    import csv
    cards = [dict(zip(['index', 'uuid', 'name', 'memory_mib', 'driver'],
                       [v.strip() for v in row], strict=True)) for row in csv.reader(result.stdout.splitlines())]
    selected = []
    for gpu in gpu_ids:
        matches = [card for card in cards if gpu in (card['index'], card['uuid'])]
        if len(matches) != 1:
            raise ValueError(f'GPU {gpu} is not visible to nvidia-smi')
        card = matches[0]
        if 'RTX PRO 6000' not in card['name'] or float(card['memory_mib']) < 90 * 1024:
            raise ValueError(f'Expected a 96 GB RTX PRO 6000, got {card["name"]}: {card["memory_mib"]} MiB')
        selected.append(card)
    if len({card['uuid'] for card in selected}) != len(gpu_ids):
        raise ValueError('GPU IDs resolve to duplicate physical devices')
    return selected
