"""Selection, provenance checks and immutable identities for all nine Qwen variants."""

import hashlib
import json
import struct
from pathlib import Path

from common.io import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def weight_files(path):
    index = path / 'model.safetensors.index.json'
    names = (sorted(set(json.loads(index.read_text())['weight_map'].values()))
             if index.is_file() else ['model.safetensors'])
    if not names:
        raise ValueError('No model weight shards')
    for name in names:
        file = path / name
        if file.parent != path or not file.is_file():
            raise FileNotFoundError(f'Missing weight shard: {file}')
        with file.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f'Incomplete weight shard: {file}')
            size = struct.unpack('<Q', prefix)[0]
            if not 0 < size <= min(100_000_000, file.stat().st_size - 8):
                raise ValueError(f'Incomplete weight header: {file}')
            header = json.loads(stream.read(size))
        ends = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
        if not ends or file.stat().st_size != 8 + size + max(ends):
            raise ValueError(f'Incomplete weight data: {file}')
    return [path / name for name in names]


def validate_checkpoint(path, entry, key, config):
    path = Path(path).resolve()
    # Exact experiment/epoch folder names prevent accidentally swapping variants.
    if path.parts[-2:] != Path(entry['path']).parts[-2:]:
        raise ValueError(f'Wrong checkpoint directory for {key}: {path}')
    is_sft = entry['kind'] == 'sft'
    marker_name = 'stage1_checkpoint.json' if is_sft else 'dapo_checkpoint.json'
    marker = json.loads((path / marker_name).read_text())
    run_name = path.parent.name
    train_root = path.parents[3 if is_sft else 2]
    log = train_root / ('logs/stage1' if is_sft else 'logs') / run_name / 'run_manifest.json'
    manifest = json.loads(log.read_text())
    if (marker['run_identity'] != manifest['identity'] or
            manifest['identity'] != digest({k: v for k, v in manifest.items() if k != 'identity'})):
        raise ValueError(f'Checkpoint/manifest identity mismatch: {key}')
    if marker['global_step'] != entry['step'] or (is_sft and marker['epoch'] != entry['epoch']):
        raise ValueError(f'Wrong epoch/step for {key}')
    if entry.get('expected_identity', marker['run_identity']) != marker['run_identity']:
        raise ValueError(f'Unexpected training run identity: {key}')
    cfg = manifest['config']
    expected_model = config['models']['instruct' if key == 'instruct_rl' else 'base']
    if any(cfg['model'][k] != expected_model[k] for k in ['repo', 'revision']):
        raise ValueError(f'Wrong starting model for {key}')
    if is_sft:
        if cfg['run_name'] != run_name:
            raise ValueError('SFT run name mismatch')
    elif (cfg['run_name_prefix'] + '_' + cfg['model_kind'] != run_name
          or cfg['training']['max_steps'] != 300):
        raise ValueError('RL run name or step budget mismatch')
    if 'parent_epoch' in entry:
        source = manifest['source']
        parent_key = key.replace('_rl', '_sft')
        expected_parent = config['models'][parent_key]
        if (source['kind'] != 'sft' or source['marker']['epoch'] != entry['parent_epoch']
                or source['marker']['global_step'] != expected_parent['step']
                or Path(source['path']).parts[-2:] != Path(expected_parent['path']).parts[-2:]
                or source['parent_run_identity'] != source['marker']['run_identity']):
            raise ValueError('Wrong SFT parent for RL checkpoint')
    model = json.loads((path / 'config.json').read_text())
    generation = json.loads((path / 'generation_config.json').read_text())
    tokenizer = json.loads((path / 'tokenizer_config.json').read_text())
    if model.get('model_type') != 'qwen2':
        raise ValueError('Expected Qwen2 architecture')
    for settings in [model, generation]:
        eos = settings.get('eos_token_id')
        if entry['eos_token_id'] not in (eos if isinstance(eos, list) else [eos]):
            raise ValueError(f'Saved EOS differs from the trained model: {key}')
    eos = tokenizer.get('eos_token')
    eos = eos.get('content') if isinstance(eos, dict) else eos
    expected_eos = '<|im_end|>' if entry['eos_token_id'] == 151645 else '<|endoftext|>'
    if eos != expected_eos:
        raise ValueError(f'Unexpected tokenizer EOS for {key}')
    template_file = path / 'chat_template.jinja'
    template = template_file.read_text() if template_file.exists() else tokenizer.get('chat_template')
    if isinstance(template, list):
        template = {t['name']: t['template'] for t in template}
    if isinstance(template, dict):
        template = template.get('default')
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f'Missing saved default chat template: {key}')
    if not (path / 'tokenizer.json').is_file():
        raise FileNotFoundError(f'Missing tokenizer.json: {path}')
    weight_files(path)
    return marker


def model_fingerprint(run):
    if run['revision']:
        return digest({'repo': run['model'], 'revision': run['revision']})
    path = Path(run['model'])
    files = set(weight_files(path))
    files.update(p for p in path.iterdir() if p.is_file() and p.suffix in {'.json', '.jinja', '.model'})
    # Exclude optimizer/RNG tensors; these are not inference inputs.
    return digest({p.name: file_hash(p) for p in sorted(files)})


def prepare_models(config, experiment_root, instruct_rl_root, output_root, selected=None):
    selected = list(config['models']) if selected is None else selected
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(config['models']):
        raise ValueError('Select unique known model keys')
    runs = []
    for key in selected:
        entry = config['models'][key]
        if 'repo' in entry:
            run = {'key': key, 'label': entry['label'], 'model': entry['repo'], 'revision': entry['revision']}
        else:
            path = (Path(instruct_rl_root) / 'checkpoints' / Path(entry['path']).parent.name / 'checkpoint-300'
                    if key == 'instruct_rl' else Path(experiment_root) / entry['path'])
            path = path.resolve()
            output = Path(output_root).resolve()
            if output == path or path.is_relative_to(output) or output.is_relative_to(path):
                raise ValueError('Evaluation output cannot contain or overwrite a checkpoint')
            marker = validate_checkpoint(path, entry, key, config)
            run = {'key': key, 'label': entry['label'], 'model': str(path), 'revision': None,
                   'checkpoint_marker': marker}
        selection = Path(output_root) / 'models' / key / 'selection.json'
        if selection.exists() and json.loads(selection.read_text()) != run:
            raise ValueError(f'Model selection changed for {key}; use a new OUTPUT_ROOT')
        write_json(selection, run)
        runs.append(run)
    return runs
