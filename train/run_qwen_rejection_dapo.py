"""Start fresh DAPO from the verified original Qwen 3B rejection-sampling SFT epoch 4."""

import json
from pathlib import Path
import struct

from common.io import digest


def select_rejection_model(config, kind, sft_root, output_root):
    if kind != 'sft':
        raise ValueError('This experiment requires --model-kind sft and the selected epoch_4')
    if not sft_root:
        raise ValueError('Set --sft-root to the five-epoch rejection-sampling run')
    root, output = Path(sft_root).resolve(), Path(output_root).resolve()
    if root == output or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Use a separate OUTPUT_ROOT outside the SFT run')
    spec = config['sft']
    if spec['epoch'] != 4 or spec['run_name'] != 'sft_qwen25_3b_base_self_rft':
        raise ValueError('Expected rejection-sampling epoch 4')
    expected_relative = f"checkpoints/stage1/{spec['run_name']}/epoch_4"
    if spec['relative_checkpoint'] != expected_relative:
        raise ValueError('Unexpected SFT checkpoint path')
    path = (root / expected_relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('SFT checkpoint must be inside SFT_ROOT')
    logs = root / 'logs/stage1' / spec['run_name']
    marker = json.loads((path / 'stage1_checkpoint.json').read_text())
    manifest = json.loads((logs / 'run_manifest.json').read_text())
    report = json.loads((logs / 'data_report.json').read_text())
    previous = manifest['config']
    if (manifest['identity'] != digest({k: v for k, v in manifest.items() if k != 'identity'})
            or marker['run_identity'] != manifest['identity']
            or manifest['data_report_digest'] != digest(report)):
        raise ValueError('SFT checkpoint, manifest and data report identities disagree')
    if (previous['run_name'] != spec['run_name'] or previous['model'] != config['model']
            or previous['training']['num_train_epochs'] != 5 or previous.get('continuation')
            or previous['data']['columns'] != {'question': 'problem', 'answer': 'response'}
            or previous['data']['hf_source'] != spec['hf_source']
            or previous.get('target_format') != 'native_eos_completion'
            or previous.get('tokenizer') or config.get('tokenizer') or config.get('generation')):
        raise ValueError('Expected Qwen base rejection-sampling SFT with native-EOS response targets')
    if (marker['epoch'] != 4 or marker['global_step'] != 7068
            or marker['global_step'] != 4 * report['optimizer_steps_per_epoch']
            or report['kept_rows'] != 28267):
        raise ValueError('Expected completed epoch 4 at step 7068 with 28267 SFT rows')
    state = json.loads((path / 'trainer_state.json').read_text())
    if state['epoch'] != 4 or state['global_step'] != marker['global_step']:
        raise ValueError('SFT Trainer state disagrees with the epoch-4 completion marker')
    for filename in ['tokenizer.json', 'tokenizer_config.json', 'config.json', 'generation_config.json']:
        if not (path / filename).is_file() or not (path / filename).stat().st_size:
            raise FileNotFoundError(f'Missing SFT model/tokenizer file: {path / filename}')
    model = json.loads((path / 'config.json').read_text())
    generation = json.loads((path / 'generation_config.json').read_text())
    tokenizer = json.loads((path / 'tokenizer_config.json').read_text())
    for settings in [model, generation]:
        eos = settings.get('eos_token_id')
        if (eos if isinstance(eos, list) else [eos]) != [151643] or settings.get('pad_token_id') != 151643:
            raise ValueError('Saved model must preserve native Qwen EOS/PAD 151643')
    if model.get('model_type') != 'qwen2':
        raise ValueError('Expected a Qwen2 model')
    for name in ['eos_token', 'pad_token']:
        value = tokenizer.get(name)
        if (value.get('content') if isinstance(value, dict) else value) != '<|endoftext|>':
            raise ValueError('Saved tokenizer must preserve native Qwen EOS/PAD')
    template_path = path / 'chat_template.jinja'
    template = template_path.read_text() if template_path.is_file() else tokenizer.get('chat_template')
    if not isinstance(template, str) or not template.strip():
        raise ValueError('Missing saved Qwen chat template')
    if digest(template) != manifest['tokenizer']['chat_template_digest']:
        raise ValueError('Saved chat template differs from the SFT training manifest')
    tokenizer_json = json.loads((path / 'tokenizer.json').read_text())
    vocabulary = dict(tokenizer_json['model']['vocab'])
    vocabulary.update({entry['content']: entry['id'] for entry in tokenizer_json.get('added_tokens', [])})
    if digest(vocabulary) != manifest['tokenizer']['tokenizer_vocab_digest']:
        raise ValueError('Saved vocabulary differs from the SFT training manifest')
    # Read shard headers only: detect truncated Drive saves without loading 3B weights.
    index = path / 'model.safetensors.index.json'
    shards = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
    if not shards:
        raise ValueError('SFT checkpoint has no weight shards')
    for name in shards:
        shard = path / name
        if shard.parent != path or not shard.is_file():
            raise FileNotFoundError(f'Missing SFT weight shard: {shard}')
        size = shard.stat().st_size
        with shard.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f'Incomplete SFT weight shard: {shard}')
            header_size = struct.unpack('<Q', prefix)[0]
            if header_size > min(size - 8, 100_000_000):
                raise ValueError(f'Incomplete SFT weight header: {shard}')
            header = json.loads(stream.read(header_size))
        ends = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
        if not ends or size != 8 + header_size + max(ends):
            raise ValueError(f'Incomplete SFT weight data: {shard}')
    return str(path), {'kind': 'sft', 'path': str(path), 'marker': marker,
                       'parent_run_identity': manifest['identity'], 'sft_dataset': spec['hf_source']}


def main():
    from train.run_dapo import main as run

    run(model_selector=select_rejection_model)


if __name__ == '__main__':
    main()
