"""Prepare and validate a separate epoch-5 to epoch-7 distillation continuation."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time

import yaml

from common.io import digest, read_jsonl, write_jsonl
from train.answer_only import prepare_run
from train.qwen_self_rft_data import BUNDLE_FILES as SHARED_FILES
from train.qwen_sft_data import latest_checkpoint
from train.stage1_qwen_self_rft import load_config
from train.stage1_sft import validate_resume

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = SHARED_FILES + [
    'configs/sft_qwen25_3b_7b_distill.yaml',
    'train/qwen_distill_continuation.py', 'train/qwen_continuation_trainer.py',
    'train/stage1_qwen_distill_continue.py', 'train/qwen_distill_continue_epochs.py',
]


def training_code_digest():
    return digest({name: (ROOT / name).read_text() for name in BUNDLE_FILES})


def expected_parent_config():
    return yaml.safe_load((ROOT / 'configs/sft_qwen25_3b_7b_distill.yaml').read_text())


def validate_parent(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != 'epoch_5' or checkpoint.parents[1].name != 'stage1':
        raise ValueError('Select the completed distillation epoch_5 checkpoint')
    name, root = checkpoint.parent.name, checkpoint.parents[3]
    logs = root / 'logs/stage1' / name
    manifest = json.loads((logs / 'run_manifest.json').read_text())
    report = json.loads((logs / 'data_report.json').read_text())
    config = manifest['config']
    if manifest['identity'] != digest({k: v for k, v in manifest.items() if k != 'identity'}):
        raise ValueError('Parent manifest identity is inconsistent')
    if manifest['data_report_digest'] != digest(report):
        raise ValueError('Parent data report differs from its manifest')
    expected = expected_parent_config()
    expected.update(output_root=str(root), run_name=name)
    for key in ['path', 'expected_rows', 'content_digest', 'ids_digest']:
        expected['data'][key] = config['data'][key]
    if config != expected:
        raise ValueError('Parent settings differ from the five-epoch 7B distillation experiment')
    if latest_checkpoint(checkpoint.parent, 5) != checkpoint:
        raise ValueError('Parent must be the latest complete epoch_5')
    marker = json.loads((checkpoint / 'stage1_checkpoint.json').read_text())
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    steps = report['optimizer_steps_per_epoch'] * 5
    if (marker != {'run_identity': manifest['identity'], 'epoch': 5, 'global_step': steps}
            or state['epoch'] != 5 or state['global_step'] != steps
            or state['max_steps'] != steps or state['num_train_epochs'] != 5):
        raise ValueError('Parent checkpoint is not a completed five-epoch run')
    return config, manifest, report


def continuation_config(parent, checkpoint, identity, output_root, run_name):
    root = Path(output_root).resolve()
    parent_root = Path(parent['output_root']).resolve()
    if root == parent_root or root.is_relative_to(parent_root) or parent_root.is_relative_to(root):
        raise ValueError('Use a separate continuation output root outside the parent run')
    config = deepcopy(parent)
    config.update(output_root=str(root), run_name=run_name)
    config['training'].update(num_train_epochs=7, learning_rate=1e-6, warmup_ratio=0.0)
    config['continuation'] = {
        'parent_checkpoint': str(Path(checkpoint).resolve()), 'parent_run_identity': identity,
        'start_epoch': 6, 'additional_epochs': 2,
        'scheduler': 'new_cosine_no_warmup_preserve_optimizer',
    }
    config['data']['path'] = str(root / 'inputs' / run_name / 'dataset.jsonl')
    return config


def prepare_continuation(checkpoint, output_root, run_name):
    parent, manifest, _ = validate_parent(checkpoint)
    config = continuation_config(parent, checkpoint, manifest['identity'], output_root, run_name)
    from train.sft_data import load_rows

    rows, _ = load_rows(parent)
    source = parent['data']['hf_source']
    if len(rows) != source['expected_rows'] or digest(rows) != source['selected_content_digest']:
        raise ValueError('Parent snapshot differs from the pinned distillation dataset')
    with tempfile.TemporaryDirectory(prefix='qwen-continuation-') as directory:
        template = Path(directory) / 'training.yaml'
        template.write_text(yaml.safe_dump(config, sort_keys=False))
        # prepare_run copies and verifies the same rows/IDs into the new run.
        return prepare_run(parent['data']['path'], template, output_root, run_name)


def validate_config(path):
    config = load_config(path)
    checkpoint = config['continuation']['parent_checkpoint']
    parent, manifest, report = validate_parent(checkpoint)
    expected = continuation_config(parent, checkpoint, manifest['identity'],
                                   config['output_root'], config['run_name'])
    if config != expected:
        raise ValueError('Continuation must preserve parent settings except the agreed epoch/LR schedule')
    return config, manifest, report


def validate_prepared_report(report, parent_manifest, parent_report):
    comparable, previous = deepcopy(report), deepcopy(parent_report)
    comparable.pop('total_optimizer_steps')
    previous.pop('total_optimizer_steps')
    comparable['dataset']['path'] = previous['dataset']['path']
    if comparable != previous or report['tokenizer'] != parent_manifest['tokenizer']:
        raise ValueError('Prepared data/tokenizer differs from the parent training run')


def select_resume(config, log_dir, checkpoint_dir, identity):
    """Resume our latest epoch, or retry from the immutable parent before epoch 6 saves."""
    selected = latest_checkpoint(checkpoint_dir, 7)
    if selected:
        if selected.name not in {'epoch_6', 'epoch_7'}:
            raise ValueError('Unexpected epoch in continuation output')
        validate_resume(checkpoint_dir, log_dir, str(selected), identity)
        return str(selected)
    manifest_path = log_dir / 'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['identity'] != identity:
            raise ValueError('Continuation config, code, data or runtime changed')
        records = read_jsonl(log_dir / 'steps.jsonl', repair_tail=True)
        if records:
            write_jsonl(log_dir / f'abandoned_steps_{time.time_ns()}.jsonl', records)
            write_jsonl(log_dir / 'steps.jsonl', [])
    elif any(checkpoint_dir.iterdir()):
        raise ValueError('Unexpected checkpoint files without a continuation manifest')
    return config['continuation']['parent_checkpoint']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-checkpoint', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    print('Continuation config:', prepare_continuation(
        args.parent_checkpoint, args.output_root, args.run_name))


if __name__ == '__main__':
    main()
