"""Validate and compare the two SFT-initialized Qwen DAPO checkpoint-300 models."""

import json
from pathlib import Path

import yaml

from common.io import digest, write_json
from eval.answer_only import BUNDLE_FILES as EVALUATOR_FILES
from train.dapo_resume import check_checkpoint

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = EVALUATOR_FILES + [
    'eval/qwen_sft_rl.py', 'configs/qwen_sft_rl_eval.yaml',
    'configs/dapo_qwen25_3b_7b_distill.yaml', 'configs/dapo_qwen25_3b_rejection.yaml',
    'train/__init__.py', 'train/dapo_resume.py',
]
LABELS = {
    'distill_rl': '7B-distillation SFT epoch 5 + RL step 300',
    'rejection_rl': '3B rejection-sampling SFT epoch 4 + RL step 300',
}
TOTALS = {'amc23': 40, 'math_500': 500}


def checkpoint_source(root, key, spec):
    if key not in LABELS:
        raise ValueError('Unknown SFT + RL variant')
    expected = yaml.safe_load((ROOT / spec['variants'][key]).read_text())
    name = expected['run_name_prefix'] + '_sft'
    root = Path(root).resolve()
    checkpoint = root / 'checkpoints' / name / 'checkpoint-300'
    marker, _ = check_checkpoint(checkpoint, require_training_state=False)
    manifest = json.loads((root / 'logs' / name / 'run_manifest.json').read_text())
    config, source = manifest['config'], manifest['source']
    if (marker['run_identity'] != manifest['identity']
            or manifest['identity'] != digest({k: v for k, v in manifest.items() if k != 'identity'})):
        raise ValueError('Checkpoint and RL manifest identities disagree')
    if (marker['global_step'] != 300 or marker.get('model_kind') != 'sft'
            or config.get('model_kind') != 'sft' or config['training']['max_steps'] != 300
            or config['run_name_prefix'] != expected['run_name_prefix']):
        raise ValueError('Expected final checkpoint-300 of the selected SFT + RL run')
    if (config['model'] != expected['model']
            or any(config['model'][k] != v for k, v in spec['base_model'].items())
            or config.get('sft') != expected['sft']
            or config.get('tokenizer') or config.get('generation')):
        raise ValueError('Wrong SFT source or native Qwen model settings')
    parent = source['marker']
    expected_step = 11780 if key == 'distill_rl' else 7068
    if (source.get('kind') != 'sft' or parent['epoch'] != expected['sft']['epoch']
            or parent['global_step'] != expected_step
            or source.get('parent_run_identity') != parent['run_identity']
            or source.get('sft_dataset') != expected['sft']['hf_source']
            or Path(source['path']).parts[-4:] != Path(expected['sft']['relative_checkpoint']).parts):
        raise ValueError('RL provenance does not match the requested SFT epoch/dataset')
    # Original SFT files and RL optimizer files are not required to evaluate saved weights.
    model = json.loads((checkpoint / 'config.json').read_text())
    generation = json.loads((checkpoint / 'generation_config.json').read_text())
    tokenizer = json.loads((checkpoint / 'tokenizer_config.json').read_text())
    if model.get('model_type') != 'qwen2':
        raise ValueError('Expected Qwen2 model weights')
    for settings in [model, generation]:
        eos = settings.get('eos_token_id')
        if (eos if isinstance(eos, list) else [eos]) != [151643] or settings.get('pad_token_id') != 151643:
            raise ValueError('Checkpoint must preserve native Qwen EOS/PAD 151643')
    for field in ['eos_token', 'pad_token']:
        value = tokenizer.get(field)
        if (value.get('content') if isinstance(value, dict) else value) != '<|endoftext|>':
            raise ValueError('Saved tokenizer must preserve native EOS/PAD')
    template_path = checkpoint / 'chat_template.jinja'
    template = template_path.read_text() if template_path.is_file() else tokenizer.get('chat_template')
    if isinstance(template, list):
        template = {item['name']: item['template'] for item in template}
    if isinstance(template, dict):
        template = template.get('default')
    if not isinstance(template, str) or not template.strip():
        raise ValueError('Missing saved default chat template')
    if not (checkpoint / 'tokenizer.json').is_file() or not (checkpoint / 'tokenizer.json').stat().st_size:
        raise FileNotFoundError('Missing saved tokenizer.json')
    return checkpoint, marker


def prepare_runs(spec, train_roots, output_root):
    if set(train_roots) != set(LABELS) or set(spec['variants']) != set(LABELS):
        raise ValueError('Select both distillation and rejection-sampling RL runs')
    output = Path(output_root).resolve()
    for root in train_roots.values():
        root = Path(root).resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError('Use a separate comparison OUTPUT_ROOT outside training folders')
    runs = []
    for key in LABELS:
        checkpoint, marker = checkpoint_source(train_roots[key], key, spec)
        runs.append({'key': key, 'model': str(checkpoint), 'revision': None,
                     'checkpoint_marker': marker, 'run_name': f'qwen_{key}_step300',
                     'output_root': str(output / key)})
    path = output / 'model_selection.json'
    if path.exists() and json.loads(path.read_text()) != runs:
        raise ValueError('Selected checkpoints changed; use a new OUTPUT_ROOT')
    write_json(path, runs)
    return runs


def result_path(run):
    return Path(run['output_root']) / 'full/results' / (run['run_name'] + '.json')


def comparison_rows(runs, config):
    results, rows = [], []
    for run in runs:
        result = json.loads(result_path(run).read_text())
        if (result['mode'] != 'full' or result['model'] != run['model']
                or result['model_revision'] != run['revision']
                or result['config'] != config or result['run_name'] != run['run_name']):
            raise ValueError('Saved result does not match selected checkpoint/settings')
        if results:
            for key in ['execution', 'protocol_digest', 'dataset_suite', 'selected_ids',
                        'packages', 'hardware', 'runner_digest', 'bundle_digest', 'chat_template_digest']:
                if result[key] != results[0][key]:
                    raise ValueError(f'Comparison changed {key}')
        results.append(result)
        for benchmark, total in TOTALS.items():
            metric = result['metrics'][benchmark]
            if (metric['problems'] != total or metric['responses'] != total
                    or metric['samples_per_problem'] != 1):
                raise ValueError(f'Incomplete full evaluation: {run["key"]} / {benchmark}')
            rows.append({'Model': LABELS[run['key']],
                         'Benchmark': 'AMC 2023' if benchmark == 'amc23' else 'MATH-500',
                         'Problems': total, 'Correct': round(metric['avg@1'] * total),
                         'Accuracy (%)': 100 * metric['avg@1'],
                         'Mean tokens': metric['response_length_tokens']['all']})
    return rows
