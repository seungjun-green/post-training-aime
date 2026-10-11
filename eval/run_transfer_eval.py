"""Greedy, resumable ARC/MMLU-Pro/non-math BBH/IFEval evaluation for nine Qwen models."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import yaml

from common.io import append_jsonl, digest, read_jsonl, write_json, write_jsonl
from eval.batched_generation import generate_problems
from eval.engines import VLLMEngine
from eval.transfer_data import prepare_data, smoke_documents
from eval.transfer_models import model_fingerprint, prepare_models
from eval.transfer_scoring import score, summarize

ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = [
    'common/__init__.py', 'common/io.py', 'common/process.py', 'common/prompts.py',
    'common/english_prompts.py', 'eval/__init__.py', 'eval/engines.py',
    'eval/batched_generation.py', 'eval/transfer_data.py', 'eval/transfer_models.py',
    'eval/transfer_scoring.py', 'eval/run_transfer_eval.py',
    'configs/qwen_transfer_eval.yaml', 'scripts/setup_eval_runtime.py',
    'scripts/setup_transfer_runtime.py', 'requirements-eval.lock', 'requirements-transfer-extra.txt',
]


def bundle_files():
    return CORE_FILES + [str(p.relative_to(ROOT)) for directory in
        ['instruction_following_eval', 'bbh', 'mmlu_pro', 'nltk_data']
        for p in sorted((ROOT / 'third_party' / directory).rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts] + ['third_party/transfer_sources.json', 'third_party/TRANSFER_NOTICE.md']


def code_digest():
    from eval.transfer_models import file_hash
    return digest({name: file_hash(ROOT / name) for name in bundle_files()})


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    g = config['generation']
    if (g['temperature'] != 0 or g['top_p'] != 1 or g['seed'] != 42
            or not 0 < g['max_new_tokens'] < g['max_model_len']
            or g['generation_config'] != 'auto' or g['trust_remote_code']):
        raise ValueError('Require greedy decoding, fixed seed, saved EOS, and a valid context budget')
    if set(config['bbh_tasks']) & set(config['bbh_excluded']):
        raise ValueError('BBH inclusion/exclusion overlap')
    return config


def runtime_metadata():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Connect a CUDA GPU runtime')
    gpu = torch.cuda.get_device_properties(0)
    return {'hardware': {'name': gpu.name, 'total_memory_bytes': gpu.total_memory},
            'packages': {name: importlib.metadata.version(name) for name in
                         ['torch', 'vllm', 'transformers', 'huggingface-hub', 'pyarrow',
                          'nltk', 'langdetect', 'immutabledict', 'absl-py', 'regex']}}


def render_prompt(tokenizer, prompt):
    # No task-independent math instruction, no added tags, and no IFEval suffix.
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True)


def tokenizer_identity(tokenizer):
    if not tokenizer.chat_template:
        raise ValueError('Model has no saved chat template')
    return {'template': digest(tokenizer.chat_template), 'vocabulary': digest(tokenizer.get_vocab()),
            'special_tokens': tokenizer.special_tokens_map, 'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_model(run, config, docs, data_manifest, output_root, *, smoke=False,
              engine_factory=VLLMEngine, metadata_factory=runtime_metadata,
              fingerprint=model_fingerprint, generator=generate_problems):
    selected = smoke_documents(docs) if smoke else docs
    root = Path(output_root) / 'models' / run['key'] / ('smoke' if smoke else 'full')
    identity = {'run': run, 'model_digest': fingerprint(run), 'config': config,
                'data_digest': data_manifest['documents_digest'], 'code_digest': code_digest(),
                'mode': 'smoke' if smoke else 'full', 'selected_ids': [d['id'] for d in selected],
                **metadata_factory()}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError('Evaluation inputs/runtime changed; use a new OUTPUT_ROOT')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Output artifacts exist without a manifest')
    else:
        write_json(manifest_path, identity)
    lookup = {d['id']: d for d in selected}
    journal = root / 'responses.jsonl'
    previous = read_jsonl(journal, repair_tail=True)
    saved = {}
    for row in previous:
        if (row['id'] in saved or row['id'] not in lookup
                or row['input_digest'] != digest(lookup[row['id']])):
            raise ValueError('Resume journal contains duplicate, changed, or unknown input')
        if row['finish_reason'] not in ['stop', 'length'] or not isinstance(row['text'], str):
            raise ValueError('Invalid saved response')
        saved[row['id']] = row
    remaining = [d for d in selected if d['id'] not in saved]
    print(f'{run["label"]}: {len(saved)}/{len(selected)} responses already saved', flush=True)
    engine = None
    if remaining:
        engine = engine_factory(run['model'], config['generation'], run['revision'])
        tokenizer = tokenizer_identity(engine.tokenizer)
        if tokenizer['eos_token_id'] != config['models'][run['key']]['eos_token_id']:
            raise ValueError('Loaded tokenizer EOS differs from selected checkpoint')
        tokenizer_path = root / 'tokenizer_identity.json'
        if tokenizer_path.exists() and json.loads(tokenizer_path.read_text()) != tokenizer:
            raise ValueError('Tokenizer changed during resume')
        write_json(tokenizer_path, tokenizer)
    elif not (root / 'tokenizer_identity.json').is_file():
        raise ValueError('Completed responses are missing tokenizer identity')
    all_records, all_metrics = [], []
    for benchmark in ['arc_challenge', 'mmlu_pro', 'bbh_nonmath', 'ifeval']:
        subset = [d for d in selected if d['benchmark'] == benchmark]
        jobs = [{'key': d['id'], 'problem': d['prompt'], 'n': 1, 'seed': 42}
                for d in subset if d['id'] not in saved]
        done = len(subset) - len(jobs)
        print(f'{benchmark}: {done}/{len(subset)} problems', flush=True)
        if jobs:
            for job, prompt, responses in generator(engine, jobs, config['execution'], prompt_renderer=render_prompt):
                if len(responses) != 1 or responses[0]['finish_reason'] not in ['stop', 'length']:
                    raise ValueError('Expected one completed greedy response')
                doc = lookup[job['key']]
                row = {'id': doc['id'], 'input_digest': digest(doc), 'formatted_prompt': prompt, **responses[0]}
                # Save raw text before scoring: scorer interruption never loses generation.
                append_jsonl(journal, row)
                saved[doc['id']] = row
                done += 1
                print(f'{benchmark}: {done}/{len(subset)} problems', flush=True)
        records = [{**doc, **saved[doc['id']], 'score': score(doc, saved[doc['id']]['text'])} for doc in subset]
        metrics = summarize(records)
        write_jsonl(root / benchmark / 'generations.jsonl', records)
        write_json(root / benchmark / 'metrics.json', metrics)
        all_records.extend(records)
        all_metrics.extend(metrics)
        for metric in metrics:
            if metric['Subject/task'] == 'overall':
                print(f'{benchmark} {metric["Metric"]}: {metric["Accuracy (%)"]:.2f}%', flush=True)
    if len(all_records) != len(selected):
        raise ValueError('Incomplete evaluation')
    result = {'identity': identity, 'tokenizer': json.loads((root / 'tokenizer_identity.json').read_text()),
              'completed': len(all_records), 'metrics': all_metrics}
    write_json(root / 'results.json', result)
    write_csv(root / 'metrics.csv', all_metrics)
    return result


def comparison(config, output_root, selected=None):
    root = Path(output_root)
    keys = list(config['models']) if selected is None else selected
    wide, detailed, reference, pending = [], [], None, []
    for key in keys:
        path = root / 'models' / key / 'full/results.json'
        if not path.is_file():
            pending.append(key)
            continue
        result = json.loads(path.read_text())
        identity = result['identity']
        if identity['mode'] != 'full' or identity['config'] != config or identity['run']['key'] != key:
            raise ValueError('Results do not match requested protocol/model')
        selection = json.loads((root / 'models' / key / 'selection.json').read_text())
        if selection != identity['run'] or result['completed'] != len(identity['selected_ids']):
            raise ValueError('Incomplete or mismatched result')
        comparable = {k: identity[k] for k in ['config', 'data_digest', 'code_digest', 'selected_ids', 'packages']}
        if reference is not None and comparable != reference:
            raise ValueError('Cannot compare changed data, protocol, code or packages')
        reference = comparable
        row = {'Model': config['models'][key]['label'], 'Key': key}
        for metric in result['metrics']:
            detailed.append({'Model': row['Model'], 'Key': key, **metric})
            if metric['Subject/task'] in ['overall', 'non_math_overall', 'task_macro']:
                names = {'arc_challenge': 'ARC-Challenge', 'mmlu_pro': 'MMLU-Pro',
                         'bbh_nonmath': 'BBH non-math', 'ifeval': 'IFEval'}
                detail = (metric['Metric'].replace('_', ' ') if metric['Benchmark'] == 'ifeval'
                          else metric['Subject/task'].replace('_', ' '))
                column = f'{names[metric["Benchmark"]]} {detail} (%)'
                row[column] = metric['Accuracy (%)']
        wide.append(row)
    write_json(root / 'comparison.json', {'completed_models': len(wide), 'pending_models': pending,
                                        'rows': wide, 'breakdowns': detailed})
    if wide:
        write_csv(root / 'comparison.csv', wide)
        write_csv(root / 'subject_task_breakdowns.csv', detailed)
    print(f'Full evaluation models complete: {len(wide)}/{len(keys)}. Pending: {pending}', flush=True)
    return wide, detailed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/qwen_transfer_eval.yaml'))
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--experiment-root', default='/content/drive/MyDrive/LG-AIME-Qwen25-3B-Experiments')
    parser.add_argument('--instruct-rl-root', default='/content/drive/MyDrive/LG-AIME-DAPO-Qwen25-3B-MiniBatch300-MemoryFix')
    parser.add_argument('--model-key')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.summarize:
        comparison(config, args.output_root)
        return
    if args.prepare_only:
        docs, manifest = prepare_data(config, args.output_root)
        print(json.dumps({'counts': manifest['counts'], 'per_model': len(docs),
                          'nine_models': 9 * len(docs), 'smoke_per_model': len(smoke_documents(docs))}, indent=2))
        return
    if args.model_key not in config['models']:
        parser.error('Choose --model-key, --prepare-only or --summarize')
    runs = prepare_models(config, args.experiment_root, args.instruct_rl_root, args.output_root, [args.model_key])
    docs, manifest = prepare_data(config, args.output_root)
    run_model(runs[0], config, docs, manifest, args.output_root, smoke=args.smoke)


if __name__ == '__main__':
    main()
