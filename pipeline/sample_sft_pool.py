"""Resumable eight-sample base-model annotation; smoke and full share one journal."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import yaml

from common.io import append_jsonl, digest, write_json, write_jsonl
from eval.final_answer import VERSION as GRADING_VERSION, score_final
from pipeline.sft_pool_quality import clean_rows

COLUMNS = ('responses', 'extracted_answers', 'correct', 'response_tokens', 'finish_reasons', 'num_correct')


def validate_config(cfg):
    if cfg['sampling']['n'] != 8 or cfg['smoke']['rows'] != 50:
        raise ValueError('This notebook requires eight responses and a 50-row smoke test')
    if cfg['sampling']['temperature'] <= 0:
        raise ValueError('Use positive temperature for eight stochastic samples')
    if not 0 < cfg['sampling']['top_p'] <= 1:
        raise ValueError('top_p must be in (0,1]')
    if not 0 < cfg['sampling']['max_new_tokens'] < cfg['engine']['max_model_len']:
        raise ValueError('Invalid context/completion budget')
    if cfg['export']['shard_rows'] < 1:
        raise ValueError('shard_rows must be positive')


def code_hash(root):
    paths = ['pipeline/sample_sft_pool.py', 'common/io.py', 'common/math_text.py',
             'common/english_prompts.py', 'eval/scoring.py', 'eval/batched_generation.py',
             'eval/engines.py', 'common/prompts.py', 'requirements-eval.lock',
             'scripts/setup_eval_runtime.py', 'eval/final_answer.py', 'eval/other_answers.py',
             'pipeline/sft_pool_quality.py', 'configs/sft_pool_quality.yaml']
    return digest({name: (Path(root) / name).read_text() for name in paths})


def load_source(cfg, revision, token):
    from datasets import load_dataset
    spec = cfg['data']
    ds = load_dataset(spec['repo'], name=spec['config'], split=spec['split'], revision=revision, token=token)
    if spec.get('expected_rows') is not None and len(ds) != spec['expected_rows']:
        raise ValueError(f'Expected {spec["expected_rows"]} rows, got {len(ds)}')
    needed = {spec['id_column'], spec['problem_column'], spec['answer_column']}
    if not needed <= set(ds.column_names):
        raise ValueError(f'Missing source columns: {needed-set(ds.column_names)}')
    if set(COLUMNS) & set(ds.column_names):
        raise ValueError('Source already contains annotation columns; use the original pinned source revision')
    ids = [str(x) for x in ds[spec['id_column']]]
    if len(ids) != len(set(ids)) or any(x in {'', 'None'} for x in ids):
        raise ValueError('Missing or duplicate source IDs')
    for row in ds.select_columns([spec['problem_column'], spec['answer_column']]):
        if not isinstance(row[spec['problem_column']], str) or not row[spec['problem_column']].strip():
            raise ValueError('Source has an empty/non-text problem')
        if row[spec['answer_column']] is None or not str(row[spec['answer_column']]).strip():
            raise ValueError('Source has an empty gold answer')
    return ds



def cleaned_source_report(spec, revision, token):
    """Require the cleanup notebook's published train path and report at this commit."""
    from huggingface_hub import DatasetCard, hf_hub_download
    card_path = hf_hub_download(spec['repo'], 'README.md', repo_type='dataset', revision=revision, token=token)
    metadata = DatasetCard(Path(card_path).read_text()).data.to_dict()
    configuration = next((c for c in metadata.get('configs', [])
                          if c['config_name'] == spec['config']), {})
    files = configuration.get('data_files', [])
    train_paths = [entry['path'] for entry in files if isinstance(entry, dict) and entry.get('split') == spec['split']]
    if (len(train_paths) != 1 or not isinstance(train_paths[0], str) or
            not train_paths[0].startswith('cleaning/text-only/') or
            not train_paths[0].endswith('/train-cleaned.parquet')):
        raise ValueError('The selected HF dataset has no published text-cleanup split. '
                         'Finish the cleanup-and-upload notebook first, then rerun with a new RUN_NAME.')
    report_path = str(Path(train_paths[0]).parent / 'summary.json')
    local_report = hf_hub_download(spec['repo'], report_path, repo_type='dataset', revision=revision, token=token)
    report = json.loads(Path(local_report).read_text())
    if (report.get('policy_version') != 'text-only-exclusions-v1' or
            report.get('repo_id') != spec['repo'] or report.get('source_config') != spec['config']):
        raise ValueError('Unexpected cleanup report; refusing to use an unverified source.')
    return report

def select_smoke(count, seed, size=50):
    if count < size:
        raise ValueError('Dataset is too small for smoke selection')
    return sorted(random.Random(seed).sample(range(count), size))



def select_smoke_rows(ds, spec):
    """Replay eligible prior IDs, then deterministically fill to 50 without replacement."""
    preferred = spec.get('preferred_ids', [])
    if len(preferred) != len(set(preferred)) or len(preferred) > spec['rows']:
        raise ValueError('Preferred smoke IDs must be unique and fit the smoke size')
    by_id = {str(row['id']): i for i, row in enumerate(ds)}
    replay = [by_id[x] for x in preferred if x in by_id]
    replay_set = set(replay)
    candidates = [i for i in range(len(ds)) if i not in replay_set]
    needed = spec['rows'] - len(replay)
    if len(candidates) < needed:
        raise ValueError('Dataset is too small for smoke selection')
    additions = random.Random(spec['seed']).sample(candidates, needed)
    return sorted(replay + additions), {
        'requested_previous_ids': preferred,
        'replayed_ids': [x for x in preferred if x in by_id],
        'excluded_previous_ids': [x for x in preferred if x not in by_id],
        'new_ids': [str(ds[i]['id']) for i in additions],
        'note': 'Compare identical question IDs only. Replacements are new random eligible rows.'}

def prepare(cfg, root, run_dir, token):
    from huggingface_hub import HfApi
    validate_config(cfg)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / 'manifest.json'
    signature = digest({'config': cfg, 'code_hash': code_hash(root)})
    api = HfApi(token=token)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['signature'] != signature:
            raise ValueError('Settings or code changed. Choose a new RUN_NAME; saved responses must not mix protocols')
    else:
        revision = api.dataset_info(cfg['data']['repo'], revision=cfg['data']['revision']).sha
        model_revision = api.model_info(cfg['model']['repo'], revision=cfg['model']['revision']).sha
        if model_revision != cfg['model']['revision']:
            raise ValueError('Model must use a full pinned commit')
        manifest = {'signature': signature, 'config': cfg, 'dataset_revision': revision,
                    'model_revision': model_revision, 'code_hash': code_hash(root),
                    'prompt_protocol': 'repository English boxed-answer instruction with pinned native tokenizer template',
                    'grading': GRADING_VERSION}
    cleanup = (cleaned_source_report(cfg['data'], manifest['dataset_revision'], token)
               if cfg['data'].get('require_text_cleanup') else None)
    ds = load_source(cfg, manifest['dataset_revision'], token)
    if manifest.get('source_rows', len(ds)) != len(ds):
        raise ValueError('Pinned source row count changed')
    manifest['source_rows'] = len(ds)
    if cleanup is not None:
        if cleanup['retained_rows'] != len(ds):
            raise ValueError('Cleanup report row count does not match the published train split')
        manifest['source_cleanup'] = cleanup
        print(f'Published text-cleanup verified: {len(ds):,} source rows', flush=True)
    # Hash source content, not the loader's cache fingerprint.
    h = hashlib.sha256()
    for row in ds:
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b'\n')
    source_digest = h.hexdigest()
    if manifest.get('source_digest', source_digest) != source_digest:
        raise ValueError('Pinned source content changed')
    manifest['source_digest'] = source_digest
    manifest['source_features'] = ds.features.to_dict()
    registry = yaml.safe_load((Path(root) / 'configs/sft_pool_quality.yaml').read_text())
    cleaned, decisions, quality = clean_rows(list(ds), registry, cfg['data'])
    from datasets import Dataset, Features, Value
    if not cleaned:
        raise ValueError('No rows passed quality screening')
    ds = Dataset.from_list(cleaned, features=Features({**ds.features,
        'gold_answer_original': Value('string'), 'quality_action': Value('string')}))
    manifest['quality'] = quality
    manifest['prepared_digest'] = digest(cleaned)
    write_json(run_dir / 'quality_summary.json', quality)
    write_jsonl(run_dir / 'quality_decisions.jsonl', decisions)
    selection, selection_report = select_smoke_rows(ds, cfg['smoke'])
    write_json(run_dir / 'smoke_selection_summary.json', selection_report)
    manifest['smoke_indices'] = selection
    write_json(manifest_path, manifest)
    write_jsonl(run_dir / 'smoke_selection.jsonl',
                ({'source_index': i, **ds[i]} for i in selection))
    print(f'Pinned dataset: {manifest["dataset_revision"]}; rows={len(ds)}', flush=True)
    print(f'Model: {cfg["model"]["repo"]}; smoke={len(selection)} rows × 8', flush=True)
    return ds, manifest


def sample_seed(seed, identifier):
    return int(hashlib.sha256(f'{seed}:{identifier}'.encode()).hexdigest()[:8], 16) % (2**31)


def annotate(responses, gold, timeout=5, problem='', audit_output=None, answer_type=None):
    if len(responses) != 8:
        raise ValueError('Expected exactly eight outputs')
    result = {key: [] for key in COLUMNS if key != 'num_correct'}
    errors = []
    for response in responses:
        text, reason, tokens = response['text'], response['finish_reason'], response['token_count']
        if not isinstance(text, str) or reason not in {'stop', 'length'} or type(tokens) is not int or tokens < 0:
            raise ValueError('Malformed generation or unexpected finish reason')
        extracted = ''
        try:
            extracted, correct, decision = score_final(text, gold, problem, timeout=timeout, answer_type=answer_type)
            errors.append(None)
        except Exception as error:
            correct = False
            errors.append(type(error).__name__)
            decision = {'method': 'grading_exception', 'error': type(error).__name__}
        if audit_output is not None:
            audit_output.append(decision)
        result['responses'].append(text)
        result['extracted_answers'].append(extracted)
        result['correct'].append(bool(correct))
        result['response_tokens'].append(tokens)
        result['finish_reasons'].append(reason)
    result['num_correct'] = sum(result['correct'])
    validate_annotation(result)
    return result, errors


def validate_annotation(row):
    for key in COLUMNS[:-1]:
        if not isinstance(row[key], list) or len(row[key]) != 8:
            raise ValueError(f'{key} must contain eight values')
    for key in ['responses', 'extracted_answers', 'finish_reasons']:
        if not all(isinstance(x, str) for x in row[key]):
            raise ValueError(f'{key} must contain strings')
    if not all(type(x) is bool for x in row['correct']):
        raise ValueError('correct must contain bools')
    if not all(type(x) is int and x >= 0 for x in row['response_tokens']):
        raise ValueError('response_tokens must contain nonnegative ints')
    if not set(row['finish_reasons']) <= {'stop', 'length'}:
        raise ValueError('Unexpected finish reason')
    if type(row['num_correct']) is not int or row['num_correct'] != sum(row['correct']):
        raise ValueError('num_correct does not match correct')


def scan_journal(path, dataset, spec):
    """Index byte offsets only, keeping multi-GB response text out of RAM on resume."""
    path = Path(path)
    if not path.exists():
        return {}
    index = {}
    with path.open('r+b') as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not line.endswith(b'\n') and not f.read(1):
                    f.truncate(start)
                    break
                raise ValueError(f'Corrupt checkpoint at offset {start}') from None
            i = entry['source_index']
            if type(i) is not int or not 0 <= i < len(dataset) or i in index:
                raise ValueError('Duplicate or invalid checkpoint index')
            if entry['id'] != str(dataset[i][spec['id_column']]):
                raise ValueError('Checkpoint/source ID mismatch')
            validate_annotation(entry['annotation'])
            index[i] = start
            if not line.endswith(b'\n'):
                f.seek(0, 2)
                f.write(b'\n')
    return index


def qwen_stop_ids(tokenizer):
    """Native base EOS plus the chat-template end-of-turn marker."""
    expected = {'<|endoftext|>': 151643, '<|im_end|>': 151645}
    if tokenizer.eos_token_id != expected['<|endoftext|>']:
        raise ValueError('Unexpected EOS for the pinned Qwen2.5-3B base tokenizer')
    for token, token_id in expected.items():
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f'Unexpected special-token encoding: {token}')
    return list(expected.values())


def make_engine(cfg, manifest, run_dir):
    import torch
    from vllm import LLM

    from common.english_prompts import render_prompt
    if not torch.cuda.is_available():
        raise RuntimeError('Choose a Colab GPU runtime')
    runtime = {'packages': {p: importlib.metadata.version(p) for p in
                           ['vllm', 'torch', 'transformers', 'math-verify', 'datasets',
                            'sympy', 'latex2sympy2-extended', 'antlr4-python3-runtime']},
               'gpu': torch.cuda.get_device_name(0)}
    path = Path(run_dir) / 'runtime.json'
    if path.exists() and json.loads(path.read_text()) != runtime:
        raise ValueError('GPU/software changed; use a new RUN_NAME to avoid mixing runtimes')
    llm = LLM(model=cfg['model']['repo'], revision=manifest['model_revision'],
              tokenizer_revision=manifest['model_revision'], trust_remote_code=False,
              generation_config='vllm', seed=cfg['sampling']['seed'],
              tensor_parallel_size=1, **cfg['engine'])
    tokenizer = llm.get_tokenizer()
    stop_ids = qwen_stop_ids(tokenizer)
    prompt_manifest = {'chat_template': tokenizer.chat_template,
                       'example': render_prompt(tokenizer, 'Compute 1+1.'),
                       'special_tokens': tokenizer.special_tokens_map,
                       'stop_token_ids': stop_ids, 'ignore_eos': False}
    prompt_path = Path(run_dir) / 'prompt.json'
    if prompt_path.exists() and json.loads(prompt_path.read_text()) != prompt_manifest:
        raise ValueError('Tokenizer prompt changed')
    write_json(prompt_path, prompt_manifest)
    write_json(path, runtime)
    return SimpleNamespace(name='vllm', llm=llm, tokenizer=tokenizer,
                           config={**cfg['sampling'], **cfg['engine'], 'stop_token_ids': stop_ids,
                                   'record_stop_diagnostics': True})


def generate(cfg, root, run_dir, token, mode):
    from common.english_prompts import render_prompt
    from eval.batched_generation import generate_problems
    from eval.engines import check_context
    ds, manifest = prepare(cfg, root, run_dir, token)
    run_dir = Path(run_dir)
    selected = manifest['smoke_indices'] if mode == 'smoke' else list(range(len(ds)))
    journal = run_dir / 'responses.jsonl'
    completed = scan_journal(journal, ds, cfg['data'])
    pending = [i for i in selected if i not in completed]
    print(f'{mode}: {len(selected)-len(pending)}/{len(selected)} rows already complete', flush=True)
    if pending:
        engine = make_engine(cfg, manifest, run_dir)
        # Fail before generating anything if even one requested prompt would truncate.
        for i in selected:
            prompt = render_prompt(engine.tokenizer, ds[i][cfg['data']['problem_column']])
            ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
            try:
                check_context(ids, engine.config)
            except ValueError as error:
                raise ValueError(f'Row {i} exceeds context budget. No prompt is truncated; adjust settings in a new run.') from error
        jobs = ({'key': str(ds[i][cfg['data']['id_column']]), 'source_index': i,
                 'problem': ds[i][cfg['data']['problem_column']], 'n': 8,
                 'seed': sample_seed(cfg['sampling']['seed'], ds[i][cfg['data']['id_column']])}
                for i in pending)
        done = len(selected) - len(pending)
        for job, _, responses in generate_problems(engine, jobs, cfg['execution']):
            i = job['source_index']
            grading_audit = []
            annotation, errors = annotate(responses, ds[i][cfg['data']['answer_column']],
                                          cfg['sampling']['verify_timeout_seconds'],
                                          problem=ds[i][cfg['data']['problem_column']], answer_type=ds[i].get('answer_type'), audit_output=grading_audit)
            append_jsonl(journal, {'source_index': i, 'id': job['key'],
                                  'annotation': annotation, 'grading_errors': errors,
                                  'grading_audit': grading_audit,
                                  'generation_diagnostics': [{k: r.get(k) for k in ['stop_reason', 'last_token_id']} for r in responses]})
            done += 1
            print(f'{mode}: {done}/{len(selected)} rows; index={i}; correct={annotation["num_correct"]}/8', flush=True)
    return export(ds, cfg, manifest, run_dir, selected, mode)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def export(dataset, cfg, manifest, run_dir, selected, mode):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Features, List, Value
    run_dir = Path(run_dir)
    index = scan_journal(run_dir / 'responses.jsonl', dataset, cfg['data'])
    if any(i not in index for i in selected):
        raise ValueError('Incomplete generation; rerun to resume before exporting')
    out = run_dir / mode
    out.mkdir(parents=True, exist_ok=True)
    (out / 'COMPLETE.json').unlink(missing_ok=True)
    features = Features({**dataset.features,
                         'responses': List(Value('string'), length=8),
                         'extracted_answers': List(Value('string'), length=8),
                         'correct': List(Value('bool'), length=8),
                         'response_tokens': List(Value('int64'), length=8),
                         'finish_reasons': List(Value('string'), length=8),
                         'num_correct': Value('int64')})
    histogram, finish, tokens, empty, errors, batch, files = Counter(), Counter(), 0, 0, 0, [], []
    extraction_methods, stop_reasons = Counter(), Counter()
    audit_file = (out / 'grading_audit.jsonl').open('w')
    json_output = (out / 'smoke_results.jsonl').open('w') if mode == 'smoke' else None
    def flush():
        filename = f'data/train-{len(files):05d}.parquet'
        path = out / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(batch, schema=features.arrow_schema), path)
        files.append(filename)
        batch.clear()
    try:
        with (run_dir / 'responses.jsonl').open('rb') as journal:
            for i in selected:
                journal.seek(index[i])
                record = json.loads(journal.readline())
                annotation = record['annotation']
                validate_annotation(annotation)
                merged = {**dataset[i], **annotation}
                histogram[annotation['num_correct']] += 1
                finish.update(annotation['finish_reasons'])
                tokens += sum(annotation['response_tokens'])
                empty += sum(x == '' for x in annotation['extracted_answers'])
                errors += sum(x is not None for x in record['grading_errors'])
                extraction_methods.update(x['method'] for x in record.get('grading_audit', []))
                stop_reasons.update(str(x.get('stop_reason')) for x in record.get('generation_diagnostics', []))
                audit_file.write(json.dumps({'id': record['id'], 'grading': record.get('grading_audit', []),
                                            'generation': record.get('generation_diagnostics', [])}) + '\n')
                if json_output:
                    json_output.write(json.dumps(merged, ensure_ascii=False) + '\n')
                batch.append(merged)
                if len(batch) >= cfg['export']['shard_rows']:
                    flush()
            if batch:
                flush()
    finally:
        audit_file.close()
        if json_output:
            json_output.close()
    total = len(selected) * 8
    summary = {'mode': mode, 'rows': len(selected), 'responses': total,
               'num_correct_histogram': {str(i): histogram[i] for i in range(9)},
               'accuracy_per_response': sum(i*c for i, c in histogram.items()) / total,
               'at_least_one_correct_fraction': 1 - histogram[0] / len(selected),
               'finish_reasons': dict(finish), 'truncated_fraction': finish['length'] / total,
               'empty_extractions': empty, 'grading_exceptions': errors,
               'extraction_methods': dict(extraction_methods), 'stop_reasons': dict(stop_reasons),
               'total_response_tokens': tokens, 'mean_response_tokens': tokens / total}
    write_json(out / 'summary.json', summary)
    write_json(out / 'provenance.json', {'run': manifest, 'mode': mode, 'source_indices': selected,
                                        'runtime': json.loads((run_dir / 'runtime.json').read_text()),
                                        'prompt': json.loads((run_dir / 'prompt.json').read_text()),
                                        'response_columns': list(COLUMNS),
                                        'reproducibility': 'per-row seeds and fixed runtime; GPU scheduling may still affect exact text'})
    files += ['summary.json', 'provenance.json', 'grading_audit.jsonl']
    import shutil
    for name in ['quality_summary.json', 'quality_decisions.jsonl', 'smoke_selection_summary.json']:
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, out / name)
            files.append(name)
    if mode == 'smoke':
        files.append('smoke_results.jsonl')
    write_json(out / 'COMPLETE.json', {'signature': manifest['signature'], 'mode': mode,
                                      'rows': len(selected), 'features': features.to_dict(),
                                      'files': {name: hash_file(out / name) for name in files}})
    print(json.dumps(summary, indent=2), flush=True)
    print('Saved result files:', out, flush=True)
    return out


def upload(cfg, root, run_dir, token):
    import pyarrow.parquet as pq
    from datasets import Features
    from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download
    ds, manifest = prepare(cfg, root, run_dir, token)
    run_dir = Path(run_dir)
    out = run_dir / 'full'
    complete = json.loads((out / 'COMPLETE.json').read_text())
    if complete['mode'] != 'full' or complete['rows'] != len(ds) or complete['signature'] != manifest['signature']:
        raise ValueError('Only a complete full run can be uploaded')
    for name, checksum in complete['files'].items():
        if hash_file(out / name) != checksum:
            raise ValueError(f'Changed artifact: {name}')
    # Check all originals, new column types/counts, and row ordering before publication.
    count = 0
    shards = sorted(name for name in complete['files'] if name.endswith('.parquet'))
    features = Features.from_dict(complete['features'])
    for name in shards:
        table = pq.read_table(out / name)
        if not table.schema.equals(features.arrow_schema, check_metadata=False):
            raise ValueError('Unexpected Parquet schema')
        for batch in table.to_batches(max_chunksize=32):
            for row in batch.to_pylist():
                validate_annotation(row)
                if {k: row[k] for k in ds.column_names} != ds[count]:
                    raise ValueError(f'Original row changed at {count}')
                count += 1
    if count != len(ds):
        raise ValueError('Uploaded rows would be incomplete')
    api = HfApi(token=token)
    target = cfg['target']['repo']
    def verify_receipt(receipt):
        for name, checksum in complete['files'].items():
            path = hf_hub_download(target, 'annotated/'+name, repo_type='dataset', revision=receipt['revision'], token=token)
            if hash_file(path) != checksum:
                raise ValueError(f'Upload checksum mismatch: {name}')
        for local, remote in [('README.md', 'README.md'), ('COMPLETE.json', 'annotated/COMPLETE.json')]:
            path = hf_hub_download(target, remote, repo_type='dataset', revision=receipt['revision'], token=token)
            if hash_file(path) != hash_file(out / local):
                raise ValueError(f'Upload checksum mismatch: {remote}')
        receipt['verified'] = True
        write_json(run_dir / 'upload_receipt.json', receipt)
        print('Verified upload:', receipt['url'], flush=True)
        return receipt
    if target == cfg['data']['repo']:
        expected = manifest['dataset_revision']
        current = api.dataset_info(target).sha
        if current != expected:
            receipt_path = run_dir / 'upload_receipt.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt.get('revision') == current and receipt.get('signature') == manifest['signature']:
                    return verify_receipt(receipt)
            raise ValueError('Source/target changed since preparation. Refusing to overwrite newer data')
    else:
        api.create_repo(target, repo_type='dataset', private=True, exist_ok=True)
        expected = api.dataset_info(target).sha
    source_card = hf_hub_download(cfg['data']['repo'], 'README.md', repo_type='dataset',
                                  revision=manifest['dataset_revision'], token=token)
    original_card = DatasetCard(Path(source_card).read_text())
    card_metadata = original_card.data.to_dict()
    # Retain original card/license text, but replace stale original feature metadata.
    card_metadata.pop('dataset_info', None)
    card_metadata['configs'] = [{'config_name': 'default', 'data_files':
                                 [{'split': 'train', 'path': 'annotated/data/*.parquet'}]}]
    card = '---\n' + yaml.safe_dump(card_metadata, sort_keys=False) + '---\n' + original_card.text + '''

## Eight Qwen2.5-3B base responses

Eligible original rows are preserved after audited quality screening. Confirmed gold
corrections retain the prior answer in gold_answer_original; quality_action identifies
changed answers. Quarantined rows and all corrections are in quality_decisions.jsonl.
Added generation columns: responses,
extracted_answers, correct, response_tokens, finish_reasons (eight values each),
and num_correct (0–8). Final boxed answers take priority; explicit unboxed conclusions
and unambiguous final math lines are also supported. Extraction never searches for the
reference value in the reasoning. Math-Verify grades the extracted answer; truncated responses are still graded
but marked `length`. Token counts are generation token IDs, excluding the prompt.
No SFT or RL checkpoint is used. See annotated/provenance.json and summary.json.
The dataset retains its upstream terms; source metadata is recorded in provenance.
'''
    card_path = out / 'README.md'
    card_path.write_text(card)
    operations = [CommitOperationAdd(path_in_repo='annotated/'+name, path_or_fileobj=str(out/name))
                  for name in complete['files']]
    operations += [CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(card_path)),
                   CommitOperationAdd(path_in_repo='annotated/COMPLETE.json', path_or_fileobj=str(out/'COMPLETE.json'))]
    commit = api.create_commit(target, repo_type='dataset', parent_commit=expected,
                               commit_message='Add eight Qwen2.5-3B base responses and Math-Verify grades', operations=operations)
    receipt = {'repo': target, 'revision': commit.oid, 'signature': manifest['signature'],
               'verified': False, 'url': f'https://huggingface.co/datasets/{target}/tree/{commit.oid}'}
    write_json(run_dir / 'upload_receipt.json', receipt)
    return verify_receipt(receipt)


def regrade_smoke(input_path, root, output, cfg):
    """CPU-only diagnosis of old outputs; never import them into the new run journal."""
    source = [json.loads(line) for line in Path(input_path).read_text().splitlines() if line.strip()]
    if len(source) != 50 or len({r['id'] for r in source}) != 50:
        raise ValueError('Expected 50 unique original smoke rows')
    for row in source:
        validate_annotation(row)
    registry = yaml.safe_load((Path(root) / 'configs/sft_pool_quality.yaml').read_text())
    # Restore the saved original only for an already audited correction. Keep the
    # input scores separately, so the comparison uses the actual previous gold.
    previous = {row['id']: row for row in source}
    originals = []
    for row in source:
        original = dict(row)
        if row.get('quality_action') == 'gold_corrected':
            action = registry['reviewed_rows'].get(row['id'], {})
            if (row.get('gold_answer_original') != action.get('expected_gold') or
                    row['gold_answer'] != action.get('corrected_gold')):
                raise ValueError('Cannot restore an unrecognized previous correction')
            original['gold_answer'] = row['gold_answer_original']
        originals.append(original)
    retained, decisions, quality = clean_rows(originals, registry, cfg['data'])
    results, changes = [], []
    old_total = new_total = 0
    for row in retained:
        generated = [{'text': row['responses'][i], 'token_count': row['response_tokens'][i],
                      'finish_reason': row['finish_reasons'][i]} for i in range(8)]
        details = []
        updated, errors = annotate(generated, row['gold_answer'], cfg['sampling']['verify_timeout_seconds'],
                                   problem=row['problem'], audit_output=details, answer_type=row.get('answer_type'))
        old_total += row['num_correct']
        new_total += updated['num_correct']
        changes.append({'id': row['id'], 'old_gold': previous[row['id']]['gold_answer'], 'gold': row['gold_answer'],
                        'old_num_correct': row['num_correct'], 'new_num_correct': updated['num_correct'],
                        'old_correct': row['correct'], 'new_correct': updated['correct'],
                        'old_extractions': row['extracted_answers'], 'new_extractions': updated['extracted_answers'],
                        'grading_audit': details, 'grading_errors': errors})
        results.append({**row, **updated})
    output = Path(output)
    write_jsonl(output / 'smoke_results_regraded.jsonl', results)
    write_jsonl(output / 'quality_decisions.jsonl', decisions)
    write_jsonl(output / 'grading_changes.jsonl', changes)
    summary = {'quality': quality, 'retained_responses': len(results)*8,
               'old_correct_on_retained_rows': old_total, 'regraded_correct_on_retained_rows': new_total,
               'input_sha256': hash_file(input_path), 'new_generation': False,
               'grading_policy_version': GRADING_VERSION,
               'grading_error_count': sum(bool(e) for c in changes for e in c['grading_errors']),
               'changed_correct_flags': sum(a != b for c in changes for a, b in zip(c['old_correct'], c['new_correct'])),
               'note': 'Diagnostic regrading only; excluded rows reduce the sample below 50. Fresh GPU smoke uses 50 eligible rows.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--mode', choices=['prepare', 'smoke', 'full', 'upload', 'regrade'], required=True)
    parser.add_argument('--input-smoke')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(__file__).resolve().parents[1]
    token = os.environ.get('HF_TOKEN')
    if args.mode == 'regrade':
        if not args.input_smoke:
            parser.error('--input-smoke is required for regrade')
        regrade_smoke(args.input_smoke, root, Path(args.run_dir) / 'regrade', cfg)
    elif args.mode == 'prepare':
        prepare(cfg, root, args.run_dir, token)
    elif args.mode == 'upload':
        upload(cfg, root, args.run_dir, token)
    else:
        generate(cfg, root, args.run_dir, token, args.mode)


if __name__ == '__main__':
    main()
