"""Add eight Math-7B-Instruct samples to rows with a 3B pass count of 0, 1, or 2."""
import argparse
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import yaml

from common.io import append_jsonl, digest, write_json
from eval.final_answer import VERSION as GRADING_VERSION
from pipeline.sample_sft_pool import (
    COLUMNS, annotate, code_hash, hash_file, sample_seed, scan_journal, validate_annotation,
)

MODEL = 'Qwen/Qwen2.5-Math-7B-Instruct'
MODEL_REVISION = 'ef9926d75ab1d54532f6a30dd5e760355eb9aa4d'
NEW_COLUMNS = tuple('7B_' + key for key in COLUMNS)
SYSTEM_PROMPT = r'Please reason step by step, and put your final answer within \boxed{}.'


def render_prompt(tokenizer, problem):
    if not tokenizer.chat_template:
        raise ValueError('The instruct tokenizer must have its native chat template')
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': problem}],
        tokenize=False, add_generation_prompt=True)


def stop_ids(tokenizer):
    if tokenizer.eos_token_id != 151645:
        raise ValueError('Math-7B-Instruct must use <|im_end|> as EOS (151645)')
    for name, number in [('<|im_end|>', 151645), ('<|endoftext|>', 151643)]:
        if tokenizer.encode(name, add_special_tokens=False) != [number]:
            raise ValueError(f'Unexpected special token: {name}')
    return [151645, 151643]


def select_rows(ds):
    required = {'id', 'problem', 'gold_answer', *COLUMNS}
    if not required <= set(ds.column_names):
        raise ValueError(f'Missing 3B/source columns: {required - set(ds.column_names)}')
    if set(NEW_COLUMNS) & set(ds.column_names):
        raise ValueError('Source already has 7B columns. Resume the original RUN_NAME instead.')
    seen, selected = set(), []
    for i, row in enumerate(ds.select_columns(['id', 'problem', 'gold_answer', 'num_correct'])):
        identifier = str(row['id'])
        if row['id'] is None or not identifier.strip() or identifier in seen:
            raise ValueError('Missing or duplicate source ID')
        seen.add(identifier)
        score = row['num_correct']
        if type(score) is not int or not 0 <= score <= 8:
            raise ValueError(f'Invalid original num_correct at {identifier}')
        if score in (0, 1, 2):
            if not isinstance(row['problem'], str) or not row['problem'].strip():
                raise ValueError(f'Empty problem at {identifier}')
            if row['gold_answer'] is None or not str(row['gold_answer']).strip():
                raise ValueError(f'Empty gold answer at {identifier}')
            selected.append(i)
    return selected


def prepare(cfg, root, run_dir, token):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    if cfg['model'] != {'repo': MODEL, 'revision': MODEL_REVISION}:
        raise ValueError('This workflow is pinned to Qwen2.5-Math-7B-Instruct')
    if cfg['sampling']['n'] != 8 or cfg['sampling']['temperature'] <= 0:
        raise ValueError('Eight stochastic responses are required')
    if not 0 < cfg['sampling']['top_p'] <= 1:
        raise ValueError('top_p must be in (0,1]')
    if not 0 < cfg['sampling']['max_new_tokens'] < cfg['engine']['max_model_len'] <= 4096:
        raise ValueError('Use the native 4096-token context and a smaller response cap')
    if cfg['export']['shard_rows'] < 1:
        raise ValueError('shard_rows must be positive')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    request_signature = digest({'config': cfg, 'shared_code': code_hash(root),
                                'workflow': hash_file(Path(root) / 'pipeline/sample_sft_pool_7b.py')})
    path = run_dir / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['request_signature'] != request_signature:
            raise ValueError('Code/settings changed. Use a new RUN_NAME.')
    else:
        revision = HfApi(token=token).dataset_info(cfg['data']['repo'], revision='main').sha
        manifest = {'request_signature': request_signature, 'dataset_revision': revision,
                    'model_revision': MODEL_REVISION, 'config': cfg, 'grading': GRADING_VERSION}
        manifest['signature'] = digest(manifest)
    ds = load_dataset(cfg['data']['repo'], name='default', split='train',
                      revision=manifest['dataset_revision'], token=token)
    selected = select_rows(ds)
    if path.exists() and (manifest['source_rows'] != len(ds) or manifest['selected_indices'] != selected):
        raise ValueError('Pinned input or selection changed')
    manifest.update(source_rows=len(ds), selected_indices=selected)
    write_json(path, manifest)
    print(f'Source: {len(ds):,} rows; 3B num_correct in [0,1,2]: {len(selected):,}; '
          f'new responses: {len(selected)*8:,}', flush=True)
    print('Pinned HF source:', manifest['dataset_revision'], flush=True)
    return ds, manifest


def make_engine(cfg, manifest, run_dir):
    import torch
    from transformers import AutoConfig
    from vllm import LLM
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU runtime is required')
    native = AutoConfig.from_pretrained(MODEL, revision=MODEL_REVISION)
    if cfg['engine']['max_model_len'] > native.max_position_embeddings:
        raise ValueError('Configured context exceeds the pinned model context')
    runtime = {'gpu': torch.cuda.get_device_name(0), 'packages': {
        p: importlib.metadata.version(p) for p in
        ['torch', 'vllm', 'transformers', 'datasets', 'math-verify', 'sympy', 'latex2sympy2-extended']}}
    runtime_path = run_dir / 'runtime.json'
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise ValueError('GPU/software changed. Use the same runtime or a new RUN_NAME.')
    llm = LLM(model=MODEL, revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
              trust_remote_code=False, generation_config='vllm', seed=cfg['sampling']['seed'],
              tensor_parallel_size=1, **cfg['engine'])
    tokenizer = llm.get_tokenizer()
    stops = stop_ids(tokenizer)
    prompt = {'system': SYSTEM_PROMPT, 'chat_template': tokenizer.chat_template,
              'example': render_prompt(tokenizer, 'Compute 1+1.'),
              'stop_token_ids': stops, 'ignore_eos': False}
    prompt_path = run_dir / 'prompt.json'
    if prompt_path.exists() and json.loads(prompt_path.read_text()) != prompt:
        raise ValueError('Prompt/tokenizer changed')
    write_json(runtime_path, runtime)
    write_json(prompt_path, prompt)
    return SimpleNamespace(name='vllm', llm=llm, tokenizer=tokenizer,
                           config={**cfg['sampling'], **cfg['engine'], 'stop_token_ids': stops,
                                   'record_stop_diagnostics': True})


def completion_budget(prompt_tokens, cfg):
    remaining = cfg['engine']['max_model_len'] - prompt_tokens
    if remaining <= 0:
        raise ValueError('A prompt fills/exceeds the native context. No prompt was truncated.')
    return min(cfg['sampling']['max_new_tokens'], remaining)


def journal_index(run_dir, ds, selected):
    index = scan_journal(run_dir / 'responses_7b.jsonl', ds, {'id_column': 'id'})
    if not set(index) <= set(selected):
        raise ValueError('Checkpoint contains a row outside the requested 0/1/2 selection')
    return index


def generate(cfg, root, run_dir, token):
    ds, manifest = prepare(cfg, root, run_dir, token)
    generate_selected(ds, manifest, cfg, run_dir, manifest['selected_indices'])
    return export(ds, manifest, cfg, run_dir)


def generate_selected(ds, manifest, cfg, run_dir, selected):
    """One worker's journal; selection always contains original source indices."""
    from eval.batched_generation import generate_problems
    completed = journal_index(run_dir, ds, selected)
    pending = [i for i in selected if i not in completed]
    write_json(run_dir / 'progress.json', {'done': len(completed), 'total': len(selected)})
    print(f'7B: {len(completed)}/{len(selected)} selected rows complete', flush=True)
    if pending:
        engine = make_engine(cfg, manifest, run_dir)
        jobs = []
        for i in pending:
            row = ds[i]
            prompt = render_prompt(engine.tokenizer, row['problem'])
            n_tokens = len(engine.tokenizer.encode(prompt, add_special_tokens=False))
            try:
                budget = completion_budget(n_tokens, cfg)
            except ValueError as error:
                raise ValueError(f'Row {row["id"]}: {error}') from error
            jobs.append({'key': str(row['id']), 'source_index': i, 'problem': row['problem'],
                         'n': 8, 'seed': sample_seed(cfg['sampling']['seed'], str(row['id'])),
                         'max_new_tokens': budget, 'prompt_tokens': n_tokens})
        for job, _, responses in generate_problems(engine, jobs, cfg['execution'], prompt_renderer=render_prompt):
            row = ds[job['source_index']]
            audit = []
            result, errors = annotate(responses, row['gold_answer'],
                                      timeout=cfg['sampling']['verify_timeout_seconds'],
                                      problem=row['problem'], answer_type=row.get('answer_type'), audit_output=audit)
            append_jsonl(run_dir / 'responses_7b.jsonl', {
                'source_index': job['source_index'], 'id': str(row['id']), 'annotation': result,
                'grading_audit': audit, 'grading_errors': errors,
                'prompt_tokens': job['prompt_tokens'], 'max_new_tokens': job['max_new_tokens'],
                'generation_diagnostics': [{k: r.get(k) for k in ['stop_reason', 'last_token_id']} for r in responses]})
            completed[job['source_index']] = True
            write_json(run_dir / 'progress.json', {'done': len(completed), 'total': len(selected)})
            print(f'7B: {len(completed)}/{len(selected)}; id={row["id"]}; '
                  f'correct={result["num_correct"]}/8', flush=True)


def export(ds, manifest, cfg, run_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Features, List, Value
    selected = set(manifest['selected_indices'])
    index = journal_index(run_dir, ds, selected)
    if set(index) != selected:
        raise ValueError('Incomplete 7B generation; resume before exporting/uploading')
    out = run_dir / 'full'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'COMPLETE.json').unlink(missing_ok=True)
    # Variable lists support a null list on unselected rows; selected lists must have length eight.
    features = Features({**ds.features, **{
        '7B_' + k: List(Value('bool' if k == 'correct' else 'int64' if k == 'response_tokens' else 'string'))
        for k in COLUMNS[:-1]}, '7B_num_correct': Value('int64')})
    files, batch, histogram, finishes = [], [], Counter(), Counter()
    total_tokens = empty = errors = 0
    def flush():
        name = f'data/train-{len(files):05d}.parquet'
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(batch, schema=features.arrow_schema), path)
        files.append(name)
        batch.clear()
    # Creating an empty journal is valid when there are no eligible rows.
    (run_dir / 'responses_7b.jsonl').touch(exist_ok=True)
    with (run_dir / 'responses_7b.jsonl').open('rb') as journal, (out / 'grading_audit.jsonl').open('w') as audit:
        for i, row in enumerate(ds):
            if i in selected:
                journal.seek(index[i])
                record = json.loads(journal.readline())
                annotation = record['annotation']
                validate_annotation(annotation)
                additions = {'7B_' + k: annotation[k] for k in COLUMNS}
                histogram[annotation['num_correct']] += 1
                finishes.update(annotation['finish_reasons'])
                total_tokens += sum(annotation['response_tokens'])
                empty += sum(x == '' for x in annotation['extracted_answers'])
                errors += sum(x is not None for x in record['grading_errors'])
                audit.write(json.dumps({k: v for k, v in record.items() if k != 'annotation'}) + '\n')
            else:
                additions = dict.fromkeys(NEW_COLUMNS)
            batch.append({**row, **additions})
            if len(batch) >= cfg['export']['shard_rows']:
                flush()
        if batch:
            flush()
    summary = {'source_rows': len(ds), 'selected_rows': len(selected),
               'unselected_rows': len(ds) - len(selected), 'responses': len(selected)*8,
               '7B_num_correct_histogram': {str(i): histogram[i] for i in range(9)},
               'finish_reasons': dict(finishes), 'response_tokens': total_tokens,
               'empty_extractions': empty, 'grading_exceptions': errors,
               'selection': 'original num_correct in [0,1,2]', 'unselected_7B_values': None}
    write_json(out / 'summary.json', summary)
    runtime = {name: json.loads((run_dir / name).read_text()) if (run_dir / name).exists() else None
               for name in ['runtime.json', 'prompt.json']}
    write_json(out / 'provenance.json', {'manifest': manifest, **runtime,
                                        'response_budget': 'min(configured cap, 4096 - prompt tokens)'})
    files += ['summary.json', 'provenance.json', 'grading_audit.jsonl']
    write_json(out / 'COMPLETE.json', {'signature': manifest['signature'], 'rows': len(ds),
                                      'selected_rows': len(selected), 'features': features.to_dict(),
                                      'files': {name: hash_file(out / name) for name in files}})
    print(json.dumps(summary, indent=2), flush=True)
    return out


def validate_export(ds, manifest, out):
    import pyarrow.parquet as pq
    from datasets import Features
    complete = json.loads((out / 'COMPLETE.json').read_text())
    selected = set(manifest['selected_indices'])
    if (complete['signature'] != manifest['signature'] or complete['rows'] != len(ds)
            or complete['selected_rows'] != len(selected)):
        raise ValueError('Export does not match this run')
    for name, checksum in complete['files'].items():
        if hash_file(out / name) != checksum:
            raise ValueError(f'Changed artifact: {name}')
    schema = Features.from_dict(complete['features']).arrow_schema
    count = 0
    for name in sorted(n for n in complete['files'] if n.endswith('.parquet')):
        parquet = pq.ParquetFile(out / name)
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ValueError('Unexpected export schema')
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                if count >= len(ds) or {k: row[k] for k in ds.column_names} != ds[count]:
                    raise ValueError(f'Original row/3B result changed at row {count}')
                if count in selected:
                    validate_annotation({k: row['7B_' + k] for k in COLUMNS})
                elif any(row[k] is not None for k in NEW_COLUMNS):
                    raise ValueError('Unselected row has 7B results')
                count += 1
    if count != len(ds):
        raise ValueError('Incomplete exported rows')
    return complete


def upload(cfg, root, run_dir, token):
    from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download
    ds, manifest = prepare(cfg, root, run_dir, token)
    out = run_dir / 'full'
    complete = validate_export(ds, manifest, out)
    repo = cfg['data']['repo']
    remote = 'annotated-7b/' + manifest['signature'][:16]
    api = HfApi(token=token)
    source_card = hf_hub_download(repo, 'README.md', repo_type='dataset',
                                  revision=manifest['dataset_revision'], token=token)
    card = DatasetCard(Path(source_card).read_text())
    metadata = card.data.to_dict()
    metadata.pop('dataset_info', None)
    configurations = [c for c in metadata.get('configs', []) if c['config_name'] != 'default']
    metadata['configs'] = [{'config_name': 'default', 'default': True,
                           'data_files': [{'split': 'train', 'path': remote + '/data/*.parquet'}]}, *configurations]
    note = ('\n\n## Eight Qwen2.5-Math-7B-Instruct responses on the 3B 0/1/2 subset\n\n'
            'All original rows and 3B columns are retained unchanged. Only rows whose original '
            '`num_correct` is 0, 1, or 2 receive eight new samples. Added columns: '
            + ', '.join('`' + k + '`' for k in NEW_COLUMNS) + '. '
            'The six new columns are null on unselected rows; null means not evaluated, not incorrect. '
            'Math-Verify grades the extracted final answer using the previous grading policy, including '
            'the other-answer notation handling. Native instruct chat template and EOS are used. '
            'Native context is 4096 tokens; per-answer limits and truncation are recorded. '
            f'See `{remote}/provenance.json`, `summary.json`, and `grading_audit.jsonl`.\n')
    (out / 'README.md').write_text('---\n' + yaml.safe_dump(metadata, sort_keys=False) + '---\n' + card.text + note)
    mapping = {remote + '/' + name: name for name in [*complete['files'], 'COMPLETE.json']}
    mapping['README.md'] = 'README.md'
    current = api.dataset_info(repo).sha
    if current == manifest['dataset_revision']:
        commit = api.create_commit(repo, repo_type='dataset', parent_commit=current,
            commit_message='Add eight Math-7B-Instruct samples to 3B num_correct 0/1/2 rows',
            operations=[CommitOperationAdd(path_in_repo=remote_name, path_or_fileobj=str(out / local))
                        for remote_name, local in mapping.items()])
        revision = commit.oid
    else:
        # Recover even if the process stopped after the commit but before saving its receipt.
        try:
            marker = hf_hub_download(repo, remote + '/COMPLETE.json', repo_type='dataset', revision=current, token=token)
            if json.loads(Path(marker).read_text()) != complete:
                raise ValueError('Different run marker')
        except Exception as error:
            raise ValueError('HF changed since preparation; refusing to overwrite newer data') from error
        revision = current
    receipt = {'revision': revision, 'signature': manifest['signature'], 'verified': False,
               'url': f'https://huggingface.co/datasets/{repo}/tree/{revision}'}
    write_json(run_dir / 'upload_receipt.json', receipt)
    for remote_name, local in mapping.items():
        path = hf_hub_download(repo, remote_name, repo_type='dataset', revision=revision, token=token)
        if hash_file(path) != hash_file(out / local):
            raise ValueError(f'Upload verification failed: {remote_name}')
    receipt['verified'] = True
    write_json(run_dir / 'upload_receipt.json', receipt)
    print('Verified HF update:', receipt['url'], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'full', 'upload'], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(__file__).resolve().parents[1]
    {'prepare': prepare, 'full': generate, 'upload': upload}[args.mode](
        cfg, root, args.run_dir, os.environ.get('HF_TOKEN'))


if __name__ == '__main__':
    main()
