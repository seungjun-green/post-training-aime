"""Build an auditable Parquet dataset/card, then optionally publish it to the Hub."""

import argparse
import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import yaml

from common.io import digest, read_jsonl, write_json
from eval.run_transfer_eval import comparison, load_config, write_csv
from eval.run_transfer_runpod import exclusive, worker_lock
from eval.transfer_models import file_hash
from eval.transfer_scoring import summarize

FORMAT = 'qwen-transfer-results-v1'
BENCHMARKS = ['arc_challenge', 'mmlu_pro', 'bbh_nonmath', 'ifeval']


def markdown_table(rows, columns=None):
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))

    def cell(value):
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{value:.2f}'
        return str(value).replace('|', r'\|').replace('\n', '<br>')

    return '\n'.join(['| ' + ' | '.join(map(cell, columns)) + ' |',
                      '| ' + ' | '.join('---' for _ in columns) + ' |'] +
                     ['| ' + ' | '.join(cell(row.get(key)) for key in columns) + ' |' for row in rows])


def result_card(config, summary, run_id, repo_id, counts):
    prefix = f'runs/{run_id}'
    keys = [row['Key'] for row in summary['rows']]
    metadata = {'language': ['en'], 'tags': ['evaluation', 'qwen2.5', 'arc', 'mmlu-pro', 'bbh', 'ifeval'],
                'configs': [{'config_name': key, 'data_files': [
                    {'split': 'test', 'path': f'{prefix}/data/{key}/*.parquet'}]} for key in keys]}
    text = '---\n' + yaml.safe_dump(metadata, sort_keys=False) + '---\n\n'
    text += '# Qwen2.5-3B transfer evaluation results\n\n'
    text += (f'Completed **{len(keys)} models**, **{sum(counts.values()):,} inference records**. '
             f'Export `{run_id}`. This repository contains evaluation outputs, not training data.\n\n')
    selection = config.get('mmlu_selection')
    if selection is not None:
        text += (f'**MMLU-Pro: {selection["per_subject"]} questions per subject '
                 f'({selection["per_subject"] * len(config["mmlu_subjects"]):,} total), custom subset.** '
                 f'Seed **{selection["seed"]}**, method `{selection["method"]}`; identical question IDs '
                 'across all models, selected before inference without consulting answers or model results. '
                 'The data manifest lists selected IDs by subject. These are subset scores, not full '
                 'MMLU-Pro scores; small per-subject differences have greater sampling uncertainty.\n\n')
    text += '## Overall results\n\n' + markdown_table(summary['rows']) + '\n\n'
    for benchmark, title in [('arc_challenge', 'ARC-Challenge'), ('mmlu_pro', 'MMLU-Pro subjects and aggregates'),
                             ('bbh_nonmath', 'Non-math BBH tasks and aggregates'), ('ifeval', 'IFEval')]:
        rows = [row for row in summary['breakdowns'] if row['Benchmark'] == benchmark]
        text += '## ' + title + '\n\n' + markdown_table(rows, [
            'Model', 'Subject/task', 'Metric', 'Problems', 'Correct', 'Denominator',
            'Accuracy (%)', 'Mean tokens', 'Truncated (%)', 'Unparsed (%)']) + '\n\n'
    generation = config['generation']
    text += ('## Evaluation protocol\n\n'
             f'Temperature **{generation["temperature"]}**, top-p **{generation["top_p"]}**, '
             f'seed **{generation["seed"]}**, one response, **{generation["max_new_tokens"]:,}** new-token cap, '
             f'**{generation["max_model_len"]:,}** context. Native saved chat templates and EOS/PAD are preserved. '
             'ARC uses zero-shot generated-answer accuracy, not likelihood `acc_norm`. MMLU-Pro uses five '
             'official same-subject CoT validation examples; BBH uses three official CoT demonstrations. '
             'IFEval uses original prompts and the official strict/loose scorer. Extraction failures are incorrect; '
             'no random guessing. These chat-formatted experiments are an internal comparison, not exact leaderboard reproduction.\n\n'
             'MMLU-Pro includes math, with a separate non-math aggregate. Subject/task macro scores weight '
             'subjects/tasks equally; overall accuracy weights individual examples. IFEval prompt metrics use '
             'prompts as denominator, and instruction metrics use individual instructions. “—” means not applicable.\n\n'
             '**BBH included:** ' + ', '.join(config['bbh_tasks']) + '.\n\n'
             '**BBH excluded:** ' + ', '.join(config['bbh_excluded']) + '.\n\n')
    text += '## Models and checkpoint selection\n\n' + markdown_table([
        {'Key': key, 'Model': config['models'][key]['label'],
         'Checkpoint / repository': config['models'][key].get('path', config['models'][key].get('repo')),
         'HF revision': config['models'][key].get('revision'),
         'EOS token ID': config['models'][key]['eos_token_id'], 'Inference records': counts[key]} for key in keys]) + '\n\n'
    text += ('## Files and loading\n\n'
             f'- [Overall CSV]({prefix}/tables/comparison.csv)\n'
             f'- [Full subject/task metrics CSV]({prefix}/tables/subject_task_breakdowns.csv)\n'
             f'- [Metrics JSON]({prefix}/tables/comparison.json)\n'
             f'- [Evaluation configuration]({prefix}/provenance/config.json)\n'
             f'- [Data sources and hashes]({prefix}/provenance/data_manifest.json)\n'
             f'- [Model manifests and results]({prefix}/provenance/models)\n'
             f'- [Export file checksums]({prefix}/export_manifest.json)\n\n'
             'Parquet preserves long multiline prompts/responses with compression. One configuration per model, '
             'with all four benchmarks in its `test` split; one file per model/benchmark. '
             'Columns include model key/label, benchmark, subject/task, example ID, prompt, formatted prompt, '
             'response, gold answer, prediction, correctness, token count, finish reason, and input digest. '
             '`score_json` preserves complete grading details (including IFEval per-instruction results); '
             '`document_json` preserves all original evaluation fields and IFEval instruction arguments. '
             'Non-IFEval correctness is null for IFEval; use its strict/loose score fields.\n\n'
             '```python\nfrom datasets import load_dataset\n'
             f'ds = load_dataset("{repo_id}", "{keys[0]}", split="test", token=True)\n'
             'print(ds[0]["response"])\n```\n\n'
             'Re-exports retain older snapshots under `runs/`; this README and dataset configurations point to '
             'the current snapshot. No model weights, OAuth credentials, environment files or console logs are uploaded.\n\n'
             '## Source attribution\n\n'
             'Benchmark materials retain their upstream terms. Exact revisions and source-file hashes are in the data manifest.\n\n')
    for repo in sorted({source['repo'] for source in config['sources'].values()}):
        text += f'- [{repo}](https://huggingface.co/datasets/{repo})\n'
    text += (f'\n[Scorer/prompt attribution]({prefix}/provenance/TRANSFER_NOTICE.md) and '
             f'[vendored-source revisions]({prefix}/provenance/transfer_sources.json).\n')
    return text


def parquet_rows(records, key, label):
    for row in records:
        document = {k: v for k, v in row.items() if k not in
                    {'score', 'text', 'formatted_prompt', 'token_count', 'finish_reason', 'input_digest'}}
        score = row['score']
        yield {'model_key': key, 'model_label': label, 'benchmark': row['benchmark'],
               'subject': row['subject'], 'example_id': row['id'], 'prompt': row['prompt'],
               'formatted_prompt': row['formatted_prompt'], 'response': row['text'],
               'gold_answer': row.get('answer'), 'prediction': score.get('prediction'),
               'correct': score.get('correct'), 'token_count': row['token_count'],
               'finish_reason': row['finish_reason'], 'input_digest': row['input_digest'],
               'score_json': json.dumps(score, ensure_ascii=False, sort_keys=True),
               'document_json': json.dumps(document, ensure_ascii=False, sort_keys=True)}


def build_export(config, output_root, repo_id, selected=None):
    """Validate completed full runs before building an isolated, allowlisted snapshot."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from eval.run_transfer_eval import ROOT

    root = Path(output_root)
    keys = list(config['models']) if selected is None else selected
    if not keys or len(set(keys)) != len(keys) or set(keys) - set(config['models']):
        raise ValueError('Select unique known models')
    with ExitStack() as stack:
        stack.enter_context(exclusive(root / 'runpod_coordinator.lock'))
        for key in keys:
            stack.enter_context(exclusive(worker_lock(root, key)))
        missing = [key for key in keys if not (root / 'models' / key / 'full/results.json').is_file()]
        if missing:
            raise ValueError('Full evaluations must finish before export: ' + ', '.join(missing))
        comparison(config, root, keys)
        summary = json.loads((root / 'comparison.json').read_text())
        data = json.loads((root / 'data/manifest.json').read_text())
        documents = read_jsonl(root / 'data/documents.jsonl')
        if digest(documents) != data['documents_digest']:
            raise ValueError('Prepared benchmark data changed')
        lookup = {doc['id']: doc for doc in documents}
        expected_ids = set(lookup)
        schemas = pa.schema([(name, pa.bool_() if name == 'correct' else
                              pa.int64() if name == 'token_count' else pa.string()) for name in [
            'model_key', 'model_label', 'benchmark', 'subject', 'example_id', 'prompt', 'formatted_prompt',
            'response', 'gold_answer', 'prediction', 'correct', 'token_count', 'finish_reason',
            'input_digest', 'score_json', 'document_json']])
        exports = root / 'hf_exports'
        exports.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='building-', dir=exports) as directory:
            staging = Path(directory)
            payload = staging / 'payload'
            counts = {}
            for key in keys:
                full = root / 'models' / key / 'full'
                result = json.loads((full / 'results.json').read_text())
                identity = result['identity']
                if identity['data_digest'] != data['documents_digest'] or set(identity['selected_ids']) != expected_ids:
                    raise ValueError(f'Incomplete or changed benchmark selection: {key}')
                if json.loads((full / 'manifest.json').read_text()) != identity:
                    raise ValueError(f'Result/manifest mismatch: {key}')
                if json.loads((full / 'tokenizer_identity.json').read_text()) != result['tokenizer']:
                    raise ValueError(f'Result/tokenizer mismatch: {key}')
                raw_rows = read_jsonl(full / 'responses.jsonl')
                raw = {row['id']: row for row in raw_rows}
                if len(raw) != len(raw_rows) or set(raw) != expected_ids:
                    raise ValueError(f'Incomplete or duplicate raw responses: {key}')
                seen, metrics = set(), []
                for benchmark in BENCHMARKS:
                    records = read_jsonl(full / benchmark / 'generations.jsonl')
                    for row in records:
                        doc = lookup.get(row['id'])
                        if (doc is None or row['id'] in seen or row['benchmark'] != benchmark
                                or row['input_digest'] != digest(doc)
                                or any(row.get(k) != v for k, v in doc.items())
                                or any(row.get(k) != raw[row['id']].get(k) for k in
                                       ['text', 'formatted_prompt', 'token_count', 'finish_reason', 'input_digest'])
                                or row['finish_reason'] not in ['stop', 'length']
                                or not isinstance(row['text'], str)):
                            raise ValueError(f'Invalid or duplicate inference record: {key}/{benchmark}')
                        seen.add(row['id'])
                    metrics.extend(summarize(records))
                    destination = payload / 'data' / key / (benchmark + '.parquet')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(pa.Table.from_pylist(list(parquet_rows(records, key, config['models'][key]['label'])),
                                                       schema=schemas), destination, compression='zstd')
                if seen != expected_ids or len(seen) != result['completed'] or metrics != result['metrics']:
                    raise ValueError(f'Incomplete results or score table mismatch: {key}')
                counts[key] = len(seen)
                model_dir = payload / 'provenance/models' / key
                for name in ['manifest.json', 'tokenizer_identity.json', 'results.json']:
                    write_json(model_dir / name, json.loads((full / name).read_text()))
            write_csv(payload / 'tables/comparison.csv', summary['rows'])
            write_csv(payload / 'tables/subject_task_breakdowns.csv', summary['breakdowns'])
            write_json(payload / 'tables/comparison.json', summary)
            write_json(payload / 'provenance/config.json', config)
            write_json(payload / 'provenance/data_manifest.json', data)
            for name in ['TRANSFER_NOTICE.md', 'transfer_sources.json']:
                shutil.copyfile(ROOT / 'third_party' / name, payload / 'provenance' / name)
            for name in ['runpod_protocol.json', 'hf_checkpoint_catalog.json']:
                if (root / name).is_file():
                    shutil.copyfile(root / name, payload / 'provenance' / name)
            hashes = {p.relative_to(payload).as_posix(): file_hash(p) for p in sorted(payload.rglob('*')) if p.is_file()}
            record = {'format': FORMAT, 'models': keys, 'counts': counts, 'files': hashes}
            run_id = digest(record)[:20]
            write_json(payload / 'export_manifest.json', {**record, 'run_id': run_id})
            destination = exports / run_id
            snapshot = destination / 'runs' / run_id
            if snapshot.exists():
                existing = json.loads((snapshot / 'export_manifest.json').read_text())
                if existing != {**record, 'run_id': run_id} or any(file_hash(snapshot / p) != h for p, h in hashes.items()):
                    raise ValueError('Existing export snapshot changed')
            else:
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                payload.rename(snapshot)
            (destination / 'README.md').write_text(result_card(config, summary, run_id, repo_id, counts))
            write_json(destination / 'transfer_export.json', {'format': FORMAT, 'run_id': run_id})
            return destination


def publish_export(folder, repo_id, *, private=True, token=None, api=None):
    """Commit only the validated export allowlist; never upload the workspace."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(repo_id)
    if '/' not in repo_id:
        raise ValueError('Use an explicit owner/dataset repository ID')
    if not token:
        raise ValueError('Set HF_TOKEN to a token with write access before uploading')
    folder = Path(folder)
    marker = json.loads((folder / 'transfer_export.json').read_text())
    if marker['format'] != FORMAT:
        raise ValueError('Unknown export format')
    prefix = 'runs/' + marker['run_id']
    manifest_path = prefix + '/export_manifest.json'
    manifest = json.loads((folder / manifest_path).read_text())
    if manifest['run_id'] != marker['run_id'] or manifest['format'] != FORMAT:
        raise ValueError('Export identity mismatch')
    files = ['README.md', 'transfer_export.json', manifest_path]
    for name, expected in manifest['files'].items():
        relative = Path(prefix) / name
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(folder / relative) != expected:
            raise ValueError('Export file changed: ' + name)
        files.append(relative.as_posix())
    api = api or HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type='dataset', private=private, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type='dataset')
    if info.private != private:
        raise ValueError('Existing repository visibility differs from HF_PRIVATE; choose a matching setting or a new repository')
    existing = set(api.list_repo_files(repo_id=repo_id, repo_type='dataset', revision=info.sha))
    if 'transfer_export.json' in existing:
        previous = json.loads(Path(api.hf_hub_download(repo_id=repo_id, repo_type='dataset',
                              filename='transfer_export.json', revision=info.sha)).read_text())
        if previous.get('format') != FORMAT:
            raise ValueError('Repository contains a different export format; choose a new repository')
    elif existing - {'.gitattributes'}:
        raise ValueError('Destination is not an empty or existing transfer-results repository; choose a new repository')
    result = api.create_commit(repo_id=repo_id, repo_type='dataset', parent_commit=info.sha,
                               operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=folder / name) for name in files],
                               commit_message=f'Add transfer evaluation results {marker["run_id"]}')
    print('Dataset:', f'https://huggingface.co/datasets/{repo_id}', flush=True)
    print('Commit:', result.commit_url, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--models')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--public', action='store_true')
    args = parser.parse_args()
    folder = build_export(load_config(args.config), args.output_root, args.repo_id,
                          args.models.split(',') if args.models else None)
    print('HF export folder:', folder, flush=True)
    if args.upload:
        publish_export(folder, args.repo_id, private=not args.public, token=os.getenv('HF_TOKEN'))


if __name__ == '__main__':
    main()
