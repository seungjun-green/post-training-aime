"""Pinned Hub inputs and atomic publication for the Stage 1 notebook."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import yaml

from common.io import digest, write_json
from pipeline.datasets import core_dapo_problem, nested
from pipeline.sft_pool import ReferenceIndex, adapt, build_pool


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_references(cfg, root, token, manifest):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    suite = json.loads((Path(root) / cfg['eval_suite']).read_text())
    if set(suite['datasets']) != {'aime_2024', 'aime_2025', 'aime_2026', 'amc23', 'math_500'}:
        raise ValueError('Expected all five evaluation references')
    refs = []
    for name, spec in suite['datasets'].items():
        if spec['role'] != 'eval' or spec['removed_rows'] != 0:
            raise ValueError('Evaluation reference must be unchanged')
        info = api.dataset_info(spec['repo'], revision=spec['revision'])
        if info.sha != spec['revision']:
            raise ValueError('Evaluation commit is not pinned')
        rows = list(load_dataset(spec['repo'], name=spec['config'], split=spec['split'],
                                 revision=info.sha, token=token))
        ids = [nested(row, spec['id_column']) for row in rows]
        if len(rows) != spec['rows'] or digest(rows) != spec['content_digest'] or ids != spec['ids']:
            raise ValueError(f'Evaluation content/IDs changed: {name}')
        refs.extend({'id': identifier, 'set': name, 'problem': row[spec['problem_column']]}
                    for identifier, row in zip(ids, rows, strict=True))
        manifest[name] = spec
    spec = dict(yaml.safe_load((Path(root) / cfg['rl_config']).read_text())['data'])
    spec['revision'] = api.dataset_info(spec['repo'], revision=spec['revision']).sha
    rows = list(load_dataset(spec['repo'], name=spec['config'], split=spec['split'],
                             revision=spec['revision'], token=token))
    provenance = json.loads(Path(hf_hub_download(spec['repo'], 'provenance.json', repo_type='dataset',
                                                revision=spec['revision'], token=token)).read_text())
    ids = provenance['retained_preparation_ids']
    # Same input invariants as train.dapo_data.load_data, without trainer imports.
    if (len(rows) != spec['expected_rows'] or digest(rows) != spec['content_digest'] or
            digest(ids) != spec['ids_digest'] or len(set(map(str, ids))) != len(rows) or
            provenance['retained_originals_digest'] != spec['content_digest']):
        raise ValueError('RL reference differs from configured DAPO training inputs')
    manifest['dapo'] = spec
    rl = [{'id': identifier, 'set': 'dapo', 'problem': core_dapo_problem(row['prompt'])}
          for identifier, row in zip(ids, rows, strict=True)]
    return refs, rl


def source_rows(cfg, token, manifest):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for name in cfg['source_priority']:
        spec = cfg['sources'][name]
        info = api.dataset_info(spec['repo'], revision=spec['revision'])
        if info.sha != spec['revision']:
            raise ValueError(f'Pin a full source commit for {name}')
        manifest[name] = dict(spec, license=(info.card_data.to_dict() if info.card_data else {}).get('license'))
        total = 0
        for config in spec['configs']:
            ds = load_dataset(spec['repo'], name=config, split=spec['split'], revision=info.sha, token=token)
            needed = {'question', 'solution', 'cot_type', 'source_type', 'metadata'} if name == 's1' else {'problem', 'solution'}
            if name == 'numina':
                needed |= {'answer', 'source', 'question_type', 'problem_type', 'problem_is_valid', 'solution_is_valid'}
            if not needed <= set(ds.column_names):
                raise ValueError(f'Schema mismatch: {name}: {needed-set(ds.column_names)}')
            # Arrow-backed rows; never materialize the entire Numina corpus in memory.
            total += len(ds)
            manifest[name].setdefault('features', {})[config] = ds.features.to_dict()
            print(f'Loading {name}/{config}: {len(ds):,} rows', flush=True)
            for i, row in enumerate(ds):
                yield adapt(name, row, i, spec, config)
        if total != spec['expected_rows']:
            raise ValueError(f'Unexpected row count for {name}: {total}')
        manifest[name]['loaded_rows'] = total


def run(cfg, root, output, token):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'COMPLETE.json'
    if marker.exists():
        marker.unlink()  # A failed rerun must never look publishable.
    provenance = {'config': cfg, 'datasets': {},
                  'code': json.loads((root / 'sft_pool_bundle_manifest.json').read_text()),
                  'python': platform.python_version(),
                  'packages': {name: importlib.metadata.version(name) for name in
                               ['datasets', 'huggingface-hub', 'math-verify', 'sympy', 'PyYAML',
                                'latex2sympy2-extended', 'antlr4-python3-runtime', 'mpmath']},
                  'normalization_version': 'sft-pool-lexical-v1',
                  'dedup_algorithm': 'exact Jaccard prefix join over distinct word n-grams',
                  'conflict_policy': 'quarantine all cluster members',
                  'reference_policy': 'exact repository eval suite and content-checked RL config',
                  'number_variant_policy': 'additional exact numeric-template review candidates, including coverage below lower bound'}
    for filename, expected in provenance['code']['files_sha256'].items():
        if sha256_file(root / filename) != expected:
            raise ValueError(f'Bundled code changed: {filename}; regenerate the notebook')
    eval_rows, rl_rows = load_references(cfg, root, token, provenance['datasets'])
    # Real acceptance probe against the actual pinned benchmark, before expensive work.
    probe = next(row for row in eval_rows if row['set'] == 'math_500')
    index = ReferenceIndex([probe], cfg['reference_ngram'])
    for text in [probe['problem'], 'Extra context. ' + probe['problem'] + ' End.']:
        if max(m['coverage'] for m in index.matches(text)) != 1.0:
            raise AssertionError('Pinned MATH-500 injection was not detected')
    summary = build_pool(source_rows(cfg, token, provenance['datasets']), eval_rows, rl_rows, cfg, output)
    write_json(output / 'provenance.json', provenance)
    (output / 'README.md').write_text('''---
pretty_name: Clean Math SFT Problem Pool
configs:
- config_name: default
  data_files:
  - split: train
    path: pool.jsonl
---
# Clean math SFT problem pool

Inference-free preparation of MATH training, NuminaMath-1.5, and original s1K.
Only questions, extracted gold answers, and provenance are retained; no generated traces.
The `metadata` column contains lossless JSON text for heterogeneous source metadata.
Source licenses remain applicable; see provenance.json for upstream metadata and revisions.
The repository's unchanged AIME 2024/2025/2026, AMC 2023, MATH-500 evaluation
references and its content-checked DAPO RL set are excluded using distinct n-gram coverage.

`summary.json` contains stage counts; `removed.jsonl` records every removal;
`conflicts.jsonl` quarantines conflicting duplicate clusters; `merged_clusters.jsonl`
records merges. The two `*_near_misses.jsonl` files list manual-review candidates
in descending coverage order. Exact number-swapped templates can appear below the
usual near-miss band and are explicitly marked. Eval near misses are recorded at
the eval stage, so a candidate can subsequently be removed for RL overlap.

Filtering uses conservative rules, not a semantic classifier; inspect the audit logs.
Symbolic parsing does not establish that an upstream gold answer is correct.
Lexical normalization removes punctuation, and is not symbolic canonicalization.
Conflicting near-duplicate clusters are dropped in full as a conservative policy.

See `provenance.json` for the complete config, pinned inputs, code commit and
bundled-file hashes. This artifact performs no model inference or training.
''', encoding='utf-8')
    files = sorted(p for p in output.iterdir() if p.suffix in {'.json', '.jsonl', '.md'} and p != marker)
    write_json(marker, {'files': {p.name: sha256_file(p) for p in files},
                        'rows': sum(c['rl_disjoint'] for c in summary['counts'].values())})
    return summary


def upload(output, repo, token, private=True):
    """Publish only a completed, checksum-verified run; retain upstream inputs locally."""
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    output = Path(output)
    complete = json.loads((output / 'COMPLETE.json').read_text())
    if complete['rows'] <= 0:
        raise ValueError('Refusing to publish an empty pool')
    for filename, checksum in complete['files'].items():
        if sha256_file(output / filename) != checksum:
            raise ValueError(f'Artifact changed since completion: {filename}')
    api = HfApi(token=token)
    api.create_repo(repo, repo_type='dataset', private=private, exist_ok=True)
    info = api.dataset_info(repo)
    if info.private != private:
        raise ValueError('Existing repository visibility differs from hf_private; update config explicitly')
    paths = [output / name for name in sorted(complete['files'])] + [output / 'COMPLETE.json']
    commit = api.create_commit(repo, repo_type='dataset', parent_commit=info.sha,
                               commit_message='Publish reproducible Stage 1 math SFT pool and audit reports',
                               operations=[CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in paths])
    for p in paths:
        remote = hf_hub_download(repo, p.name, repo_type='dataset', revision=commit.oid, token=token)
        if sha256_file(remote) != sha256_file(p):
            raise ValueError(f'Uploaded checksum mismatch: {p.name}')
    return {'repo': repo, 'revision': commit.oid, 'verified': True,
            'url': f'https://huggingface.co/datasets/{repo}/tree/{commit.oid}'}
