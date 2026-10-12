"""CPU-only Drive → HF model publication and pinned HF → RunPod checkpoint transfer."""

import json
import re
import tempfile
from pathlib import Path

import yaml

from common.io import digest, write_json
from eval.runpod_transfer_storage import check_space
from eval.transfer_models import file_hash, validate_checkpoint, weight_files

FORMAT = 'qwen-transfer-checkpoint-v1'
CATALOG_FORMAT = 'qwen-transfer-catalog-v1'
DESCRIPTOR = 'checkpoint_upload.json'
TRAIN_MANIFEST = 'training/run_manifest.json'
INFERENCE_NAMES = {
    'config.json', 'generation_config.json', 'model.safetensors.index.json',
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
    'vocab.json', 'merges.txt', 'chat_template.jinja', 'tokenizer.model',
    'stage1_checkpoint.json', 'dapo_checkpoint.json',
}


def inference_name(name):
    path = Path(name)
    return (bool(path.parts) and not path.is_absolute() and '..' not in path.parts and (
        name in INFERENCE_NAMES or re.fullmatch(r'model(?:-\d+-of-\d+)?\.safetensors', name) is not None
        or (path.parts[0] == 'chat_templates' and path.suffix == '.jinja')))


def checkpoint_path(config, key, experiment_root, instruct_rl_root):
    entry = config['models'][key]
    return (Path(instruct_rl_root) / 'checkpoints' / Path(entry['path']).parent.name / 'checkpoint-300'
            if key == 'instruct_rl' else Path(experiment_root) / entry['path'])


def training_manifest_path(path, entry):
    is_sft = entry['kind'] == 'sft'
    return path.parents[3 if is_sft else 2] / ('logs/stage1' if is_sft else 'logs') / path.parent.name / 'run_manifest.json'


def upload_plan(config, hub, experiment_root, instruct_rl_root):
    from huggingface_hub.utils import validate_repo_id

    keys = [key for key, entry in config['models'].items() if 'path' in entry]
    if set(hub['repositories']) != set(keys) or len(set(hub['repositories'].values())) != len(keys):
        raise ValueError('Map all seven trained variants to seven distinct model repositories')
    if hub['catalog_key'] not in keys or hub['catalog_filename'] != 'transfer_catalog.json':
        raise ValueError('Invalid checkpoint catalog settings')
    plan = []
    for key in keys:
        repo = hub['repositories'][key]
        validate_repo_id(repo)
        if '/' not in repo:
            raise ValueError('Use owner/model repository IDs')
        path = checkpoint_path(config, key, experiment_root, instruct_rl_root)
        validate_checkpoint(path, config['models'][key], key, config)
        files = {p.relative_to(path).as_posix(): p for p in path.rglob('*') if p.is_file()
                 and inference_name(p.relative_to(path).as_posix())}
        # The selected index determines which shards belong to this checkpoint.
        expected_weights = {p.name for p in weight_files(path)}
        if {name for name in files if name.endswith('.safetensors')} != expected_weights:
            raise ValueError(f'Unexpected extra weight shards: {path}')
        files[TRAIN_MANIFEST] = training_manifest_path(path, config['models'][key])
        plan.append({'key': key, 'repo_id': repo, 'path': str(path), 'files': files,
                     'bytes': sum(p.stat().st_size for p in files.values())})
    return plan


def read_remote(api, repo, revision, filename):
    return json.loads(Path(api.hf_hub_download(repo_id=repo, repo_type='model', revision=revision,
                                              filename=filename)).read_text())


def verify_remote(api, repo, revision, descriptor):
    info = api.model_info(repo, revision=revision, files_metadata=True)
    siblings = {item.rfilename: item for item in info.siblings}
    for name, expected in descriptor['files'].items():
        remote = siblings.get(name)
        if remote is None or remote.size != expected['size']:
            raise ValueError(f'Uploaded file missing or wrong size: {repo}/{name}')
        if remote.lfs:
            actual = remote.lfs.sha256
        else:
            if expected['size'] > 50 * 1024**2:
                raise ValueError(f'Missing large-file checksum: {repo}/{name}')
            actual = file_hash(api.hf_hub_download(repo_id=repo, repo_type='model', revision=revision, filename=name))
        if actual != expected['sha256']:
            raise ValueError(f'Uploaded checksum mismatch: {repo}/{name}')


def upload_checkpoint(config, item, *, private, api, catalog_filename):
    from huggingface_hub import CommitOperationAdd

    key, repo, path = item['key'], item['repo_id'], Path(item['path'])
    marker = validate_checkpoint(path, config['models'][key], key, config)
    descriptor = {'format': FORMAT, 'key': key, 'spec': config['models'][key],
                  'run_identity': marker['run_identity'], 'files': {}}
    for name, file in item['files'].items():
        print('Hashing:', key, name, flush=True)
        descriptor['files'][name] = {'size': file.stat().st_size, 'sha256': file_hash(file)}
    api.create_repo(repo_id=repo, repo_type='model', private=private, exist_ok=True)
    info = api.model_info(repo)
    if info.private != private:
        raise ValueError(f'{repo} visibility differs from HF_PRIVATE; choose a matching setting or new repository')
    names = {s.rfilename for s in info.siblings}
    if DESCRIPTOR in names:
        if read_remote(api, repo, info.sha, DESCRIPTOR) != descriptor:
            raise ValueError(f'{repo} already contains different weights/metadata; use a new repository')
        revision = info.sha
        # The catalog is stored in one of the seven repos. Retain its weight commit,
        # rather than publishing a new self-referential catalog on every rerun.
        if catalog_filename in names:
            old = read_remote(api, repo, info.sha, catalog_filename).get('models', {}).get(key, {})
            if old.get('repo_id') == repo and old.get('descriptor_digest') == digest(descriptor):
                revision = old['revision']
        if read_remote(api, repo, revision, DESCRIPTOR) != descriptor:
            raise ValueError('Saved model revision no longer matches its upload descriptor')
        verify_remote(api, repo, revision, descriptor)
        print('Already uploaded and verified:', repo, flush=True)
    else:
        if names - {'.gitattributes'}:
            raise ValueError(f'{repo} is not empty or owned by this checkpoint uploader; choose a new repository')
        base = config['models']['instruct' if key == 'instruct_rl' else 'base']['repo']
        metadata = {'library_name': 'transformers', 'pipeline_tag': 'text-generation',
                    'base_model': base, 'tags': ['qwen2.5', 'math', 'evaluation-checkpoint']}
        readme = ('---\n' + yaml.safe_dump(metadata, sort_keys=False) + '---\n\n# ' + config['models'][key]['label'] + '\n\n'
                  f'Full Qwen2.5-3B checkpoint for variant `{key}`. Source: `{config["models"][key]["path"]}`.\n\n'
                  'Weights, tokenizer, chat template and EOS/PAD configuration are copied verbatim. '
                  'No merging, conversion, quantization or token changes. Optimizer states are omitted.\n\n'
                  f'Step: **{marker["global_step"]}**. Epoch: **{marker.get("epoch", "not applicable")}**. '
                  f'Run identity: `{marker["run_identity"]}`.\n\n'
                  f'Base model: [{base}](https://huggingface.co/{base}); its original terms apply.\n\n'
                  f'File checksums: `{DESCRIPTOR}`. Training provenance: `{TRAIN_MANIFEST}`.\n')
        with tempfile.TemporaryDirectory(prefix='qwen-checkpoint-upload-') as directory:
            directory = Path(directory)
            write_json(directory / DESCRIPTOR, descriptor)
            (directory / 'README.md').write_text(readme)
            files = {**item['files'], DESCRIPTOR: directory / DESCRIPTOR, 'README.md': directory / 'README.md'}
            commit = api.create_commit(repo_id=repo, repo_type='model', parent_commit=info.sha,
                operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=file) for name, file in files.items()],
                commit_message=f'Publish {key} selected checkpoint')
            revision = commit.oid
        verify_remote(api, repo, revision, descriptor)
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Expected an immutable 40-character model commit revision')
    return {'repo_id': repo, 'revision': revision, 'descriptor_digest': digest(descriptor)}


def upload_all(config, hub, experiment_root, instruct_rl_root, receipt_dir, *, token, api=None):
    from huggingface_hub import CommitOperationAdd, HfApi

    if not token:
        raise ValueError('Provide a write-enabled HF_TOKEN')
    plan = upload_plan(config, hub, experiment_root, instruct_rl_root)  # Validate all seven before publishing.
    api = api or HfApi(token=token)
    receipt_dir = Path(receipt_dir)
    records = {}
    for index, item in enumerate(plan, 1):
        print(f'Checkpoint {index}/7: {item["key"]} → {item["repo_id"]}', flush=True)
        records[item['key']] = upload_checkpoint(config, item, private=hub['private'], api=api,
                                                catalog_filename=hub['catalog_filename'])
        write_json(receipt_dir / 'uploaded_models.json', records)
    catalog = {'format': CATALOG_FORMAT, 'model_specs_digest': digest(config['models']), 'models': records}
    local = receipt_dir / hub['catalog_filename']
    write_json(local, catalog)
    repo = hub['repositories'][hub['catalog_key']]
    info = api.model_info(repo)
    if (hub['catalog_filename'] in {s.rfilename for s in info.siblings}
            and read_remote(api, repo, info.sha, hub['catalog_filename']) == catalog):
        revision = info.sha
    else:
        commit = api.create_commit(repo_id=repo, repo_type='model', parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo=hub['catalog_filename'], path_or_fileobj=local)],
            commit_message='Publish verified seven-checkpoint catalog')
        revision = commit.oid
    if read_remote(api, repo, revision, hub['catalog_filename']) != catalog:
        raise ValueError('Published catalog verification failed')
    receipt = {'repo_id': repo, 'revision': revision, 'filename': hub['catalog_filename'], 'catalog': catalog}
    write_json(receipt_dir / 'upload_complete.json', receipt)
    print('All seven checkpoints verified. Catalog:', repo, '@', revision, flush=True)
    return receipt


def resolve_catalog(config, output_root, repo, revision, api):
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    path = Path(output_root) / 'hf_checkpoint_catalog.json'
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt['repo_id'] != repo or (revision != 'main' and receipt['revision'] != revision):
            raise ValueError('Checkpoint catalog changed; use a new OUTPUT_ROOT')
    else:
        try:
            sha = api.model_info(repo, revision=revision).sha
            catalog = read_remote(api, repo, sha, 'transfer_catalog.json')
        except (EntryNotFoundError, RepositoryNotFoundError) as error:
            raise FileNotFoundError(
                f'HF checkpoint catalog unavailable at {repo}. Complete the Colab seven-checkpoint '
                'upload notebook first, and use HF_TOKEN with access to those repositories.') from error
        if not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise ValueError('Expected a pinned catalog commit')
        receipt = {'repo_id': repo, 'revision': sha, 'catalog': catalog}
    catalog = receipt['catalog']
    trained = {k for k, v in config['models'].items() if 'path' in v}
    if (catalog.get('format') != CATALOG_FORMAT or catalog.get('model_specs_digest') != digest(config['models'])
            or set(catalog.get('models', {})) != trained):
        raise ValueError('Catalog does not match the seven selected training checkpoints')
    for item in catalog['models'].values():
        if not re.fullmatch(r'[0-9a-f]{40}', item['revision']):
            raise ValueError('Every trained checkpoint must use an immutable revision')
    write_json(path, receipt)
    return receipt


def download_checkpoints(config, storage_root, experiment_root, instruct_rl_root, output_root,
                         catalog_repo, *, catalog_revision='main', selected=None, token=None, api=None):
    from huggingface_hub import HfApi

    keys = list(config['models']) if selected is None else selected
    if not keys or len(set(keys)) != len(keys) or set(keys) - set(config['models']):
        raise ValueError('Select unique known model keys')
    trained = [key for key in keys if 'path' in config['models'][key]]
    if not trained:
        print('Only original base/instruct models selected; no trained checkpoint download needed.')
        return []
    api = api or HfApi(token=token)
    receipt = resolve_catalog(config, output_root, catalog_repo, catalog_revision, api)
    plan, missing = [], 0
    for key in trained:
        record = receipt['catalog']['models'][key]
        descriptor = read_remote(api, record['repo_id'], record['revision'], DESCRIPTOR)
        if (digest(descriptor) != record['descriptor_digest'] or descriptor.get('format') != FORMAT
                or descriptor.get('key') != key or descriptor.get('spec') != config['models'][key]
                or TRAIN_MANIFEST not in descriptor['files']):
            raise ValueError(f'Wrong checkpoint descriptor: {key}')
        path = checkpoint_path(config, key, experiment_root, instruct_rl_root)
        files = []
        for name, expected in descriptor['files'].items():
            if name != TRAIN_MANIFEST and not inference_name(name):
                raise ValueError(f'Unexpected checkpoint file: {name}')
            destination = training_manifest_path(path, config['models'][key]) if name == TRAIN_MANIFEST else path / name
            valid = (destination.is_file() and destination.stat().st_size == expected['size']
                     and file_hash(destination) == expected['sha256'])
            if not valid:
                missing += expected['size']
            files.append((name, expected, destination, valid))
        plan.append((key, record, path, files))
    print('Download space:', check_space(storage_root, additional_bytes=missing), flush=True)
    for index, (key, record, path, files) in enumerate(plan, 1):
        print(f'Download {index}/{len(plan)}: {key} @ {record["revision"]}', flush=True)
        for name, expected, destination, valid in files:
            if valid:
                continue
            directory = (Path(storage_root) / 'cache/checkpoint-metadata' / key
                         if name == TRAIN_MANIFEST else path)
            downloaded = Path(api.hf_hub_download(repo_id=record['repo_id'], repo_type='model',
                revision=record['revision'], filename=name, local_dir=str(directory), force_download=destination.exists()))
            if downloaded.stat().st_size != expected['size'] or file_hash(downloaded) != expected['sha256']:
                raise ValueError(f'Download checksum mismatch: {key}/{name}')
            if name == TRAIN_MANIFEST:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.json.tmp')
                temporary.write_bytes(downloaded.read_bytes())
                temporary.replace(destination)
        validate_checkpoint(path, config['models'][key], key, config)
    return [str(path) for _, _, path, _ in plan]
