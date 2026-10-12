import ast
import base64
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import yaml
from test_transfer_eval import fixture_checkpoint

from common.io import digest, write_json
from eval import transfer_hub as hub
from eval.run_transfer_eval import load_config
from eval.transfer_models import file_hash, prepare_models

ROOT = Path(__file__).resolve().parents[1]


class FakeHub:
    def __init__(self, root):
        self.root, self.repos = root, {}
        self.commits, self.downloads = [], []
        self.corrupt_download = None

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        assert repo_type == 'model' and exist_ok
        if repo_id not in self.repos:
            self.repos[repo_id] = {'private': private, 'sha': '0' * 40}
            self.directory(repo_id, '0' * 40).mkdir(parents=True)

    def directory(self, repo, revision):
        sha = self.repos[repo]['sha'] if revision in [None, 'main'] else revision
        return self.root / repo.replace('/', '__') / sha

    def model_info(self, repo_id, revision=None, files_metadata=False):
        path = self.directory(repo_id, revision)
        siblings = [SimpleNamespace(rfilename=p.relative_to(path).as_posix(), size=p.stat().st_size,
                    lfs=SimpleNamespace(sha256=file_hash(p)) if p.suffix == '.safetensors' else None)
                    for p in path.rglob('*') if p.is_file()]
        return SimpleNamespace(sha=path.name, private=self.repos[repo_id]['private'], siblings=siblings)

    def create_commit(self, repo_id, repo_type, parent_commit, operations, commit_message):
        assert parent_commit == self.repos[repo_id]['sha'] and repo_type == 'model'
        sha = f'{len(self.commits)+1:040x}'
        previous = self.directory(repo_id, parent_commit)
        path = self.directory(repo_id, sha)
        shutil.copytree(previous, path)
        for op in operations:
            destination = path / op.path_in_repo
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(op.path_or_fileobj, destination)
        self.repos[repo_id]['sha'] = sha
        self.commits.append((repo_id, [op.path_in_repo for op in operations]))
        return SimpleNamespace(oid=sha)

    def hf_hub_download(self, repo_id, repo_type, revision, filename, local_dir=None, force_download=False):
        assert repo_type == 'model'
        path = self.directory(repo_id, revision) / filename
        self.downloads.append((repo_id, revision, filename, local_dir))
        if local_dir:
            destination = Path(local_dir) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if filename == self.corrupt_download:
                destination.write_bytes(b'broken')
            return str(destination)
        return str(path)


@pytest.fixture
def source(tmp_path, monkeypatch):
    config = load_config(ROOT / 'configs/qwen_transfer_eval.yaml')
    targets = yaml.safe_load((ROOT / 'configs/qwen_transfer_hub.yaml').read_text())
    for key, entry in config['models'].items():
        if 'path' in entry:
            path = fixture_checkpoint(tmp_path / 'source', key, config)
            (path / 'optimizer.pt').write_bytes(b'do not upload')
            (path / '.env').write_text('HF_TOKEN=secret')
    api = FakeHub(tmp_path / 'remote')
    monkeypatch.setattr(hub, 'check_space', lambda *a, **kw: kw)
    return config, targets, tmp_path, api


def publish(source):
    config, targets, root, api = source
    return hub.upload_all(config, targets, root / 'source/experiment', root / 'source/instruct-rl',
                          root / 'receipts', token='fake', api=api)


def download(source, receipt, selected=None, output_name='eval'):
    config, _, root, api = source
    return hub.download_checkpoints(config, root / 'pod', root / 'pod/experiment', root / 'pod/instruct-rl',
        root / 'pod' / output_name, receipt['repo_id'], selected=selected, token='fake', api=api)


def test_seven_uploads_catalog_native_files_and_rerun(source):
    receipt = publish(source)
    config, _, root, api = source
    assert len(api.repos) == 7 and all(repo['private'] for repo in api.repos.values())
    assert len(api.commits) == 8  # Seven models plus their shared catalog in the first repo.
    assert len(receipt['catalog']['models']) == 7
    for key, record in receipt['catalog']['models'].items():
        path = api.directory(record['repo_id'], record['revision'])
        descriptor = json.loads((path / hub.DESCRIPTOR).read_text())
        assert digest(descriptor) == record['descriptor_digest']
        assert descriptor['spec'] == config['models'][key]
        assert (path / hub.TRAIN_MANIFEST).is_file()
        assert (path / 'chat_template.jinja').read_text() == 'native template'
        assert not (path / 'optimizer.pt').exists() and not (path / '.env').exists()
        saved = json.loads((path / 'tokenizer_config.json').read_text())
        assert saved['eos_token'] == ('<|im_end|>' if key in ['kimi_sft', 'instruct_rl'] else '<|endoftext|>')
    assert json.loads((root / 'receipts/upload_complete.json').read_text()) == receipt
    again = publish(source)
    assert again == receipt and len(api.commits) == 8  # No endless self-referential catalog commits.


def test_hf_download_restores_all_validation_paths_and_pins_resume(source):
    receipt = publish(source)
    config, _, root, api = source
    paths = download(source, receipt)
    assert len(paths) == 7
    runs = prepare_models(config, root / 'pod/experiment', root / 'pod/instruct-rl', root / 'pod/eval')
    assert len(runs) == 9
    assert not any((Path(path) / hub.DESCRIPTOR).exists() for path in paths)
    pinned = json.loads((root / 'pod/eval/hf_checkpoint_catalog.json').read_text())
    assert pinned['revision'] == receipt['revision']
    before = len([d for d in api.downloads if d[-1]])
    # Simulate a changed main branch after the first run: local resume must not resolve it.
    api.repos[receipt['repo_id']]['sha'] = 'f' * 40
    assert download(source, receipt) == paths
    assert len([d for d in api.downloads if d[-1]]) == before
    assert all(revision != 'main' for _, revision, _, _ in api.downloads)


def test_download_corruption_is_rejected_and_retry_repairs_it(source):
    receipt = publish(source)
    _, _, _, api = source
    api.corrupt_download = 'model.safetensors'
    with pytest.raises(ValueError, match='checksum mismatch'):
        download(source, receipt, ['base_rl'])
    api.corrupt_download = None
    assert len(download(source, receipt, ['base_rl'])) == 1


def test_missing_checkpoint_blocks_all_uploads(source):
    _, _, root, api = source
    next((root / 'source/instruct-rl').rglob('dapo_checkpoint.json')).unlink()
    with pytest.raises(FileNotFoundError):
        publish(source)
    assert not api.repos


def test_interrupted_upload_keeps_receipts_and_reuses_completed_models(source, monkeypatch):
    _, targets, root, api = source
    original = api.create_commit
    failed = False

    def interrupted(**kwargs):
        nonlocal failed
        if kwargs['repo_id'] == targets['repositories']['distill_sft'] and not failed:
            failed = True
            raise ConnectionError('temporary upload interruption')
        return original(**kwargs)

    monkeypatch.setattr(api, 'create_commit', interrupted)
    with pytest.raises(ConnectionError, match='interruption'):
        publish(source)
    saved = json.loads((root / 'receipts/uploaded_models.json').read_text())
    assert set(saved) == {'base_rl', 'kimi_sft', 'rejection_sft'}
    assert not (root / 'receipts/upload_complete.json').exists()
    receipt = publish(source)
    assert len(api.commits) == 8
    assert all(receipt['catalog']['models'][key] == record for key, record in saved.items())


def test_missing_hf_catalog_explains_required_upload(source):
    from huggingface_hub.errors import RepositoryNotFoundError

    config, _, root, _ = source

    def missing(*args, **kwargs):
        raise RepositoryNotFoundError('Repository unavailable')

    with pytest.raises(FileNotFoundError, match='Complete the Colab seven-checkpoint'):
        hub.resolve_catalog(config, root / 'eval', 'Seungjun/missing', 'main', SimpleNamespace(model_info=missing))


def test_unrelated_repo_visibility_and_changed_source_are_rejected(source):
    receipt = publish(source)
    _, targets, root, api = source
    first_repo = targets['repositories']['base_rl']
    api.repos[first_repo]['private'] = False
    with pytest.raises(ValueError, match='visibility'):
        publish(source)
    api.repos[first_repo]['private'] = True
    file = next((root / 'source/experiment/base-rl-native-eos').rglob('chat_template.jinja'))
    file.write_text('different saved template')
    with pytest.raises(ValueError, match='different weights/metadata'):
        publish(source)
    assert len(api.commits) == 8
    # Existing arbitrary content is never adopted as an upload destination.
    repo = receipt['catalog']['models']['base_rl']['repo_id']
    (api.directory(repo, None) / hub.DESCRIPTOR).unlink()
    with pytest.raises(ValueError, match='not empty'):
        publish(source)


def test_base_instruct_only_needs_no_catalog_and_disallows_bad_keys(source):
    config, _, root, api = source
    args = [config, root, root, root, root / 'eval', 'not-uploaded/model']
    assert hub.download_checkpoints(*args, selected=['base', 'instruct'], api=api) == []
    with pytest.raises(ValueError, match='unique known'):
        hub.download_checkpoints(*args, selected=['base', 'base'], api=api)
    for name in ['../config.json', '/config.json', 'training/../../file', 'optimizer.pt', '']:
        assert not hub.inference_name(name)


def test_mismatched_catalog_and_revision_fail(source):
    receipt = publish(source)
    config, _, root, api = source
    download(source, receipt, ['base_rl'])
    with pytest.raises(ValueError, match='catalog changed'):
        hub.download_checkpoints(config, root, root, root, root / 'pod/eval', receipt['repo_id'],
                                 catalog_revision='a' * 40, api=api)
    record = json.loads((root / 'pod/eval/hf_checkpoint_catalog.json').read_text())
    record['catalog']['model_specs_digest'] = 'changed'
    write_json(root / 'pod/eval/hf_checkpoint_catalog.json', record)
    with pytest.raises(ValueError, match='does not match'):
        download(source, receipt)


def test_upload_notebook_bundle_and_safe_preview(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from build_qwen_checkpoint_upload_notebook import FILES, NAME, cells
    notebook = nbformat.read(ROOT / 'notebooks' / NAME, as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells()]
    sources = [c.source for c in notebook.cells if c.cell_type == 'code']
    for source in sources:
        compile(source, 'upload_checkpoint_notebook', 'exec')
    code = '\n'.join(sources)
    assert 'RUN_UPLOAD = True' in code and 'HF_PRIVATE = True' in code
    assert 'drive.mount' in code and 'rclone' not in code
    source = next(s for s in sources if 'BUNDLE = ' in s)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(node.value)))) as archive:
        assert set(archive.namelist()) == set(FILES)
        for name in FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(tmp_path)
    subprocess.run([sys.executable, '-c', 'from eval.transfer_hub import upload_all, download_checkpoints'],
                   cwd=tmp_path, check=True, capture_output=True)
