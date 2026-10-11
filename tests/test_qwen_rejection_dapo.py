import ast
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import nbformat
import pytest
import yaml

from common.io import digest, write_json
from train.dapo_data import batch_schedule, load_config
from train.dapo_model import configure_tokenizer, stop_token_ids
from train.run_qwen_rejection_dapo import select_rejection_model
from test_qwen_base_dapo import native_tokens

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/dapo_qwen25_3b_rejection.yaml'
NOTEBOOK = ROOT / 'notebooks/train_dapo_qwen25_3b_rejection_epoch4.ipynb'


def source_fixture(root):
    cfg = load_config(CONFIG)
    spec = cfg['sft']
    path = root / spec['relative_checkpoint']
    report = {'optimizer_steps_per_epoch': 1767, 'kept_rows': 28267}
    previous = yaml.safe_load((ROOT / 'configs/sft_qwen25_3b_self_rft.yaml').read_text())
    tok = native_tokens()
    manifest = {'config': previous, 'data_report_digest': digest(report), 'tokenizer': {
        'chat_template_digest': digest(tok.chat_template), 'tokenizer_vocab_digest': digest(tok.get_vocab())}}
    identity = digest(manifest)
    logs = root / 'logs/stage1' / spec['run_name']
    write_json(logs / 'run_manifest.json', {**manifest, 'identity': identity})
    write_json(logs / 'data_report.json', report)
    write_json(path / 'stage1_checkpoint.json', {'epoch': 4, 'global_step': 7068, 'run_identity': identity})
    write_json(path / 'trainer_state.json', {'epoch': 4, 'global_step': 7068})
    write_json(path / 'config.json', {'model_type': 'qwen2', 'eos_token_id': 151643, 'pad_token_id': 151643})
    write_json(path / 'generation_config.json', {'eos_token_id': 151643, 'pad_token_id': 151643})
    write_json(path / 'tokenizer_config.json', {'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'})
    write_json(path / 'tokenizer.json', {'model': {'vocab': tok.get_vocab()}, 'added_tokens': []})
    (path / 'chat_template.jinja').write_text(tok.chat_template)
    header = json.dumps({'w': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    (path / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
    return cfg, path


def test_source_selection_preserves_native_tokens_and_rl_settings(tmp_path):
    root = tmp_path / 'sft'
    cfg, path = source_fixture(root)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    source, identity = select_rejection_model(cfg, 'sft', root, tmp_path / 'rl')
    assert source == str(path) and identity['marker']['epoch'] == 4
    assert identity['kind'] == 'sft' and identity['parent_run_identity']
    assert not (path / 'optimizer.pt').exists()  # SFT training state is not an RL resume.
    base = load_config(ROOT / 'configs/dapo_qwen25_3b_base.yaml')
    for key in ['model', 'data', 'algorithm', 'training', 'rollout', 'hardware', 'smoke']:
        assert cfg[key] == base[key]
    assert batch_schedule(cfg)['updates_per_rollout'] == 2
    assert cfg['training']['max_steps'] == 300 and cfg['training']['save_steps'] == 20
    tok = configure_tokenizer(native_tokens(), cfg)
    assert tok.eos_token_id == tok.pad_token_id == 151643 and stop_token_ids(tok, cfg) is None
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    for output in [root, root / 'child', tmp_path]:
        with pytest.raises(ValueError, match='separate OUTPUT_ROOT'):
            select_rejection_model(cfg, 'sft', root, output)
    with pytest.raises(ValueError, match='model-kind sft'):
        select_rejection_model(cfg, 'base', root, tmp_path / 'rl')
    changed = deepcopy(cfg)
    changed['sft']['epoch'] = 7
    with pytest.raises(ValueError, match='epoch 4'):
        select_rejection_model(changed, 'sft', root, tmp_path / 'rl')
    # Support Transformers' older inline-template serialization too.
    tokenizer_config = json.loads((path / 'tokenizer_config.json').read_text())
    tokenizer_config['chat_template'] = tok.chat_template
    write_json(path / 'tokenizer_config.json', tokenizer_config)
    (path / 'chat_template.jinja').unlink()
    assert select_rejection_model(cfg, 'sft', root, tmp_path / 'rl')[0] == source


def test_selected_epoch4_is_used_even_when_epoch5_exists(tmp_path):
    root = tmp_path / 'sft'
    cfg, path = source_fixture(root)
    # The completed SFT run also has epoch 5, but the user selected epoch 4.
    write_json(path.parent / 'epoch_5/stage1_checkpoint.json', {
        'epoch': 5, 'global_step': 8835, 'run_identity': 'later-checkpoint'})
    source, identity = select_rejection_model(cfg, 'sft', root, tmp_path / 'rl')
    assert source == str(path) and identity['marker']['global_step'] == 7068
    from test_qwen_distill_dapo import source_fixture as distill_fixture
    distill_root = tmp_path / 'distill'
    wrong, _ = distill_fixture(distill_root)
    with pytest.raises(ValueError, match='rejection-sampling epoch 4'):
        select_rejection_model(wrong, 'sft', distill_root, tmp_path / 'rl')


@pytest.mark.parametrize('corruption', ['eos', 'template', 'vocab', 'shard', 'epoch', 'identity'])
def test_bad_source_rejected(tmp_path, corruption):
    root = tmp_path / 'sft'
    cfg, path = source_fixture(root)
    if corruption == 'eos':
        write_json(path / 'generation_config.json', {'eos_token_id': 151645, 'pad_token_id': 151643})
    elif corruption == 'template':
        (path / 'chat_template.jinja').write_text('changed')
    elif corruption == 'vocab':
        write_json(path / 'tokenizer.json', {'model': {'vocab': {'changed': 0}}})
    elif corruption == 'shard':
        shard = path / 'model.safetensors'
        shard.write_bytes(shard.read_bytes()[:-1])
    else:
        marker_path = path / 'stage1_checkpoint.json'
        marker = json.loads(marker_path.read_text())
        marker['epoch' if corruption == 'epoch' else 'run_identity'] = 7
        write_json(marker_path, marker)
    with pytest.raises(ValueError):
        select_rejection_model(cfg, 'sft', root, tmp_path / 'rl')


def test_training_runner_loads_sft_weights_but_starts_fresh_rl(tmp_path, monkeypatch):
    import torch
    import transformers
    from train import run_dapo, run_qwen_rejection_dapo, dapo_rollout
    from test_qwen_sft import tokenizer

    sft_root, output = tmp_path / 'sft', tmp_path / 'rl'
    cfg, checkpoint = source_fixture(sft_root)
    tok = tokenizer()
    loaded, resumed, rollout_sources = [], [], []
    def load_model(source, **kwargs):
        loaded.append((source, kwargs))
        return SimpleNamespace(config=SimpleNamespace(use_cache=True), parameters=lambda: [torch.nn.Parameter(torch.ones(1))])
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: tok)
    monkeypatch.setattr(run_dapo, 'load_data', lambda *a, **k: ([], []))
    monkeypatch.setattr(run_dapo, 'prepare_data', lambda *a: ([{'id': 'q', 'prompt': 'Q', 'gold': '4'}], {}))
    monkeypatch.setattr(run_dapo, 'check_hardware', lambda c: {})
    monkeypatch.setattr(run_dapo, 'git_identity', lambda: 'test-commit')
    version = run_dapo.importlib.metadata.version
    monkeypatch.setattr(run_dapo.importlib.metadata, 'version', lambda n: '0.14.1' if n == 'vllm' else version(n))
    for name in ['reset_peak_memory_stats', 'max_memory_allocated', 'max_memory_reserved']:
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    class Rollout:
        def __init__(self, source, *a, **k):
            rollout_sources.append(source)
        def close(self):
            pass
    monkeypatch.setattr(dapo_rollout, 'VLLMRollout', Rollout)
    class Trainer:
        def __init__(self, model, **kwargs):
            self.model = model
            self.state = SimpleNamespace(global_step=0)
        def train(self, resume_from_checkpoint):
            resumed.append(resume_from_checkpoint)
            assert self.state.global_step == 0
            self.state.global_step = 300
            return SimpleNamespace(metrics={'loss': .1})
        def _save_checkpoint(self, *a, **kw):
            pass
    monkeypatch.setitem(sys.modules, 'train.dapo_trainer', SimpleNamespace(
        DAPOTrainer=Trainer, make_arguments=lambda *a: None))
    monkeypatch.setattr(sys, 'argv', ['dapo', '--config', str(CONFIG), '--model-kind', 'sft',
        '--sft-root', str(sft_root), '--output-root', str(output), '--work-dir', str(tmp_path / 'work')])
    run_qwen_rejection_dapo.main()
    assert resumed == [None]  # No SFT optimizer, scheduler, RNG or 7068-step Trainer state loaded.
    assert loaded[0][0] == rollout_sources[0] == str(checkpoint)
    assert loaded[0][1]['torch_dtype'] == torch.float32
    manifest = json.loads((output / 'logs' / (cfg['run_name_prefix'] + '_sft') / 'run_manifest.json').read_text())
    assert manifest['source']['marker']['global_step'] == 7068
    assert manifest['config']['training']['max_steps'] == 300


def test_notebook_commands_resume_and_isolated_bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from build_qwen_rejection_dapo_notebook import cells, BUNDLE_FILES

    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in cells()] == [c.source for c in nb.cells]
    codes = [c.source for c in nb.cells if c.cell_type == 'code']
    for source in codes:
        compile(source, 'rejection_rl', 'exec')
        assert '@param' not in source and '"git", "clone"' not in source
    root = tmp_path / 'sft'
    source_fixture(root)
    context = {'Path': Path, 'json': json}
    exec(codes[0], context)
    assert context['SFT_ROOT'].endswith('/base-rejection-sampling-sft')
    context.update(CODE_ROOT=str(ROOT), OUTPUT_ROOT=str(tmp_path / 'rl'), SFT_ROOT=str(root),
                   git=lambda *a: 'test-commit')
    exec(nb.cells[5].source, context)
    command = context['TRAIN_COMMAND']
    assert command[command.index('--model-kind') + 1] == 'sft'
    assert command[command.index('--sft-root') + 1] == str(root)
    assert 'train.run_qwen_rejection_dapo' in command
    calls = []
    def run(command, **kw):
        calls.append(command)
        if '--smoke' in command:
            output = Path(command[command.index('--output-root') + 1])
            log = output / 'logs' / (context['RUN_NAME'] + '_smoke') / 'steps.jsonl'
            log.parent.mkdir(parents=True)
            log.write_text('\n'.join(json.dumps(r) for r in [
                {'policy_iteration': 1, 'minibatch_index': 1, 'reused_rollout': False},
                {'policy_iteration': 1, 'minibatch_index': 2, 'reused_rollout': False,
                 'cached_rollout': True, 'new_generated_tokens': 0}]))
    context['run_logged'] = run
    exec(nb.cells[7].source, context)
    for index in [9, 11]:
        source = nb.cells[index].source
        exec(source, context)
        exec(source.replace('= False', '= True', 1), context)
    assert len(calls) == 3 and '--prepare-only' in calls[0] and '--smoke' in calls[1]
    assert '--resume-from-checkpoint' not in calls[2]
    from test_qwen_base_dapo import make_checkpoint
    def checkpoint(step):
        saved = make_checkpoint(context, step)
        write_json(saved / 'dapo_checkpoint.json', {'run_identity': 'base-run', 'global_step': step, 'model_kind': 'sft'})
        previous_config = deepcopy(context['cfg'])
        previous_config.update(model_kind='sft')
        write_json(context['run_logs'] / 'run_manifest.json', {'identity': 'base-run', 'config': previous_config,
            'source': context['SFT_IDENTITY'], 'git_commit': 'test-commit'})
        return saved
    saved = checkpoint(20)
    full = nb.cells[11].source.replace('RUN_TRAINING = False', 'RUN_TRAINING = True')
    exec(full, context)
    assert calls[-1][-2:] == ['--resume-from-checkpoint', str(saved)]
    checkpoint(300)
    before = len(calls)
    exec(full, context)
    assert len(calls) == before
    with pytest.raises(ValueError, match='latest completed'):
        exec(full.replace('RESUME_CHECKPOINT = ""', 'RESUME_CHECKPOINT = "/other"'), context)
    context['SFT_IDENTITY'] = dict(context['SFT_IDENTITY'], path='/wrong/epoch_7')
    with pytest.raises(ValueError, match='source/settings/code'):
        exec(full, context)
    setup = nb.cells[3].source
    assign = next(n for n in ast.parse(setup).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assign.value)))) as archive:
        assert set(archive.namelist()) == set(BUNDLE_FILES)
        for name in archive.namelist():
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(tmp_path / 'bundle')
    subprocess.run([sys.executable, '-m', 'train.run_qwen_rejection_dapo', '--help'],
                   cwd=tmp_path / 'bundle', check=True, capture_output=True)
