import ast
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import nbformat
import pytest
import yaml

from common.io import digest, read_jsonl, write_json, write_jsonl
from train import qwen_distill_continuation as continuation
from train.qwen_self_rft_data import configure_tokenizer, format_example, prepare_examples
from train.sft_data import AssistantCollator, load_rows, percentile
from train.stage1_sft import build_arguments, tokenizer_identity
from test_qwen_7b_distill_sft import prepare_fixture, config as base_config, ROOT
from test_qwen_sft import tokenizer


def parent_fixture(tmp_path, monkeypatch):
    path, template, _ = prepare_fixture(tmp_path)
    parent = yaml.safe_load(path.read_text())
    monkeypatch.setattr(continuation, 'expected_parent_config', lambda: yaml.safe_load(template.read_text()))
    rows, ids = load_rows(parent)
    tok = configure_tokenizer(tokenizer(), parent)
    examples, report, _ = prepare_examples(tok, rows, ids, parent)
    lengths = [sum(v != -100 for v in e['labels']) for e in examples]
    report.update(tokenizer=tokenizer_identity(tok), optimizer_steps_per_epoch=1, total_optimizer_steps=5,
        supervised_tokens_including_eot={'mean': sum(lengths)/len(lengths),
            'median': percentile(lengths, .5), 'min': min(lengths), 'max': max(lengths)})
    manifest = {'config': parent, 'data_report_digest': digest(report), 'tokenizer': tokenizer_identity(tok)}
    identity = digest(manifest)
    root = Path(parent['output_root'])
    logs = root / 'logs/stage1/test'
    write_json(logs / 'data_report.json', report)
    write_json(logs / 'run_manifest.json', {**manifest, 'identity': identity})
    checkpoint = root / 'checkpoints/stage1/test/epoch_5'
    write_json(checkpoint / 'stage1_checkpoint.json', {'epoch': 5, 'global_step': 5, 'run_identity': identity})
    write_json(checkpoint / 'trainer_state.json', {'epoch': 5, 'global_step': 5, 'max_steps': 5,
                                                'num_train_epochs': 5})
    for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'config.json', 'tokenizer_config.json',
                 'model.safetensors']:
        (checkpoint / name).write_text('fixture')
    return checkpoint, parent, report


def test_preparation_preserves_parent_and_validates_data_and_provenance(tmp_path, monkeypatch):
    checkpoint, parent, report = parent_fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in Path(parent['output_root']).rglob('*') if p.is_file()}
    output = tmp_path / 'continuation'
    path = continuation.prepare_continuation(checkpoint, output, 'continue')
    cfg, manifest, previous = continuation.validate_config(path)
    assert cfg['training']['num_train_epochs'] == 7 and cfg['training']['learning_rate'] == 1e-6
    assert cfg['training']['warmup_ratio'] == 0 and cfg['continuation']['start_epoch'] == 6
    assert read_jsonl(cfg['data']['path']) == read_jsonl(parent['data']['path'])
    new_report = deepcopy(report)
    new_report.update(dataset=cfg['data'], total_optimizer_steps=7)
    continuation.validate_prepared_report(new_report, manifest, previous)
    new_report['examples_digest'] = 'changed'
    with pytest.raises(ValueError, match='Prepared data'):
        continuation.validate_prepared_report(new_report, manifest, previous)
    assert continuation.prepare_continuation(checkpoint, output, 'continue') == path
    assert before == {str(p): p.read_bytes() for p in Path(parent['output_root']).rglob('*') if p.is_file()}
    for root in [parent['output_root'], str(Path(parent['output_root']) / 'child'), str(tmp_path)]:
        with pytest.raises(ValueError, match='separate'):
            continuation.prepare_continuation(checkpoint, root, 'continue')
    cfg['training']['gradient_accumulation_steps'] = 8
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='preserve parent settings'):
        continuation.validate_config(path)
    (checkpoint / 'optimizer.pt').unlink()
    with pytest.raises(FileNotFoundError, match='Incomplete checkpoint'):
        continuation.validate_parent(checkpoint)


def test_reject_incomplete_parent_and_changed_snapshot(tmp_path, monkeypatch):
    checkpoint, parent, _ = parent_fixture(tmp_path, monkeypatch)
    state_path = checkpoint / 'trainer_state.json'
    state = json.loads(state_path.read_text())
    write_json(state_path, {**state, 'epoch': 4.5})
    with pytest.raises(ValueError, match='completed five-epoch'):
        continuation.validate_parent(checkpoint)
    write_json(state_path, state)
    rows = read_jsonl(parent['data']['path'])
    rows[0]['response'] += ' changed'
    write_jsonl(parent['data']['path'], rows)
    with pytest.raises(ValueError):
        continuation.prepare_continuation(checkpoint, tmp_path / 'continuation', 'continue')


def test_retry_before_epoch6_archives_abandoned_steps(tmp_path):
    logs, checkpoints = tmp_path / 'logs', tmp_path / 'checkpoints'
    checkpoints.mkdir()
    cfg = {'continuation': {'parent_checkpoint': '/parent/epoch_5'}}
    assert continuation.select_resume(cfg, logs, checkpoints, 'same') == '/parent/epoch_5'
    write_json(logs / 'run_manifest.json', {'identity': 'same'})
    write_jsonl(logs / 'steps.jsonl', [{'step': 11781, 'loss': .2}])
    assert continuation.select_resume(cfg, logs, checkpoints, 'same') == '/parent/epoch_5'
    assert read_jsonl(logs / 'steps.jsonl') == []
    assert len(list(logs.glob('abandoned_steps_*.jsonl'))) == 1
    with pytest.raises(ValueError, match='changed'):
        continuation.select_resume(cfg, logs, checkpoints, 'different')


def test_continuation_prepare_only_entrypoint(tmp_path, monkeypatch):
    checkpoint, _, _ = parent_fixture(tmp_path, monkeypatch)
    path = continuation.prepare_continuation(checkpoint, tmp_path / 'continuation', 'continue')
    tok_dir = tmp_path / 'tokenizer'
    tokenizer().save_pretrained(tok_dir)
    from train import stage1_qwen_distill_continue
    monkeypatch.setattr(sys, 'argv', ['prepare', '--config', str(path), '--prepare-only',
                                   '--local-tokenizer', str(tok_dir)])
    stage1_qwen_distill_continue.main()
    report_path = tmp_path / 'continuation/logs/stage1/continue/preparation/data_report.json'
    report = json.loads(report_path.read_text())
    assert report['kept_rows'] == 3 and report['total_optimizer_steps'] == 7


def test_epoch6_evaluation_failure_resumes_before_training_epoch7(tmp_path, monkeypatch, capsys):
    from train.qwen_distill_continue_epochs import run_epochs, validate_epoch
    from eval.run_amc_math_eval import comparison_settings
    checkpoint, _, _ = parent_fixture(tmp_path, monkeypatch)
    path = continuation.prepare_continuation(checkpoint, tmp_path / 'continuation', 'continue')
    cfg, _, _ = continuation.validate_config(path)
    root = Path(cfg['output_root'])
    logs = root / 'logs/stage1/continue'
    write_json(logs / 'preparation/data_report.json', {'optimizer_steps_per_epoch': 1})
    write_json(logs / 'data_report.json', {'optimizer_steps_per_epoch': 1})
    write_json(logs / 'run_manifest.json', {'identity': 'trained', 'config': cfg,
                                         'training_code_digest': continuation.training_code_digest()})
    _, resolved, _, _ = comparison_settings(ROOT / 'configs/qwen_self_rft_eval.yaml')
    calls, fail = [], True
    def runner(command, **kwargs):
        if kwargs['mode'] == 'training':
            epoch = int(command[command.index('--stop-after-epoch') + 1])
            calls.append(('train', epoch))
            assert 'train.stage1_qwen_distill_continue' in command
            assert kwargs['step_offset'] == epoch - 1 and kwargs['total'] == 1
            saved = root / 'checkpoints/stage1/continue' / f'epoch_{epoch}'
            write_json(saved / 'stage1_checkpoint.json', {'epoch': epoch, 'global_step': epoch,
                                                         'run_identity': 'trained'})
            write_json(saved / 'config.json', {'model_type': 'qwen2', 'eos_token_id': 151643})
            write_json(saved / 'generation_config.json', {'eos_token_id': 151643})
            write_json(saved / 'tokenizer_config.json', {'eos_token': '<|endoftext|>',
                                                        'pad_token': '<|endoftext|>'})
            (saved / 'chat_template.jinja').write_text('native')
        else:
            saved = Path(command[command.index('--model') + 1])
            epoch = int(saved.name.split('_')[-1])
            calls.append(('eval', epoch))
            assert kwargs['total'] == 540 and '--bundled-code' in command
            if fail:
                raise RuntimeError('Interrupted evaluation')
            destination = Path(command[command.index('--output-root') + 1])
            name = command[command.index('--run-name') + 1]
            write_json(destination / 'full/results' / f'{name}.json', {
                'model': str(saved), 'run_name': name, 'config': resolved, 'mode': 'full',
                'metrics': {b: {'problems': n, 'responses': n, 'samples_per_problem': 1,
                               'avg@1': .5, 'response_length_tokens': {'all': 100.}}
                            for b, n in [('amc23', 40), ('math_500', 500)]}})
    kwargs = dict(code_root=ROOT, train_python='train', eval_python='eval', runner=runner)
    with pytest.raises(RuntimeError, match='Interrupted evaluation'):
        run_epochs(path, **kwargs)
    assert calls == [('train', 6), ('eval', 6)]
    calls.clear()
    fail = False
    rows = run_epochs(path, **kwargs)
    assert calls == [('eval', 6), ('train', 7), ('eval', 7)]
    assert [row['Epoch'] for row in rows] == [6, 6, 7, 7]
    assert 'Epoch 7/7 results' in capsys.readouterr().out
    calls.clear()
    run_epochs(path, **kwargs)
    assert calls == [('eval', 6), ('eval', 7)]
    summary = root / 'eval/amc2023_math500_temp0/continue/epoch_summary.json'
    assert json.loads(summary.read_text())['rows'] == rows
    saved = root / 'checkpoints/stage1/continue/epoch_7'
    write_json(saved / 'generation_config.json', {'eos_token_id': 151645})
    with pytest.raises(ValueError, match='native EOS'):
        validate_epoch(saved, cfg, 7)


def test_real_optimizer_scheduler_and_exact_epoch6_resume(tmp_path):
    torch = pytest.importorskip('torch')
    from datasets import Dataset
    from transformers import Qwen2Config, Qwen2ForCausalLM, AutoModelForCausalLM, TrainerCallback, set_seed
    from train.qwen_sft_trainer import QwenSFTTrainer
    from train.qwen_continuation_trainer import QwenContinuationTrainer
    from train.self_rft_callbacks import install_progress

    torch.set_num_threads(1)
    set_seed(42)
    cfg = base_config()
    cfg['training'].update(bf16=False, tf32=None, use_cpu=True, gradient_accumulation_steps=2,
                           dataloader_pin_memory=False, disable_tqdm=True)
    tok = configure_tokenizer(tokenizer(), cfg)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, attention_dropout=0,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True))
    data = Dataset.from_list([format_example(tok, {'problem': 'Q', 'response': answer})[0]
                             for answer in ['x y', 'y x', 'x']])
    def make(cls, model, name, **kwargs):
        return cls(model=model, args=build_arguments(cfg, tmp_path / name, tmp_path / (name + '_logs')),
            processing_class=tok, train_dataset=data, data_collator=AssistantCollator(tok.pad_token_id),
            log_dir=tmp_path / (name + '_logs'), run_identity=name, lm_head_chunk_tokens=2, **kwargs)
    parent = make(QwenSFTTrainer, model, 'parent')
    parent.train()
    checkpoint = tmp_path / 'parent/epoch_5'
    assert parent.state.global_step == 10 and parent.lr_scheduler.get_last_lr() == [0., 0.]
    parent_bytes = {p.name: p.read_bytes() for p in checkpoint.iterdir() if p.is_file()}
    saved_optimizer = torch.load(checkpoint / 'optimizer.pt', weights_only=True)
    cfg['training'].update(num_train_epochs=7, learning_rate=1e-6, warmup_ratio=0.)
    class Observe(TrainerCallback):
        def __init__(self):
            self.lrs = []
        def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            self.start_step = state.global_step
            self.scheduler_step = lr_scheduler.last_epoch
            if state.global_step == 10:
                loaded = optimizer.state_dict()
                for key in saved_optimizer['state']:
                    for field in ['step', 'exp_avg', 'exp_avg_sq']:
                        torch.testing.assert_close(loaded['state'][key][field],
                            saved_optimizer['state'][key][field], rtol=0, atol=0)
        def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
            self.lrs.append(optimizer.param_groups[0]['lr'])
    def continuation_trainer(name, source, stop=None):
        obj = make(QwenContinuationTrainer, AutoModelForCausalLM.from_pretrained(source), name,
                   parent_checkpoint=checkpoint, additional_steps=4)
        observed = Observe()
        obj.add_callback(observed)
        install_progress(obj, stop)
        obj.train(resume_from_checkpoint=str(source))
        return obj, observed
    full, full_observed = continuation_trainer('full', checkpoint)
    assert full.state.global_step == 14 and full.state.epoch == 7
    assert full_observed.start_step == 10 and full_observed.scheduler_step == 0
    assert full_observed.lrs[0] == 1e-6
    assert full.lr_scheduler.last_epoch == 4 and full.lr_scheduler.get_last_lr() == [0., 0.]
    partial, first = continuation_trainer('split', checkpoint, 6)
    assert partial.state.global_step == 12 and partial.state.epoch == 6
    resumed, second = continuation_trainer('split', tmp_path / 'split/epoch_6', 7)
    assert second.start_step == 12 and second.scheduler_step == 2
    assert second.lrs[0] == pytest.approx(5e-7)
    assert first.lrs + second.lrs == full_observed.lrs
    assert resumed.state.global_step == 14
    for a, b in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert sorted(p.name for p in (tmp_path / 'split').glob('epoch_*')) == ['epoch_6', 'epoch_7']
    assert [r['step'] for r in read_jsonl(tmp_path / 'split_logs/steps.jsonl')] == [11, 12, 13, 14]
    assert parent_bytes == {p.name: p.read_bytes() for p in checkpoint.iterdir() if p.is_file()}


def test_notebook_embedded_bundle_and_commands(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from build_qwen_7b_distill_continue_notebook import cells
    notebook = ROOT / 'notebooks/continue_qwen25_3b_7b_distill_epoch5_to7.ipynb'
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells] == [c.source for c in cells()]
    sources = [c.source for c in nb.cells if c.cell_type == 'code']
    for source in sources:
        compile(source, 'continuation_notebook', 'exec')
        assert '@param' not in source
    context = {'Path': Path, 'json': json}
    exec(sources[0], context)
    assert context['PARENT_CHECKPOINT'].endswith('/sft_qwen25_3b_base_7b_distill/epoch_5')
    assert context['TRAIN_ROOT'].endswith('/base-7b-distillation-sft-epoch5-to7')
    context.update(TRAIN_ROOT=str(tmp_path), CODE_ROOT=ROOT, TRAIN_PYTHON='train-python', EVAL_PYTHON='eval-python')
    calls = []
    context['run_logged'] = lambda cmd, **kw: calls.append(cmd)
    write_json(tmp_path / 'logs/stage1' / context['RUN_NAME'] / 'preparation/data_report.json', {
        'kept_rows': 37691, 'optimizer_steps_per_epoch': 2356})
    exec(next(s for s in sources if 'CONFIG_PATH =' in s), context)
    assert 'train.qwen_distill_continuation' in calls[0]
    assert 'train.stage1_qwen_distill_continue' in calls[1] and '--prepare-only' in calls[1]
    from train import qwen_distill_continue_epochs
    monkeypatch.setattr(qwen_distill_continue_epochs, 'run_epochs', lambda *a, **k: calls.append(k))
    run = next(s for s in sources if 'RUN_TRAINING = False' in s)
    exec(run, context)
    assert len(calls) == 2
    exec(run.replace('RUN_TRAINING = False', 'RUN_TRAINING = True'), context)
    assert calls[-1]['eval_python'] == 'eval-python'
    boot = next(s for s in sources if 'BUNDLE =' in s)
    assign = next(n for n in ast.parse(boot).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))
    bundle = tmp_path / 'isolated'
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assign.value)))) as archive:
        assert set(archive.namelist()) == set(continuation.BUNDLE_FILES)
        for name in archive.namelist():
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(bundle)
    for module in ['train.qwen_distill_continuation', 'train.stage1_qwen_distill_continue',
                   'eval.run_amc_math_eval']:
        subprocess.run([sys.executable, '-m', module, '--help'], cwd=bundle, check=True, capture_output=True)
    subprocess.run([sys.executable, '-c', 'from train.qwen_distill_continuation import training_code_digest; '
                    'assert len(training_code_digest()) == 64'], cwd=bundle, check=True, capture_output=True)
