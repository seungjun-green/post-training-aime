import json
import sys
from pathlib import Path

import pytest

from common.io import write_json
from eval.run_amc_math_eval import comparison_settings
from train.qwen_self_rft_data import training_code_digest
from train.self_rft_epochs import run_epochs, validate_epoch
from train.self_rft_display import run_progress
from train.stage1_qwen_self_rft import load_config
from test_qwen_self_rft_sft import prepare_fixture, ROOT


def test_alternating_epochs_and_resume_after_evaluation_failure(tmp_path, capsys):
    path, _, _ = prepare_fixture(tmp_path)
    config = load_config(path)
    root = Path(config['output_root'])
    logs = root / 'logs/stage1' / config['run_name']
    write_json(logs / 'preparation/data_report.json', {'optimizer_steps_per_epoch': 1})
    write_json(logs / 'data_report.json', {'optimizer_steps_per_epoch': 1})
    write_json(logs / 'run_manifest.json', {'identity': 'trained', 'config': config,
        'training_code_digest': training_code_digest()})
    _, resolved, _, _ = comparison_settings(ROOT / 'configs/qwen_self_rft_eval.yaml')
    calls = []
    fail = True
    def runner(command, **kwargs):
        if kwargs['mode'] == 'training':
            epoch = int(command[command.index('--stop-after-epoch') + 1])
            calls.append(('train', epoch))
            assert '--auto-resume' in command
            assert kwargs['step_offset'] == epoch - 1 and kwargs['total'] == 1
            checkpoint = root / 'checkpoints/stage1' / config['run_name'] / f'epoch_{epoch}'
            write_json(checkpoint / 'stage1_checkpoint.json', {
                'epoch': epoch, 'global_step': epoch, 'run_identity': 'trained'})
            write_json(checkpoint / 'config.json', {'model_type': 'qwen2', 'eos_token_id': 151643})
            write_json(checkpoint / 'generation_config.json', {'eos_token_id': 151643})
            write_json(checkpoint / 'tokenizer_config.json', {
                'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'})
            (checkpoint / 'chat_template.jinja').write_text('native')
        else:
            checkpoint = Path(command[command.index('--model') + 1])
            epoch = int(checkpoint.name.split('_')[-1])
            calls.append(('eval', epoch))
            assert kwargs['total'] == 540 and '--bundled-code' in command
            if epoch == 2 and fail:
                raise RuntimeError('Interrupted evaluation')
            output = Path(command[command.index('--output-root') + 1])
            name = command[command.index('--run-name') + 1]
            write_json(output / 'full/results' / f'{name}.json', {
                'model': str(checkpoint), 'run_name': name, 'mode': 'full', 'config': resolved,
                'metrics': {b: {'problems': n, 'responses': n, 'samples_per_problem': 1,
                               'avg@1': .5, 'response_length_tokens': {'all': 100.}}
                            for b, n in [('amc23', 40), ('math_500', 500)]}})
    kwargs = dict(code_root=ROOT, train_python='train-python', eval_python='eval-python', runner=runner)
    with pytest.raises(RuntimeError, match='Interrupted'):
        run_epochs(path, **kwargs)
    assert calls == [('train', 1), ('eval', 1), ('train', 2), ('eval', 2)]
    calls.clear()
    fail = False
    rows = run_epochs(path, **kwargs)
    assert calls == [('eval', 1), ('eval', 2), ('train', 3), ('eval', 3),
                     ('train', 4), ('eval', 4), ('train', 5), ('eval', 5)]
    assert len(rows) == 10 and [r['Epoch'] for r in rows] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    summary = root / 'eval/amc2023_math500_temp0' / config['run_name'] / 'epoch_summary.json'
    assert json.loads(summary.read_text())['rows'] == rows
    assert 'Epoch 5/5 results' in capsys.readouterr().out
    calls.clear()
    run_epochs(path, **kwargs)
    assert calls == [('eval', e) for e in range(1, 6)]  # Evaluator handles cached problems.
    checkpoint = root / 'checkpoints/stage1' / config['run_name'] / 'epoch_5'
    write_json(checkpoint / 'generation_config.json', {'eos_token_id': 151645})
    with pytest.raises(ValueError, match='native EOS'):
        validate_epoch(checkpoint, config, 5)


def test_tqdm_display_hides_step_dicts_but_keeps_logs_and_errors(tmp_path, monkeypatch, capsys):
    import tqdm.auto
    bars = []
    class Bar:
        def __init__(self, total, **kwargs):
            self.total, self.n = total, 0
            self.postfix = {}
            bars.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, amount):
            self.n += amount
        def set_postfix(self, **kwargs):
            self.postfix = kwargs
    monkeypatch.setattr(tqdm.auto, 'tqdm', Bar)
    event = {'step': 4, 'loss': .12345, 'learning_rate': 1e-5}
    script = 'print("verbose model loading"); print(' + repr('SELF_RFT_PROGRESS ' + json.dumps(event)) + ')'
    log = tmp_path / 'train.log'
    run_progress([sys.executable, '-c', script], cwd=tmp_path, log_path=log,
                 description='Training', total=2, mode='training', step_offset=2)
    assert capsys.readouterr().out == ''
    assert bars[-1].n == 2 and bars[-1].postfix['loss'] == '0.1235'
    assert 'verbose model loading' in log.read_text() and 'SELF_RFT_PROGRESS' in log.read_text()
    run_progress([sys.executable, '-c', 'print("amc23: 40/40 problems"); print("math_500: 500/500 problems")'],
                 cwd=tmp_path, log_path=tmp_path / 'eval.log', description='Eval', total=540, mode='evaluation')
    assert bars[-1].n == 540
    monkeypatch.setenv('HF_TOKEN', 'secret-for-test')
    with pytest.raises(RuntimeError, match='REDACTED') as error:
        run_progress([sys.executable, '-c', 'print("fatal secret-for-test"); raise SystemExit(1)'],
                     cwd=tmp_path, log_path=log, description='Training', total=2, mode='training')
    assert 'secret-for-test' not in str(error.value) and 'secret-for-test' not in log.read_text()
