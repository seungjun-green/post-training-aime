"""Alternate durable SFT epochs with greedy AMC/MATH evaluation on a free GPU."""

import csv
import json
from pathlib import Path

import yaml

from common.io import write_json
from eval.run_amc_math_eval import comparison_settings
from train.qwen_self_rft_data import training_code_digest
from train.self_rft_display import run_progress

EVAL_CONFIG = 'configs/qwen_self_rft_eval.yaml'
TOTALS = {'amc23': 40, 'math_500': 500}


def validate_epoch(checkpoint, config, epoch):
    root = Path(config['output_root'])
    logs = root / 'logs/stage1' / config['run_name']
    marker = json.loads((checkpoint / 'stage1_checkpoint.json').read_text())
    manifest = json.loads((logs / 'run_manifest.json').read_text())
    report = json.loads((logs / 'data_report.json').read_text())
    if (marker['epoch'] != epoch or marker['run_identity'] != manifest['identity']
            or marker['global_step'] != epoch * report['optimizer_steps_per_epoch']
            or manifest['config'] != config
            or manifest['training_code_digest'] != training_code_digest()):
        raise ValueError('Epoch checkpoint does not match this training run/code')
    model = json.loads((checkpoint / 'config.json').read_text())
    generation = json.loads((checkpoint / 'generation_config.json').read_text())
    tokenizer = json.loads((checkpoint / 'tokenizer_config.json').read_text())
    stops = generation.get('eos_token_id')
    if (model.get('model_type') != 'qwen2' or model.get('eos_token_id') != 151643
            or (stops if isinstance(stops, list) else [stops]) != [151643]):
        raise ValueError('Checkpoint must preserve Qwen base native EOS 151643')
    for key in ['eos_token', 'pad_token']:
        token = tokenizer.get(key)
        token = token.get('content') if isinstance(token, dict) else token
        if token != '<|endoftext|>':
            raise ValueError(f'Checkpoint changed {key}')
    template = checkpoint / 'chat_template.jinja'
    if not (template.read_text().strip() if template.is_file() else tokenizer.get('chat_template')):
        raise ValueError('Missing saved chat template')
    return marker


def result_rows(path, checkpoint, run_name, resolved, epoch):
    result = json.loads(path.read_text())
    if (result['model'] != str(checkpoint) or result['run_name'] != run_name
            or result['config'] != resolved or result['mode'] != 'full'):
        raise ValueError('Saved evaluation does not match the epoch/config')
    rows = []
    for name, total in TOTALS.items():
        metric = result['metrics'][name]
        if (metric['problems'] != total or metric['responses'] != total
                or metric['samples_per_problem'] != 1):
            raise ValueError(f'Incomplete epoch evaluation: {name}')
        rows.append({'Epoch': epoch, 'Benchmark': 'AMC 2023' if name == 'amc23' else 'MATH-500',
                     'Problems': total, 'Correct': round(metric['avg@1'] * total),
                     'Accuracy (%)': 100 * metric['avg@1'],
                     'Mean tokens': metric['response_length_tokens']['all']})
    return rows


def run_epochs(config_path, *, code_root, train_python, eval_python, runner=run_progress):
    config_path, code_root = Path(config_path), Path(code_root)
    config = yaml.safe_load(config_path.read_text())
    root, name = Path(config['output_root']), config['run_name']
    logs, checkpoints = root / 'logs/stage1' / name, root / 'checkpoints/stage1' / name
    preparation = json.loads((logs / 'preparation/data_report.json').read_text())
    updates = preparation['optimizer_steps_per_epoch']
    epochs = config['training']['num_train_epochs']
    _, resolved, _, _ = comparison_settings(code_root / EVAL_CONFIG)
    eval_root = root / 'eval/amc2023_math500_temp0' / name
    all_rows = []
    for epoch in range(1, epochs + 1):
        checkpoint = checkpoints / f'epoch_{epoch}'
        if not (checkpoint / 'stage1_checkpoint.json').is_file():
            runner([str(train_python), '-m', 'train.stage1_qwen_self_rft', '--config', str(config_path),
                    '--auto-resume', '--stop-after-epoch', str(epoch)],
                   cwd=code_root, log_path=root / name / 'train_console.log',
                   description=f'Epoch {epoch}/{epochs} training', total=updates,
                   mode='training', step_offset=(epoch - 1) * updates)
        validate_epoch(checkpoint, config, epoch)
        destination = eval_root / f'epoch_{epoch}'
        run_name = f'{name}_epoch{epoch}'
        # The training process has exited, freeing all model/optimizer CUDA memory.
        # The existing evaluator validates resume identity and reuses completed problems.
        runner([str(eval_python), '-m', 'eval.run_amc_math_eval', '--model', str(checkpoint),
                '--run-name', run_name, '--config', EVAL_CONFIG, '--output-root', str(destination),
                '--bundled-code'], cwd=code_root, log_path=destination / 'logs/eval_console.log',
               description=f'Epoch {epoch}/{epochs} AMC + MATH-500', total=540, mode='evaluation')
        result_path = destination / 'full/results' / f'{run_name}.json'
        rows = result_rows(result_path, checkpoint, run_name, resolved, epoch)
        all_rows.extend(rows)
        write_json(destination / 'full/summary_table.json', {'checkpoint': str(checkpoint), 'rows': rows})
        write_json(eval_root / 'epoch_summary.json', {'temperature': 0, 'rows': all_rows})
        with (eval_root / 'epoch_summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f'\nEpoch {epoch}/{epochs} results (temperature 0)', flush=True)
        print(f"{'Benchmark':<12} {'Correct':>9} {'Accuracy':>10} {'Mean tokens':>12}", flush=True)
        for row in rows:
            correct = f"{row['Correct']}/{row['Problems']}"
            print(f"{row['Benchmark']:<12} {correct:>9} {row['Accuracy (%)']:>9.1f}% "
                  f"{row['Mean tokens']:>12.1f}", flush=True)
        print('Metrics:', result_path, flush=True)
    print('\nAll five epochs and evaluations are complete. Summary:', eval_root / 'epoch_summary.csv')
    return all_rows
