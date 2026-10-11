import ast
import base64
import io
import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import yaml

from common.io import digest, write_json
from eval.qwen_sft_rl import BUNDLE_FILES, LABELS, checkpoint_source, comparison_rows, prepare_runs, result_path
from eval.run_amc_math_eval import comparison_settings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/qwen_sft_rl_eval.yaml'
NOTEBOOK = ROOT / 'notebooks/evaluate_qwen25_3b_sft_rl_checkpoint300_amc_math.ipynb'


def make_checkpoint(root, key):
    spec, _, _, _ = comparison_settings(CONFIG)
    config = yaml.safe_load((ROOT / spec['variants'][key]).read_text())
    config['model_kind'] = 'sft'
    name = config['run_name_prefix'] + '_sft'
    checkpoint = root / 'checkpoints' / name / 'checkpoint-300'
    parent = {'epoch': config['sft']['epoch'], 'global_step': 11780 if key == 'distill_rl' else 7068,
              'run_identity': 'sft-parent'}
    manifest = {'config': config, 'source': {
        'kind': 'sft', 'path': '/absent/sft/' + config['sft']['relative_checkpoint'],
        'marker': parent, 'parent_run_identity': 'sft-parent', 'sft_dataset': config['sft']['hf_source']}}
    save_manifest(root, checkpoint, manifest)
    for file in ['config.json', 'generation_config.json']:
        write_json(checkpoint / file, {'model_type': 'qwen2', 'eos_token_id': 151643, 'pad_token_id': 151643})
    write_json(checkpoint / 'tokenizer_config.json', {'eos_token': '<|endoftext|>',
        'pad_token': {'content': '<|endoftext|>'}, 'chat_template': 'native'})
    write_json(checkpoint / 'tokenizer.json', {})
    header = json.dumps({'w': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    (checkpoint / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
    return checkpoint, manifest


def save_manifest(root, checkpoint, manifest):
    manifest['identity'] = digest({k: v for k, v in manifest.items() if k != 'identity'})
    write_json(root / 'logs' / checkpoint.parent.name / 'run_manifest.json', manifest)
    write_json(checkpoint / 'dapo_checkpoint.json', {
        'global_step': 300, 'model_kind': 'sft', 'run_identity': manifest['identity']})


@pytest.mark.parametrize('key', LABELS)
@pytest.mark.parametrize('change', ['none', 'template_file', 'epoch', 'dataset', 'identity', 'native_eos',
                                   'truncated_weights', 'step', 'missing_template', 'parent', 'override'])
def test_checkpoint_provenance_and_saved_tokenizer(tmp_path, key, change):
    checkpoint, manifest = make_checkpoint(tmp_path, key)
    if change == 'epoch':
        manifest['source']['marker']['epoch'] += 1
    elif change == 'dataset':
        manifest['source']['sft_dataset'] = {'repo': 'wrong'}
    elif change == 'parent':
        manifest['source']['parent_run_identity'] = 'wrong'
    elif change == 'override':
        manifest['config']['generation'] = {'eos_token_id': 151645}
    save_manifest(tmp_path, checkpoint, manifest)
    if change == 'identity':
        manifest['identity'] = 'wrong'
        write_json(tmp_path / 'logs' / checkpoint.parent.name / 'run_manifest.json', manifest)
    elif change == 'native_eos':
        write_json(checkpoint / 'generation_config.json', {'eos_token_id': 151645, 'pad_token_id': 151643})
    elif change == 'truncated_weights':
        path = checkpoint / 'model.safetensors'
        path.write_bytes(path.read_bytes()[:-1])
    elif change == 'step':
        marker = json.loads((checkpoint / 'dapo_checkpoint.json').read_text())
        marker['global_step'] = 280
        write_json(checkpoint / 'dapo_checkpoint.json', marker)
    elif change in ['template_file', 'missing_template']:
        path = checkpoint / 'tokenizer_config.json'
        tokenizer = json.loads(path.read_text())
        tokenizer.pop('chat_template')
        write_json(path, tokenizer)
        if change == 'template_file':
            (checkpoint / 'chat_template.jinja').write_text('native')
    spec, _, _, _ = comparison_settings(CONFIG)
    if change in ['none', 'template_file']:
        assert checkpoint_source(tmp_path, key, spec)[0] == checkpoint
        assert not (checkpoint / 'optimizer.pt').exists()
    else:
        with pytest.raises(ValueError):
            checkpoint_source(tmp_path, key, spec)


def test_notebook_commands_outputs_and_isolated_bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from build_qwen_sft_rl_eval_notebook import cells
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells()]
    sources = [c.source for c in notebook.cells if c.cell_type == 'code']
    for source in sources:
        compile(source, 'eval_notebook', 'exec')
        assert '@param' not in source and '"git", "clone"' not in source
    roots = {key: tmp_path / key for key in LABELS}
    for key, root in roots.items():
        make_checkpoint(root, key)
    context = {'Path': Path, 'json': json}
    exec(sources[0], context)
    assert context['TRAIN_ROOTS']['distill_rl'].endswith('base-7b-distillation-sft-rl')
    assert context['TRAIN_ROOTS']['rejection_rl'].endswith('base-rejection-sampling-sft-rl')
    context.update(TRAIN_ROOTS=roots, OUTPUT_ROOT=str(tmp_path / 'eval'), CODE_ROOT=ROOT)
    exec(next(s for s in sources if 'def evaluation_command' in s), context)
    assert context['resolved']['temperature'] == 0
    assert context['resolved']['top_p'] == 1
    assert context['resolved']['max_new_tokens'] == 20480
    assert context['resolved']['pass_k'] == [1]
    calls = []
    context['run_logged'] = lambda command, **kw: calls.append((command, kw))
    for flag in ['RUN_SMOKE', 'RUN_EVAL']:
        source = next(s for s in sources if f'{flag} = False' in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(f'{flag} = False', f'{flag} = True'), context)
    assert len(calls) == 4
    for i, (command, kw) in enumerate(calls):
        assert command[command.index('--model') + 1] == context['runs'][i % 2]['model']
        assert command[command.index('--output-root') + 1] == context['runs'][i % 2]['output_root']
        assert ('--smoke' in command) == (i < 2)
        assert '--bundled-code' in command and '--revision' not in command
        assert sum(kw['progress_totals'].values()) == (2 if i < 2 else 540)
    # Reruns preserve the selected models; changing identity cannot mix outputs.
    assert prepare_runs(context['spec'], roots, context['eval_root']) == context['runs']
    with pytest.raises(ValueError, match='separate comparison'):
        prepare_runs(context['spec'], roots, roots['distill_rl'] / 'eval')
    selection = context['eval_root'] / 'model_selection.json'
    write_json(selection, [])
    with pytest.raises(ValueError, match='Selected checkpoints changed'):
        prepare_runs(context['spec'], roots, context['eval_root'])
    # Four-row comparison is saved only with compatible, complete results.
    for run in context['runs']:
        result = {k: 'same' for k in ['execution', 'protocol_digest', 'dataset_suite', 'selected_ids',
                  'packages', 'hardware', 'runner_digest', 'bundle_digest', 'chat_template_digest']}
        result.update(mode='full', model=run['model'], model_revision=None,
                      run_name=run['run_name'], config=context['resolved'], metrics={})
        for name, n in context['progress_totals'].items():
            result['metrics'][name] = {'problems': n, 'responses': n, 'samples_per_problem': 1,
                'avg@1': .5, 'response_length_tokens': {'all': 1000}}
        write_json(result_path(run), result)
    monkeypatch.setitem(sys.modules, 'IPython.display', SimpleNamespace(display=lambda _: None))
    context['show_comparison']()
    table = json.loads((context['eval_root'] / 'comparison_table.json').read_text())
    assert [row['Correct'] for row in table['rows']] == [20, 250, 20, 250]
    assert (context['eval_root'] / 'comparison_table.csv').is_file()
    result['chat_template_digest'] = 'different'
    write_json(result_path(run), result)
    with pytest.raises(ValueError, match='chat_template_digest'):
        comparison_rows(context['runs'], context['resolved'])
    result['chat_template_digest'] = 'same'
    result['metrics']['math_500']['responses'] = 499
    write_json(result_path(run), result)
    with pytest.raises(ValueError, match='Incomplete full evaluation'):
        comparison_rows(context['runs'], context['resolved'])
    boot = next(s for s in sources if 'BUNDLE =' in s)
    assignment = next(n for n in ast.parse(boot).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))
    isolated = tmp_path / 'isolated'
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assignment.value)))) as archive:
        assert set(archive.namelist()) == set(BUNDLE_FILES)
        for name in BUNDLE_FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(isolated)
    subprocess.run([sys.executable, '-m', 'eval.run_amc_math_eval', '--help'],
                   cwd=isolated, check=True, capture_output=True)
    subprocess.run([sys.executable, '-c',
        "from eval.run_amc_math_eval import comparison_settings; "
        "from eval.qwen_sft_rl import checkpoint_source; "
        "from eval.answer_only import evaluation_code_digest; "
        "s,c,e,d=comparison_settings('configs/qwen_sft_rl_eval.yaml'); "
        "assert list(d['datasets'])==['amc23','math_500']; "
        "assert c['temperature']==0 and len(evaluation_code_digest())==64"],
        cwd=isolated, check=True, capture_output=True)


def test_both_models_full_evaluation_and_resume(tmp_path, monkeypatch):
    from test_batched_eval import Tokenizer, install_core
    from eval import engines, english_engines, run_amc_math_eval as runner

    spec, config, _, _ = comparison_settings(CONFIG)
    roots = {key: tmp_path / key for key in LABELS}
    for key, root in roots.items():
        make_checkpoint(root, key)
    runs = prepare_runs(spec, roots, tmp_path / 'comparison')
    cores = []

    def loader(config, suite, token):
        return {name: [{'id': f'{name}-{i}', entry['problem_column']: f'Question {i}',
                        entry['answer_column']: '0'} for i in range(suite['datasets'][name]['rows'])]
                for name, entry in config['datasets'].items()}

    def create(model, config, revision):
        engine = SimpleNamespace(name='vllm', config=config, tokenizer=Tokenizer())
        cores.append(install_core(monkeypatch, engine))
        return engine, None

    monkeypatch.setattr(runner, 'load_eval_sets', loader)
    monkeypatch.setattr(runner, 'resolve_model', lambda model, revision, token: (revision, digest(model)))
    monkeypatch.setattr(runner, 'package_versions', lambda: {'runtime': 'fixed'})
    monkeypatch.setattr(engines, 'check_hardware', lambda config: {'gpu': 'test'})
    monkeypatch.setattr(english_engines, 'create_engine', create)
    for resumed in [False, True]:
        for run in runs:
            monkeypatch.setattr(sys, 'argv', ['eval', '--model', run['model'], '--run-name', run['run_name'],
                '--config', str(CONFIG), '--output-root', run['output_root'], '--bundled-code'])
            runner.main()
            assert bool(cores[-1].calls) != resumed
    rows = comparison_rows(runs, config)
    assert [row['Problems'] for row in rows] == [40, 500, 40, 500]
    assert all(row['Accuracy (%)'] == 100 for row in rows)
