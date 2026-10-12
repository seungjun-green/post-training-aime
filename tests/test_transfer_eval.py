import ast
import base64
import copy
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

from common.io import append_jsonl, digest, read_jsonl, write_json
from eval.run_transfer_eval import bundle_files, comparison, load_config, run_model
from eval.transfer_data import prepare_data, select_mmlu_rows, smoke_documents
from eval.transfer_models import model_fingerprint, prepare_models, validate_checkpoint
from eval.transfer_scoring import bbh_answer, choice_answer, ifeval_scores, summarize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/qwen_transfer_eval.yaml'


@pytest.mark.parametrize('text,expected', [
    ('I considered A and B. The answer is (C).', 'C'),
    ('The answer is (B). Wait, no. The answer is (C).', 'C'),
    ('Final answer: **C**', 'C'), ('C', 'C'), ('(C).', 'C'),
    ('Option A seems promising but B might be correct.', None),
    ('The answer is (J).', None), ('The answer is Blue.', None), ('', None),
])
def test_choice_extraction_does_not_guess(text, expected):
    assert choice_answer(text, list('ABCD')) == expected


@pytest.mark.parametrize('target', ['(A)', 'True', 'yes', 'valid', 'apple banana cat', '] } )'])
def test_bbh_final_answer(target):
    assert bbh_answer('Some reasoning. So the answer is ' + target + '.', target) == target
    assert bbh_answer(target, target) == target
    assert bbh_answer('Some unrelated reasoning', '(A)') is None


def test_official_ifeval_and_denominators():
    doc = {'key': 1, 'prompt': 'Say hello in uppercase.',
           'instruction_id_list': ['keywords:existence', 'change_case:english_capital'],
           'kwargs': [{'keywords': ['HELLO']}, {}]}
    result = ifeval_scores(doc, 'HELLO WORLD. I AM WRITING THIS RESPONSE IN ENGLISH AND EVERY LETTER IS UPPERCASE.')
    assert result['prompt_strict'] and result['prompt_loose']
    fail = ifeval_scores(doc, 'hello world.')
    assert fail['instructions_strict'] == [True, False]
    from instruction_following_eval import instructions_util
    assert instructions_util.count_sentences('Hello. Goodbye.') == 2
    rows = summarize([{'benchmark': 'ifeval', 'subject': 'all', 'token_count': 3,
                       'finish_reason': 'stop', 'score': result}])
    totals = {r['Metric']: r['Denominator'] for r in rows if r['Subject/task'] == 'overall'}
    assert totals == {'prompt_strict': 1, 'prompt_loose': 1, 'instruction_strict': 2, 'instruction_loose': 2}


def tiny_sources():
    config = load_config(CONFIG)
    config['mmlu_subjects'] = ['physics']
    config['bbh_tasks'] = ['boolean_expressions']
    config['sources'] = {k: v for k, v in config['sources'].items()
                         if not k.startswith('bbh/') or k == 'bbh/boolean_expressions'}
    for k, v in config['sources'].items():
        v['rows'] = 5 if k == 'mmlu_validation' else 1
    question = {'question_id': 0, 'category': 'physics', 'question': 'Test question',
                'options': ['First', 'Second'], 'answer': 'B', 'answer_index': 1,
                'cot_content': 'SECRET TEST REASONING'}
    demo = {**question, 'question': 'Demo question', 'options': ['First', 'Second', 'N/A'],
            'cot_content': "A: Let's think step by step. The answer is (B)."}
    source_rows = {
        'arc': [{'id': 'arc1', 'question': 'ARC question', 'choices': {'label': ['1', '2'],
                  'text': ['Wrong', 'Correct']}, 'answerKey': '2'}],
        'mmlu_test': [question], 'mmlu_validation': [demo] * 5,
        'bbh/boolean_expressions': [{'input': 'True and False', 'target': 'False'}],
        'ifeval': [{'key': 0, 'prompt': 'Say HELLO.', 'instruction_id_list': ['keywords:existence'],
                    'kwargs': [{'keywords': ['HELLO']}]}],
    }
    def loader(source):
        key = next(k for k, v in config['sources'].items() if v is source)
        return copy.deepcopy(source_rows[key])
    return config, loader


def test_mmlu_subset_balanced_reproducible_and_auditable(tmp_path):
    full_config = load_config(CONFIG)
    subset_config = load_config(ROOT / 'configs/qwen_transfer_eval_mmlu100.yaml')
    assert {k: v for k, v in subset_config.items() if k not in {'version', 'mmlu_selection'}} == {
        k: v for k, v in full_config.items() if k != 'version'}
    config, original_loader = tiny_sources()
    config['mmlu_subjects'] = subset_config['mmlu_subjects']
    config['mmlu_selection'] = subset_config['mmlu_selection']
    test_row = original_loader(config['sources']['mmlu_test'])[0]
    demo = original_loader(config['sources']['mmlu_validation'])[0]
    rows = [{**test_row, 'question_id': f'{subject}-{i}', 'category': subject}
            for subject in config['mmlu_subjects'] for i in range(120)]
    config['sources']['mmlu_test']['rows'] = len(rows)
    config['sources']['mmlu_validation']['rows'] = 70

    def loader(source):
        if source is config['sources']['mmlu_test']:
            return rows
        if source is config['sources']['mmlu_validation']:
            return [{**demo, 'category': subject} for subject in config['mmlu_subjects'] for _ in range(5)]
        return original_loader(source)

    docs, manifest = prepare_data(config, tmp_path, loader)
    assert manifest['counts'] == {'arc_challenge': 1, 'mmlu_pro': 1400, 'bbh_nonmath': 1, 'ifeval': 1}
    selected = manifest['mmlu_selection']['selected_ids_by_subject']
    assert set(selected) == set(config['mmlu_subjects'])
    assert all(len(set(ids)) == 100 for ids in selected.values())
    assert len({doc['id'] for doc in docs}) == len(docs)
    assert not any('SECRET TEST REASONING' in d['prompt'] for d in docs)
    assert all(d['prompt'].count('Demo question') == 5 for d in docs if d['benchmark'] == 'mmlu_pro')
    assert prepare_data(config, tmp_path, loader) == (docs, manifest)
    assert select_mmlu_rows(rows, config) == select_mmlu_rows(list(reversed(rows)), config)
    different_seed = copy.deepcopy(config)
    different_seed['mmlu_selection']['seed'] += 1
    assert select_mmlu_rows(rows, config) != select_mmlu_rows(rows, different_seed)
    with pytest.raises(ValueError, match='new OUTPUT_ROOT'):
        prepare_data(different_seed, tmp_path, loader)
    full = dict(config)
    del full['mmlu_selection']
    full_docs, full_manifest = prepare_data(full, tmp_path / 'full', loader)
    assert full_manifest['counts']['mmlu_pro'] == len(rows)
    assert [d for d in full_docs if d['benchmark'] != 'mmlu_pro'] == [
        d for d in docs if d['benchmark'] != 'mmlu_pro']
    with pytest.raises(ValueError, match='new OUTPUT_ROOT'):
        prepare_data(config, tmp_path / 'full', loader)
    with pytest.raises(ValueError, match='Not enough'):
        select_mmlu_rows(rows[:1], config)
    with pytest.raises(ValueError, match='Duplicate'):
        select_mmlu_rows(rows + rows[:1], config)


def test_data_prompts_pinning_resume_and_smoke(tmp_path):
    config, loader = tiny_sources()
    docs, manifest = prepare_data(config, tmp_path, loader)
    assert manifest['counts'] == {'arc_challenge': 1, 'mmlu_pro': 1, 'bbh_nonmath': 1, 'ifeval': 1}
    assert docs[0]['answer'] == 'B'  # Numeric ARC labels are mapped by position.
    mmlu = docs[1]['prompt']
    assert mmlu.count('Demo question') == 5 and 'SECRET TEST REASONING' not in mmlu
    assert 'N/A' not in mmlu
    assert docs[2]['prompt'].count('Q:') == 4
    assert docs[3]['prompt'] == 'Say HELLO.'
    assert len(smoke_documents(docs)) == 4
    assert prepare_data(config, tmp_path, lambda _: pytest.fail('Must reuse pinned prepared data'))[0] == docs
    changed = copy.deepcopy(config)
    changed['generation']['max_new_tokens'] += 1
    with pytest.raises(ValueError, match='changed'):
        prepare_data(changed, tmp_path, loader)
    append_jsonl(tmp_path / 'data/documents.jsonl', docs[0])
    with pytest.raises(ValueError, match='content changed'):
        prepare_data(config, tmp_path, loader)


def fixture_checkpoint(root, key, config):
    entry = config['models'][key]
    path = root / 'experiment' / entry['path'].lstrip('/')
    if key == 'instruct_rl':
        path = root / 'instruct-rl/checkpoints/dapo_qwen25_3b_base/checkpoint-300'
    is_sft = entry['kind'] == 'sft'
    run_name = path.parent.name
    expected_model = config['models']['instruct' if key == 'instruct_rl' else 'base']
    cfg = {'model': {k: expected_model[k] for k in ['repo', 'revision']}}
    if is_sft:
        cfg['run_name'] = run_name
    else:
        cfg.update(model_kind=run_name.rsplit('_', 1)[1], run_name_prefix=run_name.rsplit('_', 1)[0],
                   training={'max_steps': 300})
    manifest = {'config': cfg}
    if 'parent_epoch' in entry:
        parent = config['models'][key.replace('_rl', '_sft')]
        manifest['source'] = {'kind': 'sft', 'path': parent['path'], 'parent_run_identity': 'parent',
                              'marker': {'epoch': parent['epoch'], 'global_step': parent['step'], 'run_identity': 'parent'}}
    manifest['identity'] = digest(manifest)
    config['models'][key].pop('expected_identity', None)
    marker = {'run_identity': manifest['identity'], 'global_step': entry['step']}
    if is_sft:
        marker['epoch'] = entry['epoch']
    write_json(path / ('stage1_checkpoint.json' if is_sft else 'dapo_checkpoint.json'), marker)
    train_root = path.parents[3 if is_sft else 2]
    write_json(train_root / ('logs/stage1' if is_sft else 'logs') / run_name / 'run_manifest.json', manifest)
    for name in ['config.json', 'generation_config.json']:
        write_json(path / name, {'model_type': 'qwen2', 'eos_token_id': entry['eos_token_id']})
    write_json(path / 'tokenizer_config.json', {'eos_token': '<|im_end|>' if entry['eos_token_id'] == 151645 else '<|endoftext|>'})
    write_json(path / 'tokenizer.json', {'vocab': 'native'})
    (path / 'chat_template.jinja').write_text('native template')
    header = json.dumps({'w': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    (path / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
    return path


def test_all_nine_paths_and_native_eos(tmp_path):
    config = load_config(CONFIG)
    for key, entry in config['models'].items():
        if 'path' in entry:
            fixture_checkpoint(tmp_path, key, config)
    runs = prepare_models(config, tmp_path / 'experiment', tmp_path / 'instruct-rl', tmp_path / 'eval')
    assert len(runs) == 9
    paths = {r['key']: r['model'] for r in runs}
    assert paths['rejection_sft'].endswith('/epoch_4')
    assert paths['distill_sft'].endswith('/epoch_5')
    assert paths['instruct_rl'].endswith('/dapo_qwen25_3b_base/checkpoint-300')
    assert config['models']['kimi_sft']['eos_token_id'] == 151645
    assert config['models']['base_rl']['eos_token_id'] == 151643
    selected = runs[1]
    before = model_fingerprint(selected)
    (Path(selected['model']) / 'chat_template.jinja').write_text('changed template')
    assert model_fingerprint(selected) != before


@pytest.mark.parametrize('change', ['epoch', 'identity', 'weights', 'eos', 'parent'])
def test_bad_checkpoints_rejected(tmp_path, change):
    config = load_config(CONFIG)
    key = 'rejection_rl' if change == 'parent' else 'rejection_sft'
    path = fixture_checkpoint(tmp_path, key, config)
    if change == 'epoch':
        marker = json.loads((path / 'stage1_checkpoint.json').read_text())
        marker['epoch'] = 5
        write_json(path / 'stage1_checkpoint.json', marker)
    elif change == 'identity':
        marker = json.loads((path / 'stage1_checkpoint.json').read_text())
        marker['run_identity'] = 'bad'
        write_json(path / 'stage1_checkpoint.json', marker)
    elif change == 'weights':
        f = path / 'model.safetensors'
        f.write_bytes(f.read_bytes()[:-1])
    elif change == 'eos':
        write_json(path / 'generation_config.json', {'eos_token_id': 151645})
    else:
        config['models'][key]['parent_epoch'] = 5
    with pytest.raises(ValueError):
        validate_checkpoint(path, config['models'][key], key, config)


class Tokenizer:
    chat_template = 'native'
    eos_token_id, pad_token_id = 151643, 151643
    special_tokens_map = {'eos_token': '<|endoftext|>'}
    def get_vocab(self):
        return {'test': 1}
    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        return '<native>' + messages[0]['content']


def test_interrupted_generation_resume_scoring_and_protocol_guard(tmp_path):
    config, loader = tiny_sources()
    docs, data_manifest = prepare_data(config, tmp_path, loader)
    run = prepare_models(config, tmp_path, tmp_path, tmp_path, ['base'])[0]
    calls = []
    def factory(*args):
        calls.append(args)
        return SimpleNamespace(tokenizer=Tokenizer())
    def generate(engine, jobs, execution, prompt_renderer):
        for job in jobs:
            doc = next(d for d in docs if d['id'] == job['key'])
            text = 'HELLO' if doc['benchmark'] == 'ifeval' else 'The answer is ' + doc['answer'] + '.'
            yield job, prompt_renderer(engine.tokenizer, doc['prompt']), [
                {'text': text, 'token_count': 10, 'finish_reason': 'stop'}]
    def interrupted(*args, **kw):
        yield from generate(*args, **kw)
        raise RuntimeError('interrupted after saved response')
    kwargs = {'engine_factory': factory, 'metadata_factory': lambda: {'packages': {'test': '1'}, 'hardware': {}},
              'fingerprint': lambda _: 'weights'}
    with pytest.raises(RuntimeError, match='interrupted'):
        run_model(run, config, docs, data_manifest, tmp_path, generator=interrupted, **kwargs)
    journal = tmp_path / 'models/base/full/responses.jsonl'
    assert len(read_jsonl(journal)) == 1
    result = run_model(run, config, docs, data_manifest, tmp_path, generator=generate, **kwargs)
    assert result['completed'] == 4 and len(read_jsonl(journal)) == 4
    count = len(calls)
    again = run_model(run, config, docs, data_manifest, tmp_path, generator=generate, **kwargs)
    assert again == result and len(calls) == count  # Fully resumed: no GPU load.
    rows, details = comparison(config, tmp_path)
    assert len(rows) == 1
    assert all(r['Accuracy (%)'] == 100 for r in details)
    changed = copy.deepcopy(config)
    changed['generation']['max_new_tokens'] += 1
    with pytest.raises(ValueError, match='changed'):
        run_model(run, changed, docs, data_manifest, tmp_path, generator=generate, **kwargs)


def test_subject_macro_and_nonmath_aggregate():
    records = []
    for subject, answers in [('math', [False] * 3), ('physics', [True]), ('history', [False])]:
        for correct in answers:
            records.append({'benchmark': 'mmlu_pro', 'subject': subject, 'token_count': 10,
                            'finish_reason': 'length', 'score': {'correct': correct, 'unparsed': False}})
    rows = {r['Subject/task']: r for r in summarize(records)}
    assert rows['overall']['Accuracy (%)'] == 20
    assert rows['non_math_overall']['Accuracy (%)'] == 50
    assert rows['task_macro']['Accuracy (%)'] == pytest.approx(100 / 3)
    assert rows['physics']['Truncated (%)'] == 100


def test_notebook_commands_and_isolated_bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from build_qwen_transfer_eval_notebook import DISPLAY_FILES, cells
    notebook = nbformat.read(ROOT / 'notebooks/evaluate_qwen25_3b_all9_transfer.ipynb', as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells()]
    sources = [c.source for c in notebook.cells if c.cell_type == 'code']
    for source in sources:
        compile(source, 'transfer_notebook', 'exec')
        assert '@param' not in source
    context = {'Path': Path}
    exec(sources[0], context)
    assert context['MODEL_KEYS'] is None
    config = load_config(CONFIG)
    runs = [{'key': k, 'label': e['label']} for k, e in config['models'].items()]
    calls = []
    displays = []
    context.update(runs=runs, CODE_ROOT=ROOT, totals={'arc_challenge': 1172, 'mmlu_pro': 12032,
                   'bbh_nonmath': 4615, 'ifeval': 541}, command=lambda *a: list(a),
                   run_logged=lambda command, **kwargs: calls.append((command, kwargs)),
                   run_eval_logged=lambda command, **kwargs: calls.append((command, kwargs)),
                   show_results=lambda: displays.append(True))
    for flag in ['RUN_SMOKE', 'RUN_EVAL']:
        source = next(s for s in sources if flag + ' = False' in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(flag + ' = False', flag + ' = True'), context)
    assert len(calls) == 27  # Nine smoke, nine full, nine summary calls.
    assert len(displays) == 1  # Tables are displayed once, not repeated after every model.
    for i in range(9):
        assert calls[i][0] == ['--model-key', runs[i]['key'], '--smoke']
        assert calls[9+2*i][0] == ['--model-key', runs[i]['key']]
        assert sum(calls[9+2*i][1]['progress_totals'].values()) == 18360
        assert calls[9+2*i][1]['label'] == f'Model {i+1}/9 · {runs[i]["label"]}'
    boot = next(s for s in sources if 'BUNDLE =' in s)
    node = next(n for n in ast.parse(boot).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(node.value)))) as archive:
        assert set(archive.namelist()) == set(bundle_files() + DISPLAY_FILES)
        for name in bundle_files() + DISPLAY_FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(tmp_path / 'isolated')
    # Presentation changes must preserve the identity of already-running evaluations.
    from eval.run_transfer_eval import code_digest
    assert not set(DISPLAY_FILES) & set(bundle_files())
    isolated_digest = subprocess.check_output([sys.executable, '-c',
        'from eval.run_transfer_eval import code_digest; print(code_digest())'],
        cwd=tmp_path / 'isolated', text=True).strip()
    assert isolated_digest == code_digest()
    subprocess.run([sys.executable, '-m', 'eval.run_transfer_eval', '--help'], cwd=tmp_path / 'isolated',
                   check=True, capture_output=True)
    subprocess.run([sys.executable, '-c',
        "from eval.transfer_scoring import ifeval_scores; "
        "d={'key':0,'prompt':'Hello','instruction_id_list':['keywords:existence'],'kwargs':[{'keywords':['Hello']}]}; "
        "assert ifeval_scores(d,'Hello')['prompt_strict']; "
        "from instruction_following_eval import instructions_util; "
        "assert instructions_util.count_sentences('Hello. Goodbye.')==2"],
        cwd=tmp_path / 'isolated', check=True, capture_output=True)
