import ast
import base64
import io
import json
import math
import shutil
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import nbformat
import pytest
import yaml

from common.io import digest, read_jsonl, write_json
from train.qwen_self_rft_data import BUNDLE_FILES, configure_tokenizer, format_example, prepare_examples
from train.qwen_sft_data import prepare_hf_run, latest_checkpoint
from train.sft_data import AssistantCollator, load_rows
from train.stage1_qwen_self_rft import load_config
from train.stage1_sft import build_arguments
from test_qwen_sft import tokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/sft_qwen25_3b_self_rft.yaml'
NOTEBOOK = ROOT / 'notebooks/train_qwen25_3b_base_self_rft.ipynb'


def config():
    return yaml.safe_load(CONFIG.read_text())


def prepare_fixture(tmp_path):
    rows = [{'problem': 'Q', 'response': answer, 'gold_answer': 'EXCLUDE'}
            for answer in ['x y', 'y x', 'x']]
    cfg = config()
    cfg['data']['hf_source'].update(expected_rows=3, selected_content_digest=digest(
        [{k: r[k] for k in ['problem', 'response']} for r in rows]))
    template = tmp_path / 'template.yaml'
    template.write_text(yaml.safe_dump(cfg))
    path = prepare_hf_run(template, tmp_path / 'output', 'test', source_rows=rows)
    return path, template, rows


def test_snapshot_native_eos_and_loss_mask(tmp_path, monkeypatch):
    path, template, rows = prepare_fixture(tmp_path)
    cfg = load_config(path)
    assert cfg['training']['num_train_epochs'] == 5
    assert cfg['data']['hf_source']['revision'] == '96c5f70e37098a9f41f5e44adc4e906ee3f5ef15'
    pinned, ids = load_rows(cfg)
    assert len(pinned) == 3 and all(set(r) == {'problem', 'response'} for r in pinned)
    tok = tokenizer()
    before = tok.eos_token_id, tok.pad_token_id, tok.chat_template, tok.get_vocab()
    configure_tokenizer(tok, cfg)
    assert before == (tok.eos_token_id, tok.pad_token_id, tok.chat_template, tok.get_vocab())
    examples, report, preview = prepare_examples(tok, pinned, ids, cfg)
    assert report['kept_rows'] == 3 and report['unique_problems'] == 1
    assert not report['deduplication_applied'] and not report['truncation_applied']
    for e, row in zip(examples, pinned):
        assert [v for v in e['labels'] if v != -100] == tok.encode(
            row['response'] + '<|endoftext|>', add_special_tokens=False)
        assert e['input_ids'][-1] == tok.eos_token_id
        assert e['labels'][-1] == tok.pad_token_id  # Real EOS is not padding.
    assert preview[1].endswith('x y<|endoftext|>')
    assert 'EXCLUDE' not in preview[1]
    import datasets
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **kw: pytest.fail('Must reuse snapshot'))
    assert prepare_hf_run(template, tmp_path / 'output', 'test') == path
    changed = deepcopy(rows)
    changed[0]['response'] += ' y'
    with pytest.raises(ValueError, match='pinned Hugging Face'):
        prepare_hf_run(template, tmp_path / 'output', 'test', source_rows=changed)
    cfg['max_seq_length'] = len(examples[-1]['input_ids'])
    assert prepare_examples(tok, pinned, ids, cfg)[1]['dropped_overlength'] == 2
    with pytest.raises(ValueError, match='embedded'):
        format_example(tok, {'problem': 'Q', 'response': 'x<|im_end|>'})
    tok.eos_token = '<|im_end|>'
    with pytest.raises(ValueError, match='original'):
        configure_tokenizer(tok, cfg)


def test_hf_source_and_five_epoch_budget(tmp_path, monkeypatch):
    path, template, rows = prepare_fixture(tmp_path)
    import datasets
    calls = []
    def loader(repo, **kwargs):
        calls.append((repo, kwargs))
        return datasets.Dataset.from_list(rows)
    monkeypatch.setattr(datasets, 'load_dataset', loader)
    prepare_hf_run(template, tmp_path / 'fresh', 'test', token='test-token')
    cfg = config()
    source = cfg['data']['hf_source']
    assert calls == [(source['repo'], {'name': 'default', 'split': 'train',
        'revision': source['revision'], 'token': 'test-token'})]
    assert math.ceil(source['expected_rows'] / 16) * cfg['training']['num_train_epochs'] == 8835
    assert load_config(path, smoke=True)['training']['num_train_epochs'] == 1


def test_notebook_commands_and_isolated_bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from build_qwen_self_rft_sft_notebook import cells
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells] == [c.source for c in cells()]
    sources = [c.source for c in nb.cells if c.cell_type == 'code']
    for source in sources:
        compile(source, 'self_rft_notebook', 'exec')
        assert '@param' not in source and '"git", "clone"' not in source
    context = {'Path': Path, 'json': json}
    exec(sources[0], context)
    assert context['TRAIN_ROOT'].endswith('base-rejection-sampling-sft')
    context.update(TRAIN_ROOT=str(tmp_path), CODE_ROOT=ROOT, TRAIN_PYTHON='/test/python',
                   EVAL_PYTHON='/test/eval-python')
    calls = []
    context['run_logged'] = lambda command, **kw: calls.append(command)
    from train import self_rft_display, self_rft_epochs
    monkeypatch.setattr(self_rft_display, 'run_progress', lambda command, **kw: calls.append(command))
    epoch_calls = []
    monkeypatch.setattr(self_rft_epochs, 'run_epochs', lambda path, **kw: epoch_calls.append((path, kw)))
    write_json(tmp_path / 'logs/stage1' / context['RUN_NAME'] / 'preparation/data_report.json', {
        'kept_rows': 28267, 'dropped_missing_outputs': 0, 'dropped_overlength': 0,
        'token_lengths': {}, 'supervised_tokens_including_eot': {},
        'optimizer_steps_per_epoch': 1767, 'total_optimizer_steps': 8835})
    exec(next(s for s in sources if 'CONFIG_PATH =' in s), context)
    assert 'train.qwen_self_rft_data' in calls[0] and 'train.stage1_qwen_self_rft' in calls[1]
    for flag in ['RUN_SMOKE', 'RUN_TRAINING']:
        source = next(s for s in sources if f'{flag} = False' in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(f'{flag} = False', f'{flag} = True'), context)
    assert len(calls) == 3 and '--smoke' in calls[2] and '--auto-resume' in calls[2]
    assert len(epoch_calls) == 1 and epoch_calls[0][1]['eval_python'] == '/test/eval-python'
    boot = next(s for s in sources if 'BUNDLE =' in s)
    assign = next(n for n in ast.parse(boot).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))
    bundle_dir = tmp_path / 'isolated'
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assign.value)))) as archive:
        assert set(archive.namelist()) == set(BUNDLE_FILES)
        for name in BUNDLE_FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(bundle_dir)
    path, _, _ = prepare_fixture(tmp_path)
    tok_dir = tmp_path / 'tokenizer'
    tokenizer().save_pretrained(tok_dir)
    subprocess.run([sys.executable, '-m', 'train.stage1_qwen_self_rft', '--config', str(path),
                    '--prepare-only', '--local-tokenizer', str(tok_dir)],
                   cwd=bundle_dir, check=True, capture_output=True)
    subprocess.run([sys.executable, '-c', 'from train.qwen_self_rft_data import training_code_digest; '
                    'assert len(training_code_digest()) == 64'], cwd=bundle_dir, check=True, capture_output=True)
    subprocess.run([sys.executable, '-m', 'eval.run_amc_math_eval', '--help'],
                   cwd=bundle_dir, check=True, capture_output=True)
    report = json.loads((tmp_path / 'output/logs/stage1/test/preparation/data_report.json').read_text())
    assert report['kept_rows'] == 3 and report['total_optimizer_steps'] == 5


def test_real_qwen_training_partial_batch_epoch_save_reload_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    from datasets import Dataset
    from transformers import Qwen2Config, Qwen2ForCausalLM, AutoModelForCausalLM, AutoTokenizer
    from train.qwen_sft_trainer import QwenSFTTrainer

    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = config()
    cfg["training"].update(bf16=False, tf32=None, use_cpu=True, num_train_epochs=2,
        gradient_accumulation_steps=2, dataloader_pin_memory=False, disable_tqdm=True)
    tok = configure_tokenizer(tokenizer(), cfg)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, attention_dropout=0,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True))
    original = model.model.embed_tokens.weight.detach().clone()
    initial_model = deepcopy(model)
    examples = [format_example(tok, {"problem": "Q", "response": answer})[0] for answer in ["x y", "y x", "x"]]
    batch = AssistantCollator(tok.pad_token_id)(examples)
    assert batch['labels'][0, -1] == tok.eos_token_id
    assert batch['labels'][-1, -1] == -100
    assert tok.eos_token_id == tok.pad_token_id
    data = Dataset.from_list(examples)
    checkpoints, logs = tmp_path / "checkpoints", tmp_path / "logs"
    def make_trainer(model, checkpoints, logs):
        return QwenSFTTrainer(model=model, args=build_arguments(cfg, checkpoints, logs),
            processing_class=tok, train_dataset=data, data_collator=AssistantCollator(tok.pad_token_id),
            log_dir=logs, run_identity="test", lm_head_chunk_tokens=2)
    trainer = make_trainer(model, checkpoints, logs)
    trainer.train()
    assert trainer.state.global_step == 4 and trainer.state.epoch == 2
    assert not torch.equal(original, model.model.embed_tokens.weight)
    assert latest_checkpoint(checkpoints, 2) == checkpoints / "epoch_2"
    loaded = AutoModelForCausalLM.from_pretrained(checkpoints / "epoch_2")
    assert loaded.config.eos_token_id == tok.eos_token_id
    assert loaded.config.pad_token_id == tok.eos_token_id
    saved_tok = AutoTokenizer.from_pretrained(checkpoints / "epoch_2")
    assert saved_tok.eos_token == saved_tok.pad_token == "<|endoftext|>"
    stops = loaded.generation_config.eos_token_id
    assert (stops if isinstance(stops, list) else [stops]) == [tok.eos_token_id]
    assert AutoTokenizer.from_pretrained(checkpoints / "epoch_2").chat_template == tok.chat_template
    metrics = read_jsonl(logs / "steps.jsonl")
    assert len(metrics) == 4 and all("loss" in m and "grad_norm" in m for m in metrics)
    resumed_dir = tmp_path / "resumed"
    from train.self_rft_callbacks import install_progress
    interrupted = make_trainer(initial_model, resumed_dir, tmp_path / 'resume_logs')
    install_progress(interrupted, stop_after_epoch=1)
    interrupted.train()
    assert interrupted.state.global_step == 2 and interrupted.state.epoch == 1
    assert (resumed_dir / 'epoch_1/stage1_checkpoint.json').is_file()
    assert not (resumed_dir / 'epoch_2').exists()
    resumed = make_trainer(AutoModelForCausalLM.from_pretrained(resumed_dir / "epoch_1"),
                           resumed_dir, tmp_path / "resume_logs")
    install_progress(resumed, stop_after_epoch=2)
    resumed.train(resume_from_checkpoint=str(resumed_dir / "epoch_1"))
    assert resumed.state.global_step == 4
    for a, b in zip(resumed.model.parameters(), trainer.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    (resumed_dir / "epoch_2/optimizer.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete checkpoint"):
        latest_checkpoint(resumed_dir, 2)
