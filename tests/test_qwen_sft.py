import ast
import base64
import io
import json
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
from train.qwen_sft_data import BUNDLE_FILES, configure_tokenizer, latest_checkpoint, prepare_hf_run
from train.sft_data import AssistantCollator, format_example, load_rows, prepare_examples
from train.stage1_qwen_sft import load_config
from train.stage1_sft import build_arguments

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sft_qwen25_3b_s1_kimi.yaml"
NOTEBOOK = ROOT / "notebooks/train_qwen25_3b_base_s1_kimi.ipynb"
ANSWER = "kimi-style-reasoning-answer"
TARGET = "## Planning\nPlan\n## Solution and Evaluation\nSolve\n## Reflection\nRefine\n## Exploration\nCompare\nFinal answer: \\boxed{4}"


def config():
    return yaml.safe_load(CONFIG.read_text())


def source_rows():
    return [{"question": "Q", ANSWER: TARGET, "deepseek-v4-pro_reasoning": "EXCLUDE_RAW",
             "deepseek_attempt": "EXCLUDE_ORIGINAL", "deepseek_grade": "No"},
            {"question": "Q2", ANSWER: None}, {"question": "Q3", ANSWER: " "}]


def prepare_fixture(tmp_path):
    rows, cfg = source_rows(), config()
    projected = [{"question": r["question"], ANSWER: r[ANSWER]} for r in rows]
    cfg["data"]["hf_source"].update(expected_rows=len(rows), selected_content_digest=digest(projected))
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(cfg))
    path = prepare_hf_run(template, tmp_path / "output", "test", source_rows=rows)
    return path, template, rows


def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {token: i for i, token in enumerate([
        "[UNK]", "<|endoftext|>", "<|im_end|>", "<|im_start|>", "user", "assistant",
        "system", "Q", "x", "y", "Plan", "Solve", "Refine", "Compare", "Final", "answer:", "\\boxed{4}",
    ])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="<|endoftext|>", eos_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
        chat_template="{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
        "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
    )


def test_snapshot_pins_only_requested_columns_and_reuses_without_network(tmp_path, monkeypatch):
    path, template, rows = prepare_fixture(tmp_path)
    cfg = load_config(path)
    assert cfg["model"]["repo"] == "Qwen/Qwen2.5-3B"
    assert cfg["training"]["num_train_epochs"] == 5
    assert cfg["training"]["gradient_accumulation_steps"] == 16
    assert cfg["data"]["hf_source"]["revision"] == "636ecf409774771afb0bf10a436f4b1e608b5f29"
    pinned, ids = load_rows(cfg)
    assert all(set(row) == {"question", ANSWER} for row in pinned)
    assert pinned[0][ANSWER] == TARGET and pinned[1][ANSWER] is None
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: pytest.fail("Rerun must reuse snapshot"))
    assert prepare_hf_run(template, tmp_path / "output", "test") == path
    changed = deepcopy(rows)
    changed[0][ANSWER] += "changed"
    with pytest.raises(ValueError, match="pinned Hugging Face"):
        prepare_hf_run(template, tmp_path / "output", "test", source_rows=changed)
    edited = yaml.safe_load(template.read_text())
    edited["training"]["learning_rate"] *= 2
    template.write_text(yaml.safe_dump(edited))
    with pytest.raises(ValueError, match="settings changed"):
        prepare_hf_run(template, tmp_path / "output", "test")


def test_actual_hf_loader_arguments_and_column_selection(tmp_path, monkeypatch):
    path, template, rows = prepare_fixture(tmp_path)
    import datasets
    calls = []
    def loader(repo, **kwargs):
        calls.append((repo, kwargs))
        return datasets.Dataset.from_list(rows)
    monkeypatch.setattr(datasets, "load_dataset", loader)
    result = prepare_hf_run(template, tmp_path / "fresh", "test", token="test-token")
    assert load_rows(load_config(result))[0][0][ANSWER] == TARGET
    assert calls == [("Seungjun/dp_removed_s1K-1.1", {
        "name": "default", "split": "train", "revision": "636ecf409774771afb0bf10a436f4b1e608b5f29",
        "token": "test-token",
    })]


def test_native_chat_loss_region_eot_padding_and_missing_rows(tmp_path):
    path, _, rows = prepare_fixture(tmp_path)
    cfg = load_config(path)
    tok = tokenizer()
    vocab, template = tok.get_vocab(), tok.chat_template
    configure_tokenizer(tok, cfg)
    assert tok.get_vocab() == vocab and tok.chat_template == template
    assert tok.eos_token == "<|im_end|>" and tok.pad_token == "<|endoftext|>"
    examples, report, preview = prepare_examples(tok, rows, ["a", "b", "c"], cfg)
    e, text = preview
    target_ids = tok.encode(TARGET + "<|im_end|>", add_special_tokens=False)
    assert [v for v in e["labels"] if v != -100] == target_ids
    assert TARGET in text and "<think>" not in text and "<answer>" not in text
    assert "EXCLUDE_RAW" not in text and "EXCLUDE_ORIGINAL" not in text
    assert report["kept_ids"] == ["a"] and report["dropped_missing_outputs"] == 2
    assert not report["grade_filter_applied"] and not report["truncation_applied"]
    cfg["max_seq_length"] = len(e["input_ids"])
    longer = {"question": "Q", ANSWER: TARGET + " x"}
    assert prepare_examples(tok, [rows[0], longer], ["keep", "drop"], cfg)[1]["dropped_overlength"] == 1
    smoke = load_config(path, smoke=True)
    assert smoke["run_name"] == "test_smoke" and smoke["training"]["num_train_epochs"] == 1


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_chunked_qwen_loss_and_all_parameter_gradients_match_native_ce(bf16, tied):
    torch = pytest.importorskip("torch")
    from contextlib import nullcontext
    from types import SimpleNamespace
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from train.qwen_sft_trainer import QwenSFTTrainer

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, attention_dropout=0,
        tie_word_embeddings=tied))
    if bf16:
        model = model.to(torch.bfloat16)
    reference = deepcopy(model)
    inputs = {"input_ids": torch.tensor([[3, 4, 5, 6, 2], [6, 5, 2, 1, 1]]),
              "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
              "labels": torch.tensor([[-100, -100, 5, 6, 2], [-100, 5, 2, -100, -100]])}
    def context():
        return torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
    trainer = object.__new__(QwenSFTTrainer)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m, autocast=context)
    trainer.lm_head_chunk_tokens = 2
    with context():
        expected = reference(**inputs).loss
    actual = trainer.compute_loss(model, inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.backward()
    actual.backward()
    for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert a.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, rtol=.03 if bf16 else 2e-4, atol=.002 if bf16 else 2e-6)


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
    examples = [format_example(tok, {"question": "Q", ANSWER: answer}, cfg["data"]["columns"],
                               "answer_only")[0] for answer in ["x y", "y x", "x"]]
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
    stops = loaded.generation_config.eos_token_id
    assert (stops if isinstance(stops, list) else [stops]) == [tok.eos_token_id]
    assert AutoTokenizer.from_pretrained(checkpoints / "epoch_2").chat_template == tok.chat_template
    metrics = read_jsonl(logs / "steps.jsonl")
    assert len(metrics) == 4 and all("loss" in m and "grad_norm" in m for m in metrics)
    resumed_dir = tmp_path / "resumed"
    shutil.copytree(checkpoints / "epoch_1", resumed_dir / "epoch_1")
    resumed = make_trainer(AutoModelForCausalLM.from_pretrained(resumed_dir / "epoch_1"),
                           resumed_dir, tmp_path / "resume_logs")
    resumed.train(resume_from_checkpoint=str(resumed_dir / "epoch_1"))
    assert resumed.state.global_step == 4
    for a, b in zip(resumed.model.parameters(), trainer.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    (resumed_dir / "epoch_2/optimizer.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete checkpoint"):
        latest_checkpoint(resumed_dir, 2)


def test_notebook_commands_flags_and_reports(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_qwen_sft_notebook import cells
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells] == [c.source for c in cells()]
    sources = [c.source for c in nb.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "qwen_sft_notebook", "exec")
        assert "@param" not in source and '"git", "clone"' not in source
    context = {"Path": Path, "json": json}
    exec(sources[0], context)
    assert context["TRAIN_ROOT"].endswith("base-sft-v2-short")
    context.update(TRAIN_ROOT=str(tmp_path), CODE_ROOT=ROOT, TRAIN_PYTHON="/test/python")
    calls = []
    context["run_logged"] = lambda command, **kwargs: calls.append(command)
    report = {"kept_rows": 989, "dropped_missing_outputs": 7, "dropped_overlength": 0,
              "token_lengths": {}, "supervised_tokens_including_eot": {},
              "optimizer_steps_per_epoch": 62, "total_optimizer_steps": 310}
    write_json(tmp_path / "logs/stage1" / context["RUN_NAME"] / "preparation/data_report.json", report)
    exec(next(s for s in sources if "CONFIG_PATH =" in s), context)
    assert "train.qwen_sft_data" in calls[0] and "--prepare-only" in calls[1]
    for flag in ["RUN_SMOKE", "RUN_TRAINING"]:
        source = next(s for s in sources if f"{flag} = False" in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(f"{flag} = False", f"{flag} = True"), context)
    assert len(calls) == 4 and "--smoke" in calls[2] and "--smoke" not in calls[3]
    assert "--auto-resume" in calls[2] and "--auto-resume" in calls[3]
    exec(sources[-1], context)


def test_bundle_is_current_and_prepares_without_repository(tmp_path):
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = next(c.source for c in nb.cells if c.cell_type == "code" and "BUNDLE =" in c.source)
    assign = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "BUNDLE" for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assign.value)))) as archive:
        assert set(archive.namelist()) == set(BUNDLE_FILES)
        for name in BUNDLE_FILES:
            assert archive.read(name) == (ROOT / name).read_bytes()
        archive.extractall(tmp_path)
    path, _, _ = prepare_fixture(tmp_path)
    tok_dir = tmp_path / "tokenizer"
    tokenizer().save_pretrained(tok_dir)
    subprocess.run([sys.executable, "-m", "train.stage1_qwen_sft", "--config", str(path),
                    "--prepare-only", "--local-tokenizer", str(tok_dir)],
                   cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([sys.executable, "-c", "from train.qwen_sft_data import training_code_digest; "
                    "assert len(training_code_digest())==64"], cwd=tmp_path, check=True, capture_output=True)
    report = json.loads((tmp_path / "output/logs/stage1/test/preparation/data_report.json").read_text())
    assert report["kept_rows"] == 1 and report["total_optimizer_steps"] == 5
