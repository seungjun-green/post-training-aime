import json
from copy import deepcopy
from pathlib import Path

import pytest

from common.english_prompts import INSTRUCTION
from common.io import digest
from train.sft_data import (
    AssistantCollator,
    format_example,
    load_rows,
    prepare_examples,
    sanity_text,
)
from train.stage1_sft import build_arguments, load_config, validate_resume

ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    """Reversible toy native template; EOT is one token, trailing newline is not."""

    chat_template = "native"
    eos_token = "~"
    eos_token_id = ord("~")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "[system]~\n"
        for message in messages:
            text += "[" + message["role"] + "]" + message["content"]
            text += "\n" if message["role"] == "user" else "~\n"
        if add_generation_prompt:
            text += "[assistant]"
        return text

    def encode(self, text, **kwargs):
        assert not kwargs.get("truncation", False)
        assert not kwargs["add_special_tokens"]
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def row(grade="No", reasoning="Reasoning"):
    return {
        "question": "What is 2+2?", "deepseek_thinking_trajectory": reasoning,
        "deepseek_attempt": r"The answer is \boxed{4}.", "deepseek_grade": grade,
        "solution": "MUST NOT TRAIN ON THIS", "gemini_attempt": "ALSO NOT USED",
    }


def config():
    return load_config(ROOT / "configs/stage1_sft.yaml")


def test_shared_prompt_exact_loss_region_and_eot():
    tokenizer = CharacterTokenizer()
    example, text = format_example(tokenizer, row())
    assert INSTRUCTION in text
    assert "MUST NOT" not in text and "ALSO NOT" not in text
    supervised = tokenizer.decode([v for v in example["labels"] if v != -100])
    assert supervised == "<think>\nReasoning\n</think>\n\nThe answer is \\boxed{4}.~"
    assert example["labels"][-1] == -100  # trailing template newline
    assert example["labels"][-2] == tokenizer.eos_token_id
    assert example["labels"][text.index("<think>") - 1] == -100
    assert "[MASK]" in sanity_text(tokenizer, (example, text), "[MASK]")


def test_drop_strictly_over_limit_keep_incorrect_and_log_ids():
    tokenizer = CharacterTokenizer()
    first = row()
    limit = len(format_example(tokenizer, first)[0]["input_ids"])
    cfg = config()
    cfg["max_seq_length"] = limit
    examples, report, _ = prepare_examples(
        tokenizer, [first, row("Yes", "Reasoning!")], ["keep", "drop"], cfg,
    )
    assert len(examples) == 1 and report["kept_ids"] == ["keep"]
    assert report["dropped"] == [{"id": "drop", "tokens": limit + 1}]
    assert report["incorrect_grade_rows"] == 1
    assert report["token_lengths"]["p99"] == limit
    assert len(examples[0]["input_ids"]) == limit


def test_local_data_requires_exact_original_content_and_ids(tmp_path):
    cfg = config()
    cfg["data"].update(expected_rows=1, content_digest=digest([row()]), ids_digest=digest(["id"]))
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"id": "id", "original": row()}))
    assert load_rows(cfg, local_rows=path) == ([row()], ["id"])
    path.write_text(json.dumps({"id": "id", "original": row("Yes")}))
    with pytest.raises(ValueError, match="Training rows differ"):
        load_rows(cfg, local_rows=path)


def test_collator_preserves_eot_when_pad_equals_eos():
    pytest.importorskip("torch")
    collator = AssistantCollator(9)
    batch = collator([
        {"input_ids": [1, 2, 9], "attention_mask": [1, 1, 1], "labels": [-100, 2, 9]},
        {"input_ids": [1, 9], "attention_mask": [1, 1], "labels": [-100, 9]},
    ])
    assert batch["labels"].tolist() == [[-100, 2, 9], [-100, 9, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_resume_refuses_other_runs_and_archives_uncheckpointed_steps(tmp_path):
    checkpoints, logs = tmp_path / "checkpoints", tmp_path / "logs"
    checkpoints.mkdir()
    logs.mkdir()
    validate_resume(checkpoints, logs, None, "identity")
    epoch = checkpoints / "epoch_1"
    epoch.mkdir()
    (epoch / "stage1_checkpoint.json").write_text(json.dumps({"run_identity": "identity", "global_step": 2}))
    (logs / "run_manifest.json").write_text(json.dumps({"identity": "identity"}))
    (logs / "steps.jsonl").write_text('{"step": 2}\n{"step": 3}\n')
    with pytest.raises(ValueError, match="Existing run"):
        validate_resume(checkpoints, logs, None, "identity")
    with pytest.raises(ValueError, match="changed"):
        validate_resume(checkpoints, logs, epoch, "wrong")
    validate_resume(checkpoints, logs, epoch, "identity")
    assert (logs / "steps.jsonl").read_text() == '{"step": 2}\n'
    assert len(list(logs.glob("abandoned_steps_*.jsonl"))) == 1


def test_real_sft_trainer_partial_batch_epoch_saves_and_reload(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )

    from train.sft_trainer import Stage1Trainer

    torch.set_num_threads(1)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "x": 3, "y": 4}))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        chat_template="unchanged-template",
    )
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=5, n_layer=1, n_head=1, n_embd=8, n_positions=32,
        bos_token_id=2, eos_token_id=2, pad_token_id=1,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    ))
    before = model.transformer.wte.weight.detach().clone()
    cfg = deepcopy(config())
    cfg["training"].update(
        bf16=False, tf32=None, use_cpu=True, num_train_epochs=2,
        gradient_accumulation_steps=2, dataloader_pin_memory=False, disable_tqdm=True,
    )
    logs = tmp_path / "logs"
    checkpoints = tmp_path / "checkpoints"
    data = Dataset.from_list([
        {"input_ids": [3, 4, 2], "attention_mask": [1, 1, 1], "labels": [-100, 4, 2]}
        for _ in range(3)
    ])
    trainer = Stage1Trainer(
        model=model, args=build_arguments(cfg, checkpoints, logs), train_dataset=data,
        processing_class=tokenizer, data_collator=AssistantCollator(tokenizer.pad_token_id),
        log_dir=logs, run_identity="test",
    )
    trainer.train()
    # Three rows / accumulation 2 => two steps per epoch, including the remainder.
    assert trainer.state.global_step == 4 and trainer.state.epoch == 2
    assert trainer.lr_scheduler.get_last_lr() == [0.0, 0.0]
    assert not torch.equal(before, model.transformer.wte.weight)
    assert set(p.name for p in checkpoints.glob("epoch_*")) == {"epoch_1", "epoch_2"}
    for epoch in [1, 2]:
        folder = checkpoints / f"epoch_{epoch}"
        loaded = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True)
        assert loaded.config.vocab_size == 5
        assert AutoTokenizer.from_pretrained(folder).chat_template == tokenizer.chat_template
        assert (folder / "optimizer.pt").is_file() and (folder / "rng_state.pth").is_file()
    metrics = [json.loads(line) for line in (logs / "steps.jsonl").read_text().splitlines()]
    assert [m["step_tokens"] for m in metrics] == [6, 3, 6, 3]
    assert all(set(["loss", "learning_rate", "grad_norm", "tokens_per_second"]) <= m.keys() for m in metrics)
    assert metrics[-1]["learning_rate"] <= metrics[1]["learning_rate"]

    # Resume with the original total schedule, including an epoch-end remainder.
    import shutil

    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    shutil.copytree(checkpoints / "epoch_1", resumed_dir / "epoch_1")
    resumed = Stage1Trainer(
        model=AutoModelForCausalLM.from_pretrained(resumed_dir / "epoch_1"),
        args=build_arguments(cfg, resumed_dir, tmp_path / "resume_logs"), train_dataset=data,
        processing_class=tokenizer, data_collator=AssistantCollator(tokenizer.pad_token_id),
        log_dir=tmp_path / "resume_logs", run_identity="test",
    )
    resumed.train(resume_from_checkpoint=str(resumed_dir / "epoch_1"))
    assert resumed.state.global_step == 4 and resumed.state.epoch == 2
    assert torch.equal(resumed.model.transformer.wte.weight, trainer.model.transformer.wte.weight)


@pytest.mark.parametrize("kind", ["train", "evaluate"])
def test_stage1_notebooks_are_separate_thin_and_fresh(monkeypatch, kind):
    nbformat = pytest.importorskip("nbformat")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_stage1_notebook import cells, eval_cells

    notebook = nbformat.read(ROOT / f"notebooks/{kind}_stage1_sft.ipynb", as_version=4)
    nbformat.validate(notebook)
    expected = cells() if kind == "train" else eval_cells()
    assert [c.source for c in notebook.cells] == [c.source for c in expected]
    text = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    if kind == "train":
        assert "RUN_TRAINING = False" in text and "RUN_STAGE1_EVAL" not in text
        assert "setup_eval_runtime" not in text and "eval.run_" not in text
        assert "train/stage1_sft.py" in text
    else:
        assert "RUN_STAGE1_EVAL = False" in text and "RUN_TRAINING" not in text
        assert "setup_stage1_runtime" not in text and "train/stage1_sft.py" not in text
        assert "eval.run_batched_eval" in text and "eval.run_stage1_amc" not in text
    assert "BUNDLE =" not in text
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "stage1_notebook", "exec")
            assert cell.execution_count is None and not cell.outputs


def test_evaluation_notebook_prioritizes_final_epoch_and_makes_others_optional(tmp_path, monkeypatch):
    pytest.importorskip("nbformat")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_stage1_notebook import eval_cells

    train_root, baseline_root = tmp_path / "training", tmp_path / "baseline"
    for epoch in range(1, 6):
        folder = train_root / "checkpoints/stage1/sft_s1k" / f"epoch_{epoch}"
        folder.mkdir(parents=True)
        (folder / "stage1_checkpoint.json").write_text(json.dumps({
            "epoch": epoch, "run_identity": "same-run",
        }))
    results = baseline_root / "full/results"
    results.mkdir(parents=True)
    for name in ["eval_protocol.json", "eval_runtime.json"]:
        (results / name).write_text("{}")
    calls = []
    context = {
        "Path": Path, "json": json, "CODE_ROOT": str(ROOT),
        "TRAIN_ROOT": str(train_root), "BASELINE_ROOT": str(baseline_root),
        "EVAL_ENV": "/test/eval-env", "CONFIG": "configs/stage1_sft.yaml",
        "run_logged": lambda command, **kwargs: calls.append((command, kwargs)),
    }
    sources = [c.source for c in eval_cells() if c.cell_type == "code"]
    preflight = next(source for source in sources if "run_identities = set()" in source)
    run = next(source for source in sources if "RUN_STAGE1_EVAL = False" in source)
    exec(preflight, context)
    exec(run, context)
    assert calls == []  # default notebook execution performs no evaluation
    exec(run.replace("RUN_STAGE1_EVAL = False", "RUN_STAGE1_EVAL = True"), context)
    assert len(calls) == 1
    assert calls[0][0][calls[0][0].index("--run_name") + 1] == "sft_s1k_batched"
    calls.clear()
    include_earlier = preflight.replace("INCLUDE_EARLIER_EPOCHS = False", "INCLUDE_EARLIER_EPOCHS = True")
    exec(include_earlier, context)
    exec(run.replace("RUN_STAGE1_EVAL = False", "RUN_STAGE1_EVAL = True"), context)
    assert len(calls) == 5
    for epoch, (command, kwargs) in zip([5, 1, 2, 3, 4], calls, strict=True):
        assert command[1:3] == ["-m", "eval.run_batched_eval"]
        assert Path(command[command.index("--model") + 1]).name == f"epoch_{epoch}"
        expected_name = "sft_s1k" if epoch == 5 else f"sft_s1k_epoch{epoch}"
        assert command[command.index("--run_name") + 1] == expected_name + "_batched"
        assert command[command.index("--reuse_run_name") + 1] == expected_name
        assert command[command.index("--execution_config") + 1] == "configs/eval_execution.yaml"
        assert "--smoke" not in command and "--datasets" not in command
        assert kwargs["progress_totals"] == {
            "aime_2024": 30, "aime_2025": 30, "aime_2026": 30, "amc23": 40, "math_500": 500,
        }
        assert kwargs["log_path"] == train_root / f"full_eval_epoch{epoch}_batched_console.log"
    (train_root / "checkpoints/stage1/sft_s1k/epoch_3/stage1_checkpoint.json").unlink()
    calls.clear()
    # Missing an optional checkpoint must not block the default epoch-5 evaluation.
    exec(preflight, context)
    exec(run.replace("RUN_STAGE1_EVAL = False", "RUN_STAGE1_EVAL = True"), context)
    assert len(calls) == 1
    with pytest.raises(FileNotFoundError):
        exec(include_earlier, context)
