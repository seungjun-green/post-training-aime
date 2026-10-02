import json
from copy import deepcopy
from pathlib import Path

import pytest

from train.llama_sft_data import configure_tokenizer, prepare_llama_examples, select_smoke_examples
from train.stage1_llama_lora import attach_lora, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage1_sft_llama31_lora.yaml"


def tokenizer_fixture():
    """Tiny byte tokenizer with Llama's text-only native chat layout, no gated files."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast

    special = ["<|begin_of_text|>", "<|eot_id|>", "<|finetune_right_pad_id|>",
               "<|start_header_id|>", "<|end_header_id|>", "[UNK]"]
    raw = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(["What is 2+2? Reasoning. The answer is 4."],
                            trainers.BpeTrainer(vocab_size=262, special_tokens=special,
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token=special[0],
                                       eos_token=special[1], unk_token="[UNK]")
    tokenizer.chat_template = (
        "{{ bos_token }}<|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + "
        "message['content']|trim + '<|eot_id|>' }}{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
    )
    return configure_tokenizer(tokenizer, load_config(CONFIG))


def row(reasoning="Reasoning"):
    return {"question": "What is 2+2?", "deepseek_thinking_trajectory": reasoning,
            "deepseek_attempt": "The answer is \\boxed{4}.\n\n", "deepseek_grade": "No",
            "deepseek-v4-pro_reasoning": "WRONG COLUMN", "deepseek-v4-pro_answer": "WRONG COLUMN"}


def test_llama_native_mask_and_whole_sequence_filter():
    tokenizer = tokenizer_fixture()
    cfg = load_config(CONFIG)
    rows = [row(), row("Reasoning " * 10)]
    examples, _, _ = prepare_llama_examples(tokenizer, rows, ["short", "long"], cfg)
    cfg["max_seq_length"] = len(examples[0]["input_ids"])
    examples, report, preview = prepare_llama_examples(tokenizer, rows, ["short", "long"], cfg)
    ex = examples[0]
    assert len(ex["input_ids"]) == cfg["max_seq_length"]
    visible = tokenizer.decode([v for v in ex["labels"] if v != -100])
    assert visible == "<think>\nReasoning\n</think>\n\nThe answer is \\boxed{4}.<|eot_id|>"
    assert "WRONG COLUMN" not in preview[1]
    assert ex["labels"][0] == -100
    assert ex["labels"][-1] == tokenizer.eos_token_id
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    assert report["dropped_overlength"] == 1
    assert report["kept_ids"] == ["short"]
    assert not report["grade_filter_applied"] and not report["truncation_applied"]
    assert report["native_template_trailing_whitespace_normalized_rows"] == 2
    assert rows[0]["deepseek_attempt"].endswith("\n\n")  # source is unmodified


def test_config_and_smoke_isolation():
    full = load_config(CONFIG)
    smoke = load_config(CONFIG, smoke=True)
    assert full["training"]["num_train_epochs"] == 5
    assert full["training"]["learning_rate"] == 5e-5
    assert full["training"]["gradient_accumulation_steps"] == 16
    assert full["max_seq_length"] == 20480
    assert full["lora"]["r"] == 32 and full["lora"]["lora_alpha"] == 64
    assert full["lora"]["lora_dropout"] == 0.05
    assert smoke["run_name"] == full["run_name"] + "_smoke"
    assert smoke["training"]["num_train_epochs"] == smoke["training"]["gradient_accumulation_steps"] == 1
    examples = [{"input_ids": [1] * n} for n in [10, 20, 15]]
    selected, report = select_smoke_examples(examples, {"kept_ids": ["a", "b", "c"]}, smoke)
    assert selected == [examples[1]]
    assert report["smoke"] == {"selection": "longest", "ids": ["b"], "lengths": [20]}


def test_notebook_commands_are_training_only_and_smoke_is_separate(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    pytest.importorskip("nbformat")
    from build_llama_sft_notebook import cells
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks/train_stage1_llama31_lora.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells()]
    sources = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "llama_notebook", "exec")
        assert "eval.run_" not in source
    calls = []
    command = ["python", "train/stage1_llama_lora.py", "--config", str(CONFIG),
               "--output_root", str(tmp_path)]
    context = dict(Path=Path, TRAIN_ROOT=str(tmp_path), CODE_ROOT=str(ROOT), TRAIN_COMMAND=command,
                   run_logged=lambda command, **kw: calls.append((command, kw)))
    for flag in ["RUN_SMOKE", "RUN_TRAINING"]:
        source = next(s for s in sources if f"{flag} = False" in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(f"{flag} = False", f"{flag} = True"), context)
    smoke, full = [c[0] for c in calls]
    assert "--smoke" in smoke and "--smoke" not in full
    assert Path(smoke[-2]).is_relative_to(tmp_path / "smoke_attempts")
    assert full == command and context["TRAIN_COMMAND"] == command


def test_tiny_llama_lora_training_save_reload_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from datasets import Dataset
    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback, set_seed
    from train.sft_data import AssistantCollator
    from train.sft_trainer import Stage1Trainer
    from train.stage1_sft import build_arguments, validate_resume

    torch.set_num_threads(1)
    set_seed(42)
    tokenizer = tokenizer_fixture()
    cfg = load_config(CONFIG)
    cfg["training"].update(bf16=False, use_cpu=True, dataloader_pin_memory=False,
                           disable_tqdm=True, gradient_accumulation_steps=4)
    model_config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=16,
                               intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, num_key_value_heads=1,
                               max_position_embeddings=1024, eos_token_id=tokenizer.eos_token_id,
                               pad_token_id=tokenizer.pad_token_id, use_cache=False)
    base = LlamaForCausalLM(model_config)
    base_path = tmp_path / "base"
    base.save_pretrained(base_path)
    base_state = {name: p.detach().clone() for name, p in base.named_parameters()}
    model, report = attach_lora(LlamaForCausalLM.from_pretrained(base_path), cfg)
    assert len(report["targeted_modules"]) == 7
    adapters_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    examples, _, _ = prepare_llama_examples(tokenizer, [row()] * 6, list(range(6)), cfg)
    logs, checkpoints = tmp_path / "logs", tmp_path / "checkpoints"
    logs.mkdir()
    identity = "tiny-lora-test"
    (logs / "run_manifest.json").write_text(json.dumps({"identity": identity}))

    class StopAfterFirstEpoch(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_training_stop = True

    def trainer_for(model, callbacks=None):
        return Stage1Trainer(model=model, args=build_arguments(cfg, checkpoints, logs),
                             train_dataset=Dataset.from_list(examples), processing_class=tokenizer,
                             data_collator=AssistantCollator(tokenizer.pad_token_id),
                             log_dir=logs, run_identity=identity, callbacks=callbacks)

    trainer = trainer_for(model, [StopAfterFirstEpoch()])
    trainer.train()
    assert trainer.state.global_step == 2  # includes the final partial accumulation
    saved = checkpoints / "epoch_1"
    for name in ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
                 "rng_state.pth", "trainer_state.json", "tokenizer_config.json", "stage1_checkpoint.json"]:
        assert (saved / name).is_file()
    adapter_config = json.loads((saved / "adapter_config.json").read_text())
    assert adapter_config["revision"] == cfg["model"]["revision"]
    assert any(not torch.equal(p, dict(model.named_parameters())[n]) for n, p in adapters_before.items())
    for name, p in model.get_base_model().named_parameters():
        if "lora_" not in name:
            assert torch.equal(p, base_state[name.replace(".base_layer", "")])
    reloaded = PeftModel.from_pretrained(LlamaForCausalLM.from_pretrained(base_path), saved)
    batch = AssistantCollator(tokenizer.pad_token_id)(examples[:1])
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).logits, reloaded(**batch).logits)

    validate_resume(checkpoints, logs, str(saved), identity)
    resumed, _ = attach_lora(LlamaForCausalLM.from_pretrained(base_path), cfg)
    trainer = trainer_for(resumed)
    trainer.train(resume_from_checkpoint=str(saved))
    assert trainer.state.global_step == 10
    for epoch in range(1, 6):
        assert (checkpoints / f"epoch_{epoch}/stage1_checkpoint.json").is_file()
    records = [json.loads(line) for line in (logs / "steps.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == list(range(1, 11))
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in records)
