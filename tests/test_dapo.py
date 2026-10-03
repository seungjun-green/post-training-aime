import json
from pathlib import Path

import pytest

from common.io import digest, write_json
from train.dapo_data import candidate_order, load_config, prepare_data, select_model, verify_rows
from train.dapo_sampling import collect_groups, score_completion

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dapo.yaml"


def small_config():
    cfg = load_config(CONFIG)
    cfg["model_kind"] = "base"
    cfg["algorithm"].update(group_size=2, retained_groups=2, candidate_groups_per_batch=2,
                            max_generation_batches=3, max_completion_length=64,
                            soft_length_limit=32, logprob_chunk_tokens=8)
    return cfg


def completion(answer, length=10):
    return {"text": f"Answer: \\boxed{{{answer}}}", "token_ids": [1] * length,
            "logprobs": [-1.0] * length, "finish_reason": "stop"}


def test_shaping_boundaries_and_final_boxed_reward():
    a = load_config(CONFIG)["algorithm"]
    for length, penalty in [(100, 0), (16384, 0), (18432, -0.5), (20480, -1)]:
        good = score_completion(r"Earlier \boxed{12}. Final \boxed{34}.", "34", length, a)
        bad = score_completion(r"Final \boxed{12}.", "34", length, a)
        assert good["correct"] and good["reward"] == 1 + penalty
        assert not bad["correct"] and bad["reward"] == -1 + penalty
    assert not score_completion("34", "34", 2, a)["correct"]


def test_dynamic_refill_uses_mixed_correctness_not_shaped_reward_variance():
    cfg = small_config()
    rows = [{"id": str(i), "gold": "4"} for i in range(6)]
    calls, journal = [], []

    def generate(batch, step, batch_index):
        calls.extend(r["id"] for r in batch)
        if batch_index == 0:
            return [[completion(4), completion(4)], [completion(5), completion(5, 60)]]
        return [[completion(4), completion(5)] for _ in batch]

    selected, stats = collect_groups(rows, generate, cfg, 0, record_batch=lambda r, s: journal.extend(r))
    assert len(selected) == 4
    assert stats["all_correct_groups"] == stats["all_wrong_groups"] == 1
    assert stats["retained_groups"] == stats["mixed_groups"] == 2
    assert stats["acceptance_rate"] == 0.5
    assert len(calls) == len(set(calls)) == 4
    assert sum(r["retained"] for r in journal) == 4
    assert [r["correct"] for r in selected] == [True, False, True, False]


def test_dynamic_sampling_bounded_and_deterministic():
    cfg = small_config()
    rows = [{"id": str(i), "gold": "4"} for i in range(20)]
    assert candidate_order(rows, 42, 3) == candidate_order(rows, 42, 3)
    assert candidate_order(rows, 42, 3) != candidate_order(rows, 42, 4)
    batches = []
    def generate(batch, step, index):
        batches.append(batch)
        return [[completion(0), completion(0)] for _ in batch]
    with pytest.raises(RuntimeError, match="No optimizer update"):
        collect_groups(rows, generate, cfg, 0)
    assert len(batches) == 3


def test_base_does_not_require_sft_and_wrong_sft_source_rejected(tmp_path):
    cfg = load_config(CONFIG)
    source, identity = select_model(cfg, "base", tmp_path)
    assert source == cfg["model"]["repo"] and identity["kind"] == "base"
    with pytest.raises(ValueError):
        select_model(cfg, "pro", tmp_path)
    path = tmp_path / cfg["sft"]["relative_checkpoint"]
    write_json(path / "stage1_checkpoint.json", {"epoch": 5, "run_identity": "original"})
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"fixture")
    manifest = {"identity": "original", "config": {"run_name": "sft_s1k", "model": cfg["model"], "data": {}}}
    manifest_path = tmp_path / "logs/stage1/sft_s1k/run_manifest.json"
    write_json(manifest_path, manifest)
    assert select_model(cfg, "sft", tmp_path)[0] == str(path)
    manifest["config"]["data"]["columns"] = {"reasoning": "deepseek-v4-pro_reasoning"}
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="original DeepSeek"):
        select_model(cfg, "sft", tmp_path)


def test_data_digest_and_wrapper_and_no_prompt_truncation():
    from test_stage1_sft import CharacterTokenizer

    from pipeline.datasets import DAPO_PREFIX
    tokenizer = CharacterTokenizer()
    cfg = small_config()
    rows = [{"prompt": [{"role": "user", "content": DAPO_PREFIX + "What is 2+2?"}], "solution": "4"},
            {"prompt": "x" * 1000, "solution": "5"}]
    spec = {"expected_rows": 2, "content_digest": digest(rows), "ids_digest": digest(["a", "b"])}
    verify_rows(rows, ["a", "b"], spec)
    with pytest.raises(ValueError):
        verify_rows(rows[::-1], ["a", "b"], spec)
    cfg["data"]["max_prompt_tokens"] = 500
    prepared, report = prepare_data(rows, ["a", "b"], tokenizer, cfg)
    assert len(prepared) == 1 and prepared[0]["id"] == "a"
    assert "Solve the problem step by step in English" in prepared[0]["prompt"]
    assert DAPO_PREFIX not in prepared[0]["prompt"]
    assert not report["truncation_applied"] and report["dropped_prompts"][0]["id"] == "b"


def test_notebook_plain_settings_and_commands(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_dapo_notebook import cells
    nb = nbformat.read(ROOT / "notebooks/train_dapo_exaone.ipynb", as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells] == [c.source for c in cells()]
    sources = [c.source for c in nb.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "dapo_notebook", "exec")
        assert "@param" not in source and "eval.run_" not in source
    context = dict(Path=Path)
    exec(sources[0], context)
    context.update(MODEL_KIND="base", CODE_ROOT=str(ROOT), OUTPUT_ROOT=str(tmp_path), SFT_ROOT=str(tmp_path / "missing"))
    exec(next(s for s in sources if "TRAIN_COMMAND =" in s), context)
    calls = []
    context["run_logged"] = lambda command, **kw: calls.append((command, kw))
    for flag in ["RUN_SMOKE", "RUN_TRAINING"]:
        code = next(s for s in sources if f"{flag} = False" in s)
        n = len(calls)
        exec(code, context)
        assert len(calls) == n
        exec(code.replace(f"{flag} = False", f"{flag} = True"), context)
    smoke, full = [c[0] for c in calls]
    assert "--smoke" in smoke and "--smoke" not in full
    assert full[full.index("--model-kind") + 1] == "base"
    assert smoke[smoke.index("--output-root") + 1] != full[full.index("--output-root") + 1]


def test_full_config_and_smoke_keep_real_generation_budget():
    full, smoke = load_config(CONFIG), load_config(CONFIG, smoke=True)
    assert full["algorithm"]["group_size"] == smoke["algorithm"]["group_size"] == 8
    assert full["algorithm"]["max_completion_length"] == smoke["algorithm"]["max_completion_length"] == 20480
    assert full["training"]["max_steps"] == 100 and smoke["training"]["max_steps"] == 1
    assert full["training"]["warmup_steps"] == 20 and smoke["training"]["warmup_steps"] == 0
    assert full["algorithm"]["retained_groups"] == 16 and smoke["algorithm"]["retained_groups"] == 2


def test_real_trl_update_save_reload_resume_and_token_loss(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from datasets import Dataset
    from test_llama_lora_sft import tokenizer_fixture
    from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback, set_seed

    from train.dapo_trainer import DAPOTrainer, chunk_logps, make_arguments
    from train.run_dapo import validate_resume

    torch.set_num_threads(1)
    set_seed(42)
    tokenizer = tokenizer_fixture()
    cfg = small_config()
    cfg["training"].update(max_steps=2, save_steps=1, warmup_steps=0, learning_rate=1e-3,
                           bf16=False, use_cpu=True, dataloader_pin_memory=False, disable_tqdm=True)
    cfg["data"]["max_prompt_tokens"] = 500
    rows = [{"prompt": "What is 2+2?", "solution": "4"} for _ in range(8)]
    prepared, _ = prepare_data(rows, list(map(str, range(8))), tokenizer, cfg)
    model_config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=1024,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=False)
    model = LlamaForCausalLM(model_config)
    base = tmp_path / "base"
    model.save_pretrained(base)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}

    class FakeRollout:
        def __init__(self):
            self.syncs = 0
            self.sleep_calls = 0
        def sync(self, policy):
            self.policy = policy
            self.syncs += 1
        def sleep(self):
            self.sleep_calls += 1
        def generate(self, questions, step, index):
            groups = []
            for q in questions:
                outputs = []
                for answer in [4, 5]:
                    text = f"\\boxed{{{answer}}}"
                    ids = tokenizer.encode(text + tokenizer.eos_token, add_special_tokens=False)
                    all_ids = torch.tensor([q["prompt_token_ids"] + ids])
                    p = len(q["prompt_token_ids"])
                    with torch.no_grad():
                        logits = self.policy(input_ids=all_ids, use_cache=False).logits[0, p-1:-1]
                        logps = chunk_logps(logits, torch.tensor(ids), 1).tolist()
                    outputs.append({"token_ids": ids, "logprobs": logps, "text": text, "finish_reason": "stop"})
                groups.append(outputs)
            return groups

    logs, saved = tmp_path / "logs", tmp_path / "checkpoints"
    logs.mkdir()
    write_json(logs / "run_manifest.json", {"identity": "test"})
    class StopAfterOne(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_training_stop = True
    def trainer_for(model, callbacks=None):
        return DAPOTrainer(model=model, args=make_arguments(cfg, saved, logs),
            processing_class=tokenizer, train_dataset=Dataset.from_list(prepared), dapo_config=cfg,
            prepared_rows=prepared, rollout=FakeRollout(), log_dir=logs, run_identity="test", callbacks=callbacks)
    trainer = trainer_for(model, [StopAfterOne()])
    trainer.train()
    assert trainer.rollout.syncs == trainer.rollout.sleep_calls == 1
    assert any(not torch.equal(p, initial[n]) for n, p in model.named_parameters())
    path = saved / "checkpoint-1"
    for f in ["model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "dapo_checkpoint.json"]:
        assert (path / f).is_file()
    reloaded = LlamaForCausalLM.from_pretrained(path)
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, dict(reloaded.named_parameters())[n])
    validate_resume(saved, logs, str(path), "test")
    resumed = trainer_for(LlamaForCausalLM.from_pretrained(base))
    resumed.train(resume_from_checkpoint=str(path))
    assert resumed.state.global_step == 2 and resumed.rollout.syncs == 1
    records = [json.loads(line) for line in (logs / "steps.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 2]
    assert all(r["retained_groups"] == 2 for r in records)
    assert (saved / "checkpoint-2/dapo_checkpoint.json").is_file()

    # Analytic check: asymmetric clipping and one denominator across microbatches.
    old_hook = resumed._get_per_token_logps_and_entropies
    def fake_hook(model, ids, mask, logits_to_keep, **kwargs):
        logps = torch.log(torch.tensor([[1.5, 1.1]], requires_grad=True))
        return logps, torch.zeros_like(logps)
    resumed._get_per_token_logps_and_entropies = fake_hook
    inputs = {"prompt_ids": torch.ones((1, 1), dtype=torch.long), "prompt_mask": torch.ones((1, 1)),
              "completion_ids": torch.ones((1, 2), dtype=torch.long), "completion_mask": torch.ones((1, 2)),
              "advantages": torch.tensor([1.0]), "old_per_token_logps": torch.zeros((1, 2)),
              "importance_sampling_ratio": torch.ones((1, 2)), "num_items_in_batch": torch.tensor(4)}
    loss = resumed._compute_loss(resumed.model, inputs)
    assert float(loss.detach()) == pytest.approx(-(1.28 + 1.1) / 4)
    inputs["advantages"] = torch.tensor([-1.0])
    inputs["old_per_token_logps"] = torch.log(torch.tensor([[2.5, 2.0]]))
    loss = resumed._compute_loss(resumed.model, inputs)
    assert float(loss.detach()) == pytest.approx((0.8 + 0.8) / 4)
    resumed._get_per_token_logps_and_entropies = old_hook


def test_rollout_transport_updates_weights_and_preserves_sample_logps(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import sys
    from types import SimpleNamespace

    from train.dapo_rollout import VLLMRollout

    class WorkerModel:
        def load_weights(self, items):
            self.weights = dict(items)
            return set(self.weights)
    class FakeLLM:
        def __init__(self, **kwargs):
            self.model = WorkerModel()
            self.sleep_levels, self.wakes, self.resets = [], 0, 0
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: None))
        def sleep(self, level):
            self.sleep_levels.append(level)
        def wake_up(self):
            self.wakes += 1
        def apply_model(self, fn):
            return [fn(self.model)]
        def reset_prefix_cache(self):
            self.resets += 1
        def generate(self, prompts, sampling_params, use_tqdm):
            self.params = sampling_params
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[
                SimpleNamespace(token_ids=[2], text="test", finish_reason="stop",
                                logprobs=[{9: SimpleNamespace(logprob=-0.1), 2: SimpleNamespace(logprob=-2.0)}])
            ]) for p in prompts]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    cfg = small_config()
    engine = VLLMRollout("/fake/source", cfg, tmp_path)
    policy = torch.nn.Linear(2, 2)
    engine.sync(policy)
    assert engine.llm.model.weights["weight"].dtype == torch.bfloat16
    with torch.no_grad():
        policy.weight.fill_(2.0)
    engine.sleep()
    engine.sync(policy)
    assert torch.all(engine.llm.model.weights["weight"] == 2.0)
    generated = engine.generate([{"id": "one", "prompt_token_ids": [1, 4]}], 0, 0)
    assert len(generated) == 1 and len(generated[0]) == 2
    assert all(r["logprobs"] == [-2.0] for r in generated[0])
    assert len({p.seed for p in engine.llm.params}) == 2
    assert engine.llm.wakes == engine.llm.resets == 2
    engine.close()
