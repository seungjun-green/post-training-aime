"""Exercise rollout reuse through the pinned TRL optimizer loop, not a mock loss."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from train.dapo_data import load_config, prepare_data

ROOT = Path(__file__).resolve().parents[1]


def test_reuse_config_and_cycle_boundaries(tmp_path):
    path = ROOT / "configs/dapo_reuse.yaml"
    full, smoke = load_config(path), load_config(path, smoke=True)
    assert full["algorithm"]["num_iterations"] == smoke["algorithm"]["num_iterations"] == 2
    assert full["training"]["max_steps"] == 100
    assert smoke["training"]["max_steps"] == smoke["training"]["save_steps"] == 2
    for section, field, value in [("algorithm", "num_iterations", 0),
                                   ("algorithm", "num_iterations", True),
                                   ("training", "max_steps", 3), ("training", "save_steps", 1)]:
        bad = deepcopy(full)
        bad[section][field] = value
        target = tmp_path / "bad.yaml"
        target.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError, match="num_iterations|rollout cycle"):
            load_config(target)
    # Old manifests retain their exact configuration, including the absent key.
    assert "num_iterations" not in load_config(ROOT / "configs/dapo.yaml")["algorithm"]
    assert "num_iterations" not in load_config(ROOT / "configs/dapo_continue_300.yaml")["algorithm"]


@pytest.mark.parametrize("llama", [False, True])
@pytest.mark.parametrize("minibatch", [False, True])
def test_clipped_gradients_fixed_old_policy_and_exact_resume(tmp_path, llama, minibatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from datasets import Dataset
    from test_llama_lora_sft import tokenizer_fixture
    from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback, set_seed

    from train.dapo_model import configure_tokenizer
    from train.dapo_trainer import DAPOTrainer, make_arguments

    torch.set_num_threads(1)
    set_seed(42)
    tokenizer = tokenizer_fixture()
    cfg = load_config(ROOT / "configs" / ("dapo_llama32_3b.yaml" if llama else "dapo_reuse.yaml"))
    if llama:
        tokenizer.add_special_tokens({"additional_special_tokens": [
            "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>", "<|finetune_right_pad_id|>"]})
        configure_tokenizer(tokenizer, cfg)
    cfg["model_kind"] = "base"
    cfg["algorithm"].update(group_size=2, retained_groups=2, candidate_groups_per_batch=2,
                            max_completion_length=128, soft_length_limit=120, logprob_chunk_tokens=8)
    if minibatch:
        cfg["algorithm"].update(num_iterations=1, mini_batch_size=2)
    # Deliberately large test-only LR makes clipping observable in a tiny model.
    # Production LR/clip thresholds stay unchanged.
    cfg["training"].update(max_steps=4, save_steps=2, warmup_steps=0, learning_rate=0.05,
                           max_grad_norm=1000, bf16=False, use_cpu=True,
                           dataloader_pin_memory=False, disable_tqdm=True)
    cfg["data"]["max_prompt_tokens"] = 500
    rows = [{"prompt": "What is 2+2?" + " Please calculate." * i, "solution": "4"} for i in range(8)]
    prepared, _ = prepare_data(rows, list(map(str, range(8))), tokenizer, cfg)
    initial = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=1024,
        tie_word_embeddings=llama,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=False))

    def logps(model, prompt, completion):
        ids = torch.tensor([prompt + completion])
        logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt)-1:-1]
        return logits.log_softmax(-1).gather(-1, torch.tensor(completion)[:, None]).squeeze(-1)

    class Rollout:
        def __init__(self):
            self.steps, self.snapshots = [], []
        def sync(self, model):
            self.old_policy = deepcopy(model)
            self.snapshots.append({n: p.detach().clone() for n, p in model.named_parameters()})
        def sleep(self):
            pass
        def generate(self, questions, step, index):
            self.steps.append(step)
            groups, self.records = [], []
            for q in questions:
                group = []
                for text, sign in [(r"\boxed{4}", 1), (r"Let me think. \boxed{5}", -1)]:
                    ending = "<|eom_id|>" if llama and sign < 0 else tokenizer.eos_token
                    ids = tokenizer.encode(text + ending, add_special_tokens=False)
                    with torch.no_grad():
                        old = logps(self.old_policy, q["prompt_token_ids"], ids)
                    self.records.append((q["prompt_token_ids"], ids, old, sign))
                    group.append({"text": text, "token_ids": ids, "logprobs": old.tolist(), "finish_reason": "stop"})
                groups.append(group)
            return groups

    class AuditTrainer(DAPOTrainer):
        def __init__(self, *args, **kwargs):
            self.seen = {}
            super().__init__(*args, **kwargs)
        def _compute_loss(self, model, inputs):
            self.seen.setdefault(self.state.global_step, []).append({
                key: inputs[key].detach().clone()
                for key in ["prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
                            "old_per_token_logps", "advantages", "num_items_in_batch"]})
            return super()._compute_loss(model, inputs)

    class AuditGradient(TrainerCallback):
        def __init__(self, rollout):
            self.rollout, self.clipped = rollout, []
        def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
            oracle = deepcopy(model)
            oracle.zero_grad(set_to_none=True)
            terms, tokens, clipped = [], 0, 0
            records = self.rollout.records
            if minibatch:
                # Locate the exact disjoint examples selected AFTER TRL's shuffle.
                lookup = {(tuple(p), tuple(ids)): (old, sign) for p, ids, old, sign in records}
                records = []
                for batch in self.trainer.seen[state.global_step]:
                    prompt = batch["prompt_ids"][batch["prompt_mask"].bool()].tolist()
                    ids = batch["completion_ids"][batch["completion_mask"].bool()].tolist()
                    old, sign = lookup[(tuple(prompt), tuple(ids))]
                    torch.testing.assert_close(batch["old_per_token_logps"][batch["completion_mask"].bool()],
                                               old, rtol=1e-6, atol=1e-6)
                    torch.testing.assert_close(batch["advantages"], torch.tensor([sign / (2**0.5 + 1e-4)]))
                    records.append((prompt, ids, old, sign))
                assert len(records) == 2
                denominator = sum(len(ids) for _, ids, _, _ in records)
                assert all(int(b["num_items_in_batch"]) == denominator for b in self.trainer.seen[state.global_step])
            for prompt, ids, old, sign in records:
                ratio = (logps(oracle, prompt, ids) - old).exp()
                advantage = sign / (2**0.5 + 1e-4)
                terms.append(-torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.28) * advantage).sum())
                clipped += int(((ratio > 1.28) if sign > 0 else (ratio < 0.8)).sum())
                tokens += len(ids)
            (sum(terms) / tokens).backward()
            for name, param in model.named_parameters():
                torch.testing.assert_close(param.grad, dict(oracle.named_parameters())[name].grad,
                                           rtol=3e-4, atol=3e-6)
            self.clipped.append(clipped)

    class StopAfterCycle(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_training_stop = True

    def make_trainer(name, stop=False):
        set_seed(42)
        rollout = Rollout()
        audit = AuditGradient(rollout)
        trainer = AuditTrainer(model=deepcopy(initial),
            args=make_arguments(cfg, tmp_path / name / "weights", tmp_path / name / "logs"),
            processing_class=tokenizer, train_dataset=Dataset.from_list(prepared), dapo_config=cfg,
            prepared_rows=prepared, rollout=rollout, log_dir=tmp_path / name / "logs", run_identity="reuse-test",
            callbacks=[audit] + ([StopAfterCycle()] if stop else []))
        audit.trainer = trainer
        return trainer, audit

    full, audit = make_trainer("full")
    full.train()
    assert full.rollout.steps == [0, 2]  # Four optimizer updates, only two fresh rollouts.
    assert audit.clipped[0] == audit.clipped[2] == 0
    assert audit.clipped[1] > 0  # Actual second-pass clipping, no fabricated log-probabilities.
    for first, second in [(0, 1), (2, 3)]:
        if minibatch:
            def keys(step):
                return {(tuple(b["prompt_ids"][b["prompt_mask"].bool()].tolist()),
                         tuple(b["completion_ids"][b["completion_mask"].bool()].tolist()))
                        for b in full.seen[step]}
            assert len(keys(first)) == len(keys(second)) == 2
            assert not keys(first) & keys(second)  # No repeated answers across updates.
            assert len(keys(first) | keys(second)) == 4
        else:
            for a, b in zip(full.seen[first], full.seen[second], strict=True):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert any(not torch.equal(full.rollout.snapshots[0][n], p)
               for n, p in full.rollout.snapshots[1].items())
    logs = [json.loads(line) for line in (tmp_path / "full/logs/steps.jsonl").read_text().splitlines()]
    assert [r["policy_iteration"] for r in logs] == ([1, 1, 1, 1] if minibatch else [1, 2, 1, 2])
    assert [r["rollout_first_update"] for r in logs] == [1, 1, 3, 3]
    assert [r["reused_rollout"] for r in logs] == ([False] * 4 if minibatch else [False, True, False, True])
    if minibatch:
        assert [r["minibatch_index"] for r in logs] == [1, 2, 1, 2]
    assert logs[0]["new_generated_tokens"] > 0 and logs[1]["new_generated_tokens"] == 0
    assert logs[1]["policy_metrics"]["clip_ratio/region_mean"] > 0
    if llama:
        # EOM-terminated samples must not be misreported as truncated by TRL's
        # single-EOS length metrics. Tied input/output embeddings remain tied.
        assert logs[0]["policy_metrics"]["completions/clipped_ratio"] == 0
        assert full.model.lm_head.weight is full.model.model.embed_tokens.weight

    stopped, _ = make_trainer("resumed", stop=True)
    stopped.train()
    checkpoint = tmp_path / "resumed/weights/checkpoint-2"
    resumed, _ = make_trainer("resumed")
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.rollout.steps == [2]
    for name, p in full.model.named_parameters():
        torch.testing.assert_close(p, dict(resumed.model.named_parameters())[name], rtol=0, atol=0)
    resumed.state.global_step = 3
    with pytest.raises(ValueError, match="complete rollout cycle"):
        resumed._save_checkpoint(resumed.model, trial=None)
