import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from train.dapo_data import load_config, select_model
from train.dapo_model import configure_tokenizer, stop_token_ids, training_prompt

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dapo_llama32_3b.yaml"


class NativeTokenFixture:
    chat_template = "native-template"
    vocabulary = {"<|end_of_text|>": 128001, "<|eom_id|>": 128008,
                  "<|eot_id|>": 128009, "<|finetune_right_pad_id|>": 128004}
    def get_vocab(self):
        return self.vocabulary
    @property
    def eos_token_id(self):
        return self.vocabulary[self.eos_token]
    @property
    def pad_token_id(self):
        return self.vocabulary[self.pad_token]
    def apply_chat_template(self, messages, **kwargs):
        self.received = (messages, kwargs)
        return "formatted"


def test_llama_source_tokens_and_fixed_prompt_date():
    cfg = load_config(CONFIG)
    source, identity = select_model(cfg, "base", "/does/not/exist")
    assert source == "meta-llama/Llama-3.2-3B-Instruct"
    assert identity["revision"] == "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    assert identity["trust_remote_code"] is False
    with pytest.raises(ValueError, match="no SFT source"):
        select_model(cfg, "sft", "/does/not/exist")
    tokenizer = configure_tokenizer(NativeTokenFixture(), cfg)
    assert tokenizer.padding_side == "left"
    assert stop_token_ids(tokenizer, cfg) == [128001, 128008, 128009]
    training_prompt(tokenizer, "Compute 2+2", cfg)
    messages, kwargs = tokenizer.received
    assert messages == [{"role": "user", "content":
                        r"Compute 2+2" + "\n\n" + r"Solve the problem step by step in English and put your final answer inside \boxed{}."}]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "date_string": "26 Jul 2024"}
    bad = deepcopy(cfg)
    bad["tokenizer"]["pad_token"] = "missing"
    with pytest.raises(ValueError, match="missing native"):
        configure_tokenizer(NativeTokenFixture(), bad)
    bad = deepcopy(cfg)
    bad["generation"]["stop_tokens"].append(tokenizer.pad_token)
    with pytest.raises(ValueError, match="exclude padding"):
        stop_token_ids(tokenizer, bad)


def test_rollout_receives_all_llama_stops_and_keeps_eom_tokens(tmp_path, monkeypatch):
    from train.dapo_rollout import VLLMRollout

    class Engine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def sleep(self, level):
            pass
        def generate(self, prompts, sampling_params, use_tqdm):
            self.params = sampling_params
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[SimpleNamespace(
                token_ids=[42, 128008], text=r"\boxed{4}", finish_reason="stop",
                logprobs=[{42: SimpleNamespace(logprob=-1.)}, {128008: SimpleNamespace(logprob=-2.)}])])
                for p in prompts]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine,
                         SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    cfg = load_config(CONFIG)
    tok = configure_tokenizer(NativeTokenFixture(), cfg)
    rollout = VLLMRollout("/pinned/local/model", cfg, tmp_path, tokenizer=tok)
    groups = rollout.generate([{"id": "sample", "prompt_token_ids": [3, 4]}], 0, 0)
    assert rollout.llm.kwargs["trust_remote_code"] is False
    assert all(p.stop_token_ids == [128001, 128008, 128009] for p in rollout.llm.params)
    assert len(groups[0]) == 8
    assert all(r["token_ids"] == [42, 128008] and r["logprobs"] == [-1., -2.]
               and r["finish_reason"] == "stop" for r in groups[0])


def test_llama_notebook_smoke_full_and_resume_commands(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_llama_dapo_notebook import cells
    nb = nbformat.read(ROOT / "notebooks/train_dapo_llama32_3b.ipynb", as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells] == [c.source for c in cells()]
    codes = [c.source for c in nb.cells if c.cell_type == "code"]
    for source in codes:
        compile(source, "llama_dapo_notebook", "exec")
        assert "@param" not in source and "eval.run_" not in source
    context = {"Path": Path, "json": json}
    exec(codes[0], context)
    assert "Llama32-3B" in context["OUTPUT_ROOT"]
    context.update(CODE_ROOT=str(ROOT), OUTPUT_ROOT=str(tmp_path))
    exec(next(s for s in codes if "TRAIN_COMMAND =" in s), context)
    assert "--sft-root" not in context["TRAIN_COMMAND"]
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "--smoke" in command:
            root = Path(command[command.index("--output-root") + 1])
            log = root / "logs" / (context["RUN_NAME"] + "_smoke") / "steps.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text('\n'.join(json.dumps(r) for r in [
                {"policy_iteration": 1, "minibatch_index": 1, "reused_rollout": False},
                {"policy_iteration": 1, "minibatch_index": 2, "reused_rollout": False,
                 "cached_rollout": True, "new_generated_tokens": 0}]))
    context["run_logged"] = run
    for flag in ["RUN_SMOKE", "RUN_TRAINING"]:
        source = next(s for s in codes if f"{flag} = False" in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(f"{flag} = False", f"{flag} = True"), context)
    smoke, full = calls
    assert "--smoke" in smoke and "--smoke" not in full
    assert smoke[smoke.index("--output-root") + 1] != full[full.index("--output-root") + 1]
    source = next(s for s in codes if "RUN_TRAINING = False" in s)
    exec(source.replace("RUN_TRAINING = False", "RUN_TRAINING = True")
         .replace('RESUME_CHECKPOINT = ""', 'RESUME_CHECKPOINT = "/saved/checkpoint-20"'), context)
    assert calls[-1][-2:] == ["--resume-from-checkpoint", "/saved/checkpoint-20"]


def test_llama_training_settings_match_reuse_experiment():
    llama, original = load_config(CONFIG), load_config(ROOT / "configs/dapo_reuse.yaml")
    for section in ["data", "hardware", "algorithm", "training", "rollout", "smoke"]:
        assert llama[section] == original[section]
