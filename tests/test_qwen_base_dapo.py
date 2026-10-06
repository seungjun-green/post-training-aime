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

from common.io import write_json
from train.dapo_data import batch_schedule, load_config, select_model
from train.dapo_model import configure_tokenizer, stop_token_ids, training_prompt

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dapo_qwen25_3b_base.yaml"
NOTEBOOK = ROOT / "notebooks/train_dapo_qwen25_3b_base.ipynb"


def native_tokens():
    from test_llama_dapo import NativeTokenFixture

    class Tokens(NativeTokenFixture):
        vocabulary = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643}
        eos_token = pad_token = "<|endoftext|>"
    return Tokens()


def test_base_source_and_identical_dapo_settings():
    cfg = load_config(CONFIG)
    instruct = load_config(ROOT / "configs/dapo_qwen25_3b.yaml")
    source, identity = select_model(cfg, "base", "/unused")
    assert source == "Qwen/Qwen2.5-3B"
    assert identity["revision"] == "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
    assert cfg["run_name_prefix"] != instruct["run_name_prefix"]
    assert "sft" not in cfg
    for key in ["data", "hardware", "algorithm", "training", "rollout", "smoke"]:
        assert cfg[key] == instruct[key], key
    assert "tokenizer" not in cfg and "generation" not in cfg
    tok = configure_tokenizer(native_tokens(), cfg)
    assert stop_token_ids(tok, cfg) is None
    assert tok.eos_token_id == tok.pad_token_id == 151643
    training_prompt(tok, "Compute 2+2", cfg)
    assert tok.received[0][0]["content"].startswith("Compute 2+2\n\nSolve the problem")
    assert tok.received[1] == {"tokenize": False, "add_generation_prompt": True}
    assert batch_schedule(cfg)["updates_per_rollout"] == 2
    smoke = load_config(CONFIG, smoke=True)
    assert batch_schedule(smoke)["rollout_batch_size"] == 16
    assert smoke["training"]["max_steps"] == 2


@pytest.mark.parametrize("base", [True, False])
def test_native_base_eos_preserved_without_changing_instruct_sampling(tmp_path, monkeypatch, base):
    from train.dapo_rollout import VLLMRollout

    class Engine:
        def __init__(self, **kwargs):
            pass
        def sleep(self, level):
            pass
        def generate(self, prompts, sampling_params, use_tqdm):
            self.params = sampling_params
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[SimpleNamespace(
                token_ids=[42, tok.eos_token_id], text=r"\boxed{4}", finish_reason="stop",
                logprobs=[{42: SimpleNamespace(logprob=-1.)}, {tok.eos_token_id: SimpleNamespace(logprob=-2.)}])])
                for p in prompts]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace))
    cfg = load_config(CONFIG if base else ROOT / "configs/dapo_qwen25_3b.yaml")
    tok = configure_tokenizer(native_tokens(), cfg)
    for source in ["/original-snapshot", "/resumed-checkpoint"]:
        rollout = VLLMRollout(source, cfg, tmp_path, tokenizer=tok)
        groups = rollout.generate([{"id": "q", "prompt_token_ids": [3, 4]}], 0, 0)
        assert len(groups[0]) == 8
        assert all(r["token_ids"] == [42, 151643 if base else 151645] for r in groups[0])
        for params in rollout.llm.params:
            assert getattr(params, "stop_token_ids", None) == (None if base else [151645])
            assert not getattr(params, "ignore_eos", False)
            assert params.max_tokens == 20480 and params.temperature == params.top_p == 1


def make_checkpoint(context, step):
    cfg = context["cfg"]
    checkpoint = context["checkpoint_root"] / f"checkpoint-{step}"
    write_json(checkpoint / "dapo_checkpoint.json", {"run_identity": "base-run", "global_step": step})
    write_json(checkpoint / "trainer_state.json", {"global_step": step, "max_steps": 300})
    for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json", "tokenizer_config.json"]:
        (checkpoint / name).write_text("fixture")
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (checkpoint / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    write_json(context["run_logs"] / "run_manifest.json", {"identity": "base-run", "config": cfg})
    return checkpoint


def test_notebook_new_run_smoke_auto_resume_and_completed_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_qwen_base_dapo_notebook import cells

    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells] == [c.source for c in cells()]
    codes = [c.source for c in nb.cells if c.cell_type == "code"]
    for code in codes:
        compile(code, "base_rl", "exec")
        assert "@param" not in code and "RECOVER_FROM_CHECKPOINT" not in code
        assert '"--recover-from-checkpoint"' not in code
    context = {"Path": Path, "json": json}
    exec(codes[0], context)
    assert context["OUTPUT_ROOT"].endswith("Experiments/base-rl-native-eos")
    context.update(CODE_ROOT=str(ROOT), OUTPUT_ROOT=str(tmp_path))
    exec(next(s for s in codes if "TRAIN_COMMAND =" in s), context)
    assert context["RUN_NAME"] == "dapo_qwen25_3b_pretrained_base"
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
    exec(next(s for s in codes if 'TRAIN_COMMAND + ["--prepare-only"]' in s), context)
    for flag in ["RUN_SMOKE", "RUN_TRAINING"]:
        source = next(s for s in codes if f"{flag} = False" in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(f"{flag} = False", f"{flag} = True"), context)
    assert len(calls) == 3 and "--resume-from-checkpoint" not in calls[-1]
    full = next(s for s in codes if "RUN_TRAINING = False" in s).replace("RUN_TRAINING = False", "RUN_TRAINING = True")
    checkpoint = make_checkpoint(context, 20)
    exec(full, context)
    assert calls[-1][-2:] == ["--resume-from-checkpoint", str(checkpoint)]
    # Even a completed old run copied into the new folder must be rejected.
    old_config = dict(context["cfg"], tokenizer={"eos_token": "<|im_end|>"})
    write_json(context["run_logs"] / "run_manifest.json", {"identity": "base-run", "config": old_config})
    with pytest.raises(ValueError, match="another training run"):
        exec(full, context)
    make_checkpoint(context, 20)
    with pytest.raises(ValueError, match="latest completed"):
        exec(full.replace('RESUME_CHECKPOINT = ""', 'RESUME_CHECKPOINT = "/other/run"'), context)
    make_checkpoint(context, 300)
    before = len(calls)
    exec(full, context)
    assert len(calls) == before
    exec(codes[-1], context)


def test_bundle_is_current_isolated_and_reproducible(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_qwen_base_dapo_notebook import BUNDLE_FILES

    nb = nbformat.read(NOTEBOOK, as_version=4)
    setup = nb.cells[3].source
    assign = next(n for n in ast.parse(setup).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "BUNDLE" for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assign.value)))) as archive:
        assert set(archive.namelist()) == set(BUNDLE_FILES)
        for name in archive.namelist():
            assert archive.read(name) == (ROOT / name).read_bytes()
    (tmp_path / "setup.txt").write_text(setup.split('subprocess.check_call([sys.executable')[0])
    script = '''
import sys
from pathlib import Path
from types import SimpleNamespace
root = Path(sys.argv[1])
sys.modules["google.colab"] = SimpleNamespace(
    drive=SimpleNamespace(mount=lambda _: None), userdata=SimpleNamespace(get=lambda _: "fake-token"))
commits = []
for folder in ["first", "first", "second"]:
    namespace = {"CODE_PARENT": str(root / folder)}
    exec((root / "setup.txt").read_text(), namespace)
    commits.append(namespace["git"]("rev-parse", "HEAD"))
assert len(set(commits)) == 1
code = Path(namespace["CODE_ROOT"])
sys.path.insert(0, str(code))
from train import run_dapo, dapo_rollout, dapo_sampling
from eval.run_english_eval import git_identity
assert run_dapo.ROOT == code and git_identity() == commits[0]
cfg = run_dapo.load_config(code / "configs/dapo_qwen25_3b_base.yaml")
assert run_dapo.select_model(cfg, "base", "")[0] == "Qwen/Qwen2.5-3B"
assert "generation" not in cfg and "tokenizer" not in cfg
for lock in code.glob("requirements*.lock"):
    for line in lock.read_text().splitlines():
        if line.startswith("-r "):
            assert (code / line[3:]).is_file()
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
