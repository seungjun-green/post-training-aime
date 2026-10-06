import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import nbformat
import pytest

from train.dapo_data import load_config, select_model
from train.dapo_model import configure_tokenizer, stop_token_ids, training_prompt

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dapo_qwen25_3b.yaml"


def recovery_parent(root):
    from test_dapo_continuation import parent_run

    from common.io import write_json

    old, manifest = parent_run(root)
    parent = root / "checkpoints/dapo_qwen25_3b_base"
    old.parent.rename(parent)
    checkpoint = parent / "checkpoint-40"
    (parent / old.name).rename(checkpoint)
    cfg = load_config(CONFIG)
    cfg["model_kind"] = "base"
    cfg["data"]["revision"] = "a" * 40
    manifest["config"] = cfg
    write_json(root / "logs/dapo_qwen25_3b_base/run_manifest.json", manifest)
    write_json(checkpoint / "dapo_checkpoint.json", {"run_identity": "parent", "global_step": 40, "model_kind": "base"})
    write_json(checkpoint / "trainer_state.json", {"global_step": 40, "max_steps": 300})
    return checkpoint, manifest


def test_memory_recovery_pins_parent_preserves_settings_and_runtime(tmp_path):
    from train.dapo_resume import check_extension_runtime, recovery_source

    checkpoint, previous = recovery_parent(tmp_path / "original")
    before = {str(p.relative_to(tmp_path / "original")): p.read_bytes()
              for p in (tmp_path / "original").rglob("*") if p.is_file()}
    cfg = load_config(CONFIG)
    cfg["model_kind"] = "base"
    parent, recovery = recovery_source(cfg, checkpoint, tmp_path / "fixed")
    assert parent == previous
    assert recovery["from_step"] == 40 and recovery["target_step"] == 300
    assert cfg == previous["config"]
    changed = deepcopy(cfg)
    changed["algorithm"]["max_completion_length"] = 8192
    with pytest.raises(ValueError, match="preserve all original"):
        recovery_source(changed, checkpoint, tmp_path / "fixed")
    with pytest.raises(ValueError, match="separate OUTPUT_ROOT"):
        recovery_source(cfg, checkpoint, tmp_path / "original")
    runtime = deepcopy(previous)
    runtime["git_commit"] = "memory-fix-code"
    check_extension_runtime(previous, runtime)
    runtime["packages"]["torch"] = "changed"
    with pytest.raises(ValueError, match="packages"):
        check_extension_runtime(previous, runtime)
    after = {str(p.relative_to(tmp_path / "original")): p.read_bytes()
             for p in (tmp_path / "original").rglob("*") if p.is_file()}
    assert before == after


def test_memory_recovery_retry_before_first_new_checkpoint(tmp_path):
    from common.io import read_jsonl, write_json, write_jsonl
    from train.run_dapo import validate_resume

    checkpoints, logs = tmp_path / "checkpoints", tmp_path / "logs"
    checkpoints.mkdir()
    recovery = {"from_step": 40}
    validate_resume(checkpoints, logs, None, "fixed", recovery)
    write_json(logs / "run_manifest.json", {"identity": "fixed"})
    write_jsonl(logs / "steps.jsonl", [{"step": 41}, {"step": 42}, {"step": 43}])
    validate_resume(checkpoints, logs, None, "fixed", recovery)
    assert read_jsonl(logs / "steps.jsonl") == []
    assert read_jsonl(next(logs.glob("abandoned_steps_*.jsonl"))) == [{"step": 41}, {"step": 42}, {"step": 43}]
    with pytest.raises(ValueError, match="Existing DAPO run"):
        validate_resume(checkpoints, logs, None, "different", recovery)
    new_checkpoint = checkpoints / "checkpoint-60"
    write_json(new_checkpoint / "dapo_checkpoint.json", {"run_identity": "fixed", "global_step": 60})
    with pytest.raises(ValueError, match="Existing DAPO run"):
        validate_resume(checkpoints, logs, None, "fixed", recovery)
    validate_resume(checkpoints, logs, str(new_checkpoint), "fixed", recovery)


def test_recovery_notebook_selects_parent_then_new_checkpoint(tmp_path):
    from common.io import write_json

    nb = nbformat.read(ROOT / "notebooks/train_dapo_qwen25_3b.ipynb", as_version=4)
    context = {"Path": Path, "json": json}
    exec(nb.cells[1].source, context)
    assert context["OUTPUT_ROOT"].endswith("-MemoryFix")
    assert context["RECOVER_FROM_CHECKPOINT"].endswith("checkpoint-40")
    context.update(CODE_ROOT=str(ROOT), OUTPUT_ROOT=str(tmp_path))
    exec(nb.cells[5].source, context)
    calls = []
    context["run_logged"] = lambda command, **kwargs: calls.append(command)
    source = nb.cells[11].source.replace("RUN_TRAINING = False", "RUN_TRAINING = True")
    exec(source, context)
    assert "--recover-from-checkpoint" in calls[-1] and "--resume-from-checkpoint" not in calls[-1]
    checkpoint = context["checkpoint_root"] / "checkpoint-60"
    write_json(checkpoint / "dapo_checkpoint.json", {"global_step": 60})
    exec(source, context)
    assert calls[-1][-2:] == ["--resume-from-checkpoint", str(checkpoint)]
    assert "--recover-from-checkpoint" in calls[-1]
    write_json(context["checkpoint_root"] / "checkpoint-300/dapo_checkpoint.json", {"global_step": 300})
    count = len(calls)
    exec(source, context)
    assert len(calls) == count


def test_qwen_source_template_tokens_and_unchanged_hyperparameters():
    from test_llama_dapo import NativeTokenFixture

    class QwenTokens(NativeTokenFixture):
        vocabulary = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643}

    cfg = load_config(CONFIG)
    source, identity = select_model(cfg, "base", "/missing")
    assert source == "Qwen/Qwen2.5-3B-Instruct" and not identity["trust_remote_code"]
    assert identity["revision"] == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    with pytest.raises(ValueError, match="no SFT source"):
        select_model(cfg, "sft", "/missing")
    tok = configure_tokenizer(QwenTokens(), cfg)
    assert tok.eos_token_id == 151645 and tok.pad_token_id == 151643
    assert stop_token_ids(tok, cfg) == [151645]
    training_prompt(tok, "Compute 2+2", cfg)
    messages, kwargs = tok.received
    assert messages[0]["content"].startswith("Compute 2+2\n\nSolve the problem")
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}
    previous = load_config(ROOT / "configs/dapo_minibatch.yaml")
    for key in ["data", "hardware", "algorithm", "training", "rollout", "smoke"]:
        assert cfg[key] == previous[key]


def test_bundled_setup_isolated_imports_and_reproducible_resume_identity(tmp_path):
    nb = nbformat.read(ROOT / "notebooks/train_dapo_qwen25_3b.ipynb", as_version=4)
    # Execute real extraction and Git snapshot, excluding pip/CUDA installation.
    setup = nb.cells[3].source.split('subprocess.check_call([sys.executable')[0]
    (tmp_path / "setup.txt").write_text(setup)
    script = '''
import json, sys
from pathlib import Path
from types import SimpleNamespace
root = Path(sys.argv[1])
sys.modules["google.colab"] = SimpleNamespace(
    drive=SimpleNamespace(mount=lambda path: None), userdata=SimpleNamespace(get=lambda key: "fake-token"))
commits = []
for folder in ["first", "first", "second"]:
    namespace = {"CODE_PARENT": str(root / folder)}
    exec((root / "setup.txt").read_text(), namespace)
    commits.append(namespace["git"]("rev-parse", "HEAD"))
assert len(set(commits)) == 1
code = Path(namespace["CODE_ROOT"])
sys.path.insert(0, str(code))
from train import run_dapo
from eval.run_english_eval import git_identity
assert run_dapo.ROOT == code and git_identity() == commits[0]
cfg = run_dapo.load_config(code / "configs/dapo_qwen25_3b.yaml")
assert cfg["training"]["max_steps"] == 300
assert run_dapo.select_model(cfg, "base", "")[0] == "Qwen/Qwen2.5-3B-Instruct"
for lock in code.glob("requirements*.lock"):
    for line in lock.read_text().splitlines():
        if line.startswith("-r "):
            assert (code / line[3:]).is_file()
print(json.dumps({"model": cfg["model"]["repo"], "commit": commits[0]}))
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["model"] == "Qwen/Qwen2.5-3B-Instruct"
