import json
import struct
from copy import deepcopy
from pathlib import Path

import pytest

from common.io import write_json
from train.dapo_data import load_config
from train.dapo_resume import check_checkpoint, check_extension_runtime, extension_source

ROOT = Path(__file__).resolve().parents[1]


def parent_run(root):
    config = load_config(ROOT / "configs/dapo.yaml")
    config["model_kind"] = "base"
    config["data"]["revision"] = "a" * 40
    checkpoint = root / "checkpoints/dapo_exaone_base/checkpoint-100"
    write_json(checkpoint / "dapo_checkpoint.json", {"run_identity": "parent", "global_step": 100})
    write_json(checkpoint / "trainer_state.json", {"global_step": 100, "max_steps": 100})
    for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json", "tokenizer_config.json"]:
        (checkpoint / name).write_text("fixture")
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (checkpoint / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    manifest = {"identity": "parent", "config": config, "git_commit": "original-commit",
                "source": {"kind": "base"}, "data_report_digest": "data", "packages": {"torch": "pinned"},
                "parameter_precision": "float32", "compute_precision": "bfloat16"}
    write_json(root / "logs/dapo_exaone_base/run_manifest.json", manifest)
    return checkpoint, manifest


def test_only_total_step_extension_allowed_and_parent_preserved(tmp_path):
    checkpoint, old = parent_run(tmp_path / "original")
    config = load_config(ROOT / "configs/dapo_continue_300.yaml")
    config["model_kind"] = "base"
    previous, info = extension_source(config, checkpoint, tmp_path / "continuation")
    assert previous == old and info["from_step"] == 100 and info["target_step"] == 300
    assert config["data"]["revision"] == old["config"]["data"]["revision"]
    assert config["training"]["warmup_steps"] == 20
    for section, key, value in [("training", "learning_rate", 2e-6), ("algorithm", "group_size", 4),
                                ("training", "max_steps", 100)]:
        bad = deepcopy(config)
        bad[section][key] = value
        with pytest.raises(ValueError, match="only increase"):
            extension_source(bad, checkpoint, tmp_path / "continuation")
    with pytest.raises(ValueError, match="separate OUTPUT_ROOT"):
        extension_source(config, checkpoint, tmp_path / "original")
    assert json.loads((tmp_path / "original/logs/dapo_exaone_base/run_manifest.json").read_text()) == old
    current = deepcopy(old)
    current["git_commit"] = "new-continuation-code"
    check_extension_runtime(old, current)
    current["packages"]["torch"] = "different"
    with pytest.raises(ValueError, match="packages"):
        check_extension_runtime(old, current)


def test_checkpoint_rejects_partial_weights_and_missing_optimizer(tmp_path):
    checkpoint, _ = parent_run(tmp_path / "original")
    assert check_checkpoint(checkpoint)[1]["global_step"] == 100
    shard = checkpoint / "model.safetensors"
    complete = shard.read_bytes()
    shard.write_bytes(complete[:-1])
    with pytest.raises(ValueError, match="Incomplete model shard data"):
        check_checkpoint(checkpoint)
    shard.write_bytes(complete)
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(FileNotFoundError, match="optimizer.pt"):
        check_checkpoint(checkpoint)


def test_continuation_notebook_commands_and_auto_resume(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_dapo_continuation_notebook import cells
    nb = nbformat.read(ROOT / "notebooks/continue_dapo_exaone_100_to_300.ipynb", as_version=4)
    nbformat.validate(nb)
    assert [c.source for c in nb.cells] == [c.source for c in cells()]
    sources = [c.source for c in nb.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "continuation_notebook", "exec")
        assert "@param" not in source
    original = tmp_path / "original"
    checkpoint, _ = parent_run(original)
    context = {"Path": Path, "json": json}
    exec(sources[0], context)
    context.update(CODE_ROOT=str(ROOT), ORIGINAL_ROOT=str(original), OUTPUT_ROOT=str(tmp_path / "continued"))
    exec(next(s for s in sources if "TRAIN_COMMAND =" in s), context)
    calls = []
    context["run_logged"] = lambda command, **kwargs: calls.append(command)
    train_cell = next(s for s in sources if "RUN_TRAINING = False" in s)
    exec(train_cell, context)
    assert not calls
    exec(train_cell.replace("RUN_TRAINING = False", "RUN_TRAINING = True"), context)
    assert "--resume-from-checkpoint" not in calls[-1]
    assert calls[-1][calls[-1].index("--extend-from-checkpoint") + 1] == str(checkpoint)
    latest = context["checkpoint_root"] / "checkpoint-120"
    write_json(latest / "dapo_checkpoint.json", {"global_step": 120})
    exec(train_cell.replace("RUN_TRAINING = False", "RUN_TRAINING = True"), context)
    assert calls[-1][calls[-1].index("--resume-from-checkpoint") + 1] == str(latest)
    write_json(context["checkpoint_root"] / "checkpoint-300/dapo_checkpoint.json", {"global_step": 300})
    count = len(calls)
    exec(train_cell.replace("RUN_TRAINING = False", "RUN_TRAINING = True"), context)
    assert len(calls) == count
