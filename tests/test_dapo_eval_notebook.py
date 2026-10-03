import json
from pathlib import Path

import pytest
import yaml

from common.io import write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_rl_checkpoint_eval_notebook_uses_existing_full_evaluator(tmp_path, monkeypatch, temperature):
    nbformat = pytest.importorskip("nbformat")
    from test_dapo_continuation import parent_run

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_dapo_eval_notebook import cells
    notebook = nbformat.read(ROOT / "notebooks/evaluate_dapo_checkpoint100.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells()]
    sources = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "rl_eval_notebook", "exec")
        assert "@param" not in source and "train/run_dapo.py" not in source
    checkpoint, _ = parent_run(tmp_path)
    write_json(checkpoint / "dapo_checkpoint.json",
               {"global_step": 100, "run_identity": "parent", "model_kind": "base"})
    # Evaluation needs only inference artifacts, not optimizer/RNG state.
    for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]:
        (checkpoint / name).unlink()
    context = {"Path": Path, "json": json}
    exec(sources[0], context)
    context.update(RL_ROOT=str(tmp_path), CODE_ROOT=str(ROOT), TEMPERATURE=temperature)
    prepare = next(s for s in sources if "def evaluation_command" in s)
    exec(prepare, context)
    assert context["checkpoint"] == checkpoint
    assert sum(context["progress_totals"].values()) == 630
    assert set(context["progress_totals"]) == {"aime_2024", "aime_2025", "aime_2026", "amc23", "math_500"}
    config = context["resolved"]
    assert config["temperature"] == temperature and config["pass_k"] == [1]
    assert config["max_new_tokens"] == 20480 and "budget_forcing" not in config
    assert all(d["n"] == 1 for d in config["datasets"].values())
    assert yaml.safe_load(context["sampling_path"].read_text())["temperature"] == temperature
    assert f"temperature_{temperature}_top_p_1.0" in str(context["result_path"])
    calls = []
    context["run_logged"] = lambda command, **kwargs: calls.append((command, kwargs))
    for switch in ["RUN_SMOKE_EVAL", "RUN_EVAL"]:
        cell = next(s for s in sources if f"{switch} = False" in s)
        count = len(calls)
        exec(cell, context)
        assert len(calls) == count
        exec(cell.replace(f"{switch} = False", f"{switch} = True"), context)
    smoke, full = calls
    assert "--smoke" in smoke[0] and "--smoke" not in full[0]
    assert sum(smoke[1]["progress_totals"].values()) == 5
    assert sum(full[1]["progress_totals"].values()) == 630
    for command, _ in calls:
        assert "eval.run_batched_eval" in command
        assert command[command.index("--model") + 1] == str(checkpoint)
        assert command[command.index("--stage") + 1] == "dapo"
        assert "--budget_config" not in command
    write_json(checkpoint / "dapo_checkpoint.json",
               {"global_step": 100, "run_identity": "wrong-run", "model_kind": "base"})
    with pytest.raises(ValueError, match="identity"):
        exec(prepare, context)
