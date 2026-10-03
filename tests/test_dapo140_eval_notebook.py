import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint140_notebook_paths_subset_commands_and_table(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    from test_dapo_continuation import parent_run

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_dapo140_eval_notebook import cells
    notebook = nbformat.read(ROOT / "notebooks/evaluate_dapo_checkpoint140_amc_math.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells()]
    sources = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "checkpoint140_eval", "exec")
        assert "@param" not in source and "train/run_dapo.py" not in source
    old, _ = parent_run(tmp_path / "continuation")
    checkpoint = old.with_name("checkpoint-140")
    old.rename(checkpoint)
    write_json(checkpoint / "dapo_checkpoint.json",
               {"global_step": 140, "model_kind": "base", "run_identity": "parent"})
    for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]:
        (checkpoint / name).unlink()
    context = {"Path": Path, "json": json}
    exec(sources[0], context)
    assert context["RL_ROOT"].endswith("LG-AIME-DAPO-100to300")
    context.update(RL_ROOT=str(tmp_path / "continuation"), CODE_ROOT=str(ROOT), OUTPUT_ROOT=str(tmp_path / "eval"))
    prepare = next(s for s in sources if "def evaluation_command" in s)
    exec(prepare, context)
    assert context["checkpoint"] == checkpoint
    assert context["progress_totals"] == {"amc23": 40, "math_500": 500}
    config = context["resolved"]
    assert config["temperature"] == 0 and config["top_p"] == 1 and config["pass_k"] == [1]
    assert config["max_new_tokens"] == 20480 and "budget_forcing" not in config
    calls = []
    context["run_logged"] = lambda command, **kwargs: calls.append((command, kwargs))
    for switch in ["RUN_SMOKE", "RUN_EVAL"]:
        cell = next(s for s in sources if f"{switch} = False" in s)
        before = len(calls)
        exec(cell, context)
        assert len(calls) == before
        exec(cell.replace(f"{switch} = False", f"{switch} = True"), context)
    assert len(calls) == 2  # One smoke and one full call, with no baseline inference.
    smoke, full = calls
    assert "--smoke" in smoke[0] and "--smoke" not in full[0]
    assert sum(smoke[1]["progress_totals"].values()) == 2
    assert sum(full[1]["progress_totals"].values()) == 540
    for command, _ in calls:
        assert "eval.run_amc_math_eval" in command
        assert command[command.index("--model") + 1] == str(checkpoint)
        assert command[command.index("--run-name") + 1] == "dapo_checkpoint140"
    metrics = {name: {"problems": n, "responses": n, "samples_per_problem": 1,
                      "avg@1": 0.5, "response_length_tokens": {"all": 800.}}
               for name, n in context["progress_totals"].items()}
    result = {"model": str(checkpoint), "mode": "full", "config": config, "metrics": metrics}
    write_json(context["result_path"], result)
    # Only notebook display is stubbed; execute the real result validation and
    # table arithmetic without requiring the Colab UI packages in local tests.
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace(
        DataFrame=lambda rows: SimpleNamespace(style=SimpleNamespace(format=lambda _: rows))))
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(display=lambda _: None))
    context["show_results"]()
    table = json.loads((tmp_path / "eval/full/summary_table.json").read_text())
    assert [r["Correct"] for r in table["rows"]] == [20, 250]
    assert [r["DAPO-140 accuracy (%)"] for r in table["rows"]] == [50., 50.]
    result["metrics"]["amc23"]["problems"] = 1
    write_json(context["result_path"], result)
    with pytest.raises(ValueError, match="Incomplete"):
        context["show_results"]()
    write_json(checkpoint / "dapo_checkpoint.json",
               {"global_step": 140, "model_kind": "base", "run_identity": "wrong"})
    with pytest.raises(ValueError, match="checkpoint 140"):
        exec(prepare, context)
