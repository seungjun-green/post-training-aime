import json
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


def notebook_context(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_stage1_notebook import pro_eval_cells

    notebook = nbformat.read(ROOT / "notebooks/evaluate_stage1_sft_deepseek_pro.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in pro_eval_cells()]
    sources = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "pro_eval_notebook", "exec")
    context = {"Path": Path, "json": json}
    exec(sources[0], context)
    context.update(CODE_ROOT=str(ROOT), DRIVE_ROOT=str(tmp_path))
    locate = next(s for s in sources if "checkpoint_relative =" in s)
    return sources, context, locate


def checkpoint(root, run="sft_s1k_deepseek_pro"):
    path = root / "checkpoints/stage1" / run / "epoch_5"
    path.mkdir(parents=True)
    (path / "stage1_checkpoint.json").write_text(json.dumps({"epoch": 5, "run_identity": "pro-run"}))
    return path


def test_pro_epoch5_selection_and_both_commands(tmp_path, monkeypatch):
    sources, context, locate = notebook_context(tmp_path, monkeypatch)
    checkpoint(tmp_path / "LG-AIME-Stage1", run="sft_s1k")
    training = tmp_path / "LG-AIME-Stage1-Pro-20261002-120000"
    expected_checkpoint = checkpoint(training)
    exec(locate, context)
    assert context["TRAIN_ROOT"] == str(training)
    assert context["CONFIG"] == "configs/stage1_sft_deepseek_pro.yaml"
    exec(next(s for s in sources if "run_identities = set()" in s), context)
    assert context["eval_runs"] == [(5, expected_checkpoint, "sft_s1k_deepseek_pro")]
    assert context["result_root"].is_relative_to(training / "evaluation_deepseek_pro")
    assert context["sampling"] == {"temperature": 0.0, "top_p": 1.0}
    assert context["budget"]["max_reasoning_tokens"] == 16384
    assert context["budget"]["answer_budget_tokens"] == 4096
    assert all(d["n"] == 1 for d in context["resolved_config"]["datasets"].values())
    calls = []
    context["run_logged"] = lambda command, **kwargs: calls.append((command, kwargs))
    for flag, count in [("RUN_SMOKE_EVAL", 5), ("RUN_STAGE1_EVAL", 630)]:
        source = next(s for s in sources if f"{flag} = False" in s)
        before = len(calls)
        exec(source, context)
        assert len(calls) == before
        exec(source.replace(f"{flag} = False", f"{flag} = True"), context)
        command, kwargs = calls[-1]
        assert command[command.index("--model") + 1] == str(expected_checkpoint)
        assert command[command.index("--run_name") + 1] == "sft_s1k_deepseek_pro"
        assert "--budget_config" in command
        assert ("--smoke" in command) == (count == 5)
        assert sum(kwargs["progress_totals"].values()) == count


def test_pro_discovery_requires_unique_or_explicit_selection(tmp_path, monkeypatch):
    _, context, locate = notebook_context(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="No unique"):
        exec(locate, context)
    first = tmp_path / "LG-AIME-Stage1"
    second = tmp_path / "LG-AIME-Stage1-Pro-another"
    checkpoint(first)
    checkpoint(second)
    with pytest.raises(ValueError, match="No unique"):
        exec(locate, context)
    context["TRAIN_ROOT"] = str(second)
    exec(locate, context)
    assert context["BASELINE_ROOT"] == str(second / "evaluation_deepseek_pro")
