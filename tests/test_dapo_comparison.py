import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.io import write_json
from eval.dapo_comparison import comparison_rows
from eval.run_amc_math_eval import comparison_settings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dapo_compare_eval.yaml"


def test_subset_is_filtered_before_loading_and_shared_evaluator_resumes(tmp_path, monkeypatch):
    from test_batched_eval import Tokenizer, install_core

    from eval import engines, english_engines
    from eval import run_amc_math_eval as runner

    spec, config, _, suite = comparison_settings(CONFIG)
    assert list(config["datasets"]) == list(suite["datasets"]) == ["amc23", "math_500"]
    assert sum(s["rows"] for s in suite["datasets"].values()) == 540
    calls, cores = [], []
    def loader(config, suite, token):
        calls.append(list(config["datasets"]))
        return {name: [{"id": f"{name}-{i}", entry["problem_column"]: f"Question {i}",
                        entry["answer_column"]: "0"} for i in range(2)]
                for name, entry in config["datasets"].items()}
    def create(model, config, revision):
        engine = SimpleNamespace(name="vllm", config=config, tokenizer=Tokenizer())
        cores.append(install_core(monkeypatch, engine))
        return engine, None
    monkeypatch.setattr(runner, "load_eval_sets", loader)
    monkeypatch.setattr(runner, "resolve_model", lambda model, revision, token: (revision, "digest"))
    monkeypatch.setattr(runner, "git_identity", lambda: "commit")
    monkeypatch.setattr(runner, "package_versions", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(engines, "check_hardware", lambda config: {"gpu": "test"})
    monkeypatch.setattr(english_engines, "create_engine", create)
    monkeypatch.setattr(sys, "argv", ["eval", "--model", spec["base_model"]["repo"], "--revision",
                                      spec["base_model"]["revision"], "--run-name", "base",
                                      "--config", str(CONFIG), "--output-root", str(tmp_path), "--smoke"])
    runner.main()
    first = json.loads((tmp_path / "smoke/results/base.json").read_text())
    assert set(first["metrics"]) == {"amc23", "math_500"}
    assert all(m["responses"] == 1 and m["avg@1"] == 1 for m in first["metrics"].values())
    assert len(cores[0].calls) == 2
    assert all(p.temperature == 0 and p.n == 1 for _, _, p in cores[0].calls)
    runner.main()
    assert not cores[1].calls
    assert calls == [["amc23", "math_500"], ["amc23", "math_500"]]


def test_table_percentage_point_differences_and_protocol_checks():
    base = {key: "same" for key in ["config", "execution", "protocol_digest", "dataset_suite", "selected_ids",
                                    "packages", "hardware", "runner_digest", "chat_template_digest"]}
    base.update(mode="full", metrics={})
    for name, n, acc in [("amc23", 40, 0.325), ("math_500", 500, 0.662)]:
        base["metrics"][name] = {"problems": n, "responses": n, "samples_per_problem": 1,
                                  "avg@1": acc, "response_length_tokens": {"all": 700}}
    rl = deepcopy(base)
    rl["metrics"]["amc23"]["avg@1"] = .4
    rl["metrics"]["math_500"]["avg@1"] = .65
    rows = comparison_rows(base, rl)
    assert rows[0]["Difference (pp)"] == pytest.approx(7.5)
    assert rows[1]["Difference (pp)"] == pytest.approx(-1.2)
    rl["mode"] = "smoke"
    with pytest.raises(ValueError, match="full"):
        comparison_rows(base, rl)
    rl["mode"] = "full"
    rl["protocol_digest"] = "different"
    with pytest.raises(ValueError, match="protocol_digest"):
        comparison_rows(base, rl)


def test_comparison_notebook_runs_both_models_then_table(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    from test_dapo_continuation import parent_run

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from build_dapo_comparison_notebook import cells
    notebook = nbformat.read(ROOT / "notebooks/compare_base_dapo100_amc_math.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells] == [c.source for c in cells()]
    sources = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in sources:
        compile(source, "comparison_notebook", "exec")
        assert "@param" not in source
    checkpoint, _ = parent_run(tmp_path / "training")
    write_json(checkpoint / "dapo_checkpoint.json",
               {"global_step": 100, "model_kind": "base", "run_identity": "parent"})
    context = {"Path": Path, "json": json}
    exec(sources[0], context)
    context.update(CODE_ROOT=str(ROOT), RL_ROOT=str(tmp_path / "training"), OUTPUT_ROOT=str(tmp_path / "eval"))
    exec(next(s for s in sources if "def evaluation_command" in s), context)
    assert context["progress_totals"] == {"amc23": 40, "math_500": 500}
    assert context["eval_runs"][1][1] == str(checkpoint)
    calls, tables = [], []
    context["run_logged"] = lambda command, **kwargs: calls.append((command, kwargs))
    context["show_comparison"] = lambda: tables.append(True)
    for switch in ["RUN_SMOKE", "RUN_COMPARISON"]:
        cell = next(s for s in sources if f"{switch} = False" in s)
        n = len(calls)
        exec(cell, context)
        assert len(calls) == n
        exec(cell.replace(f"{switch} = False", f"{switch} = True"), context)
    assert len(calls) == 4 and tables == [True]
    for index, (command, kwargs) in enumerate(calls):
        assert "eval.run_amc_math_eval" in command
        assert ("--revision" in command) == (index % 2 == 0)
        assert ("--smoke" in command) == (index < 2)
        assert sum(kwargs["progress_totals"].values()) == (2 if index < 2 else 540)
