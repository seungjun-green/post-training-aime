from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from test_batched_eval import fixture, install_core
from eval.batched_generation import generate_problems
from eval.budget_generation import validate_budget
from eval.profiles import load_profile
from eval.run_batched_eval import evaluate_batched


def setup(monkeypatch, mode="forced", answer_cap=False):
    engine, datasets, config, execution = fixture()
    config["datasets"]["test"]["n"] = 1
    config["pass_k"] = [1]
    config["max_new_tokens"] = 128
    config["budget_forcing"] = {
        "min_reasoning_tokens": 0, "max_reasoning_tokens": 64, "answer_budget_tokens": 64,
        "end_thinking": "</think>", "answer_prefix": "\n\nFinal Answer:",
    }
    engine.tokenizer.encode = lambda text, **kwargs: list(map(ord, text))
    core = install_core(monkeypatch, engine)

    def step():
        core.step_sizes.append(len(core.pending))
        key = list(core.pending)[-1]
        params = core.pending.pop(key)
        if key.endswith("-answer"):
            assert not hasattr(params, "stop")
            text = "a" * params.max_tokens if answer_cap else r"\boxed{42}"
            reason, stop_reason = ("length" if answer_cap else "stop"), None
        elif mode == "forced":
            text, reason, stop_reason = "<think>" + "x" * 57, "length", None
        elif mode == "natural":
            text, reason, stop_reason = "<think>done</think>", "stop", "</think>"
        else:
            text, reason, stop_reason = r"\boxed{42}", "stop", None
        return [SimpleNamespace(request_id=key, finished=True, outputs=[SimpleNamespace(
            index=0, text=text, token_ids=list(map(ord, text)), finish_reason=reason,
            stop_reason=stop_reason)])]

    core.step = step
    return engine, datasets, config, execution, core


@pytest.mark.parametrize("mode", ["forced", "natural", "eos"])
def test_two_phase_transition_preserves_ids_sampling_and_budget(monkeypatch, mode):
    engine, _, config, execution, core = setup(monkeypatch, mode)
    jobs = [{"key": i, "problem": str(i), "n": 1, "seed": i + 100} for i in range(3)]
    results = list(generate_problems(engine, jobs, execution))
    assert len(results) == 3 and len(core.calls) == (3 if mode == "eos" else 6)
    assert max(core.step_sizes) == 2  # continuous batching survives the phase switch
    assert not core.pending
    for job, prompt, responses in results:
        response = responses[0]
        details = response["budget_forcing"]
        assert details["forced"] == (mode == "forced")
        assert response["text"].endswith(r"\boxed{42}")
        assert response["token_count"] <= 128
        if mode == "eos":
            assert details["answer_tokens"] == details["injected_tokens"] == 0
        else:
            assert response["text"].count("</think>") == 1
            assert details["injected_tokens"] > 0
            assert details["answer_tokens"] == len(r"\boxed{42}")
    by_id = {key: (prompt, params) for key, prompt, params in core.calls}
    for key, prompt, params in core.calls:
        assert params.temperature == config["temperature"] and params.top_p == config["top_p"]
        assert params.n == 1 and not hasattr(params, "min_tokens")
        if key.endswith("-answer"):
            parent_prompt, parent_params = by_id[key.removesuffix("-answer")]
            assert params.seed == parent_params.seed
            original_ids = parent_prompt["prompt_token_ids"]
            assert prompt["prompt_token_ids"][:len(original_ids)] == original_ids
            assert len(prompt["prompt_token_ids"]) - len(original_ids) + params.max_tokens <= 128
        else:
            assert params.max_tokens == 64 and params.stop == ["</think>"]
            assert params.include_stop_str_in_output is True


def test_total_cap_includes_injected_text(monkeypatch):
    engine, _, _, execution, _ = setup(monkeypatch, answer_cap=True)
    output = list(generate_problems(engine, [{"key": 0, "problem": "Q", "n": 1, "seed": 1}], execution))[0][2][0]
    assert output["token_count"] == 128
    assert output["finish_reason"] == "length"
    assert output["budget_forcing"]["answer_tokens"] < 64


def test_budget_records_scored_and_resumed_after_answer(tmp_path, monkeypatch):
    engine, datasets, config, execution, core = setup(monkeypatch)
    journal = tmp_path / "problems.jsonl"
    records, metrics = evaluate_batched(engine, datasets, config, execution, journal)
    assert len(records) == 3 and metrics["test"]["avg@1"] == 1
    assert all(row["budget_forcing"]["forced"] for row in records)
    assert len(core.calls) == 6
    assert evaluate_batched(engine, datasets, config, execution, journal) == (records, metrics)
    assert len(core.calls) == 6


def test_cancel_cleans_pending_reasoning_and_answer_requests(monkeypatch):
    engine, _, _, execution, core = setup(monkeypatch)
    stream = generate_problems(engine, [{"key": i, "problem": str(i), "n": 1, "seed": i}
                                       for i in range(3)], execution)
    next(stream)
    stream.close()
    assert core.aborted and not core.pending


def test_engine_failure_cancels_pending_answer_phase(monkeypatch):
    engine, _, _, execution, core = setup(monkeypatch)
    step = core.step

    def fail_during_answer():
        if any(key.endswith("-answer") for key in core.pending):
            raise RuntimeError("interrupted answer generation")
        return step()

    core.step = fail_during_answer
    with pytest.raises(RuntimeError, match="interrupted answer"):
        list(generate_problems(engine, [{"key": i, "problem": str(i), "n": 1, "seed": i}
                                       for i in range(2)], execution))
    assert any(key.endswith("-answer") for key in core.aborted)
    assert not core.pending


@pytest.mark.parametrize("key,value", [("min_reasoning_tokens", 1),
    ("max_reasoning_tokens", 129), ("answer_budget_tokens", 0), ("end_thinking", "")])
def test_invalid_budget_rejected(monkeypatch, key, value):
    _, _, config, _, _ = setup(monkeypatch)
    config["budget_forcing"][key] = value
    with pytest.raises(ValueError):
        validate_budget(config)


def test_production_profile_has_agreed_budgets_and_one_sample():
    config = yaml.safe_load(Path("configs/eval_english.yaml").read_text())
    execution = yaml.safe_load(Path("configs/eval_execution.yaml").read_text())
    original = deepcopy(config)
    result, _ = load_profile("sample1_budget16k", config, execution)
    assert config == original
    assert result["budget_forcing"]["max_reasoning_tokens"] == 16384
    assert result["budget_forcing"]["answer_budget_tokens"] == 4096
    assert result["budget_forcing"]["min_reasoning_tokens"] == 0
    assert result["max_new_tokens"] == 20480
    assert result["temperature"] == 1 and result["top_p"] == 0.7
    assert all(spec["n"] == 1 for spec in result["datasets"].values())
    ordinary, _ = load_profile("sample1", result, execution)
    assert "budget_forcing" not in ordinary


def test_notebook_smoke_runs_only_final_checkpoint_and_all_five(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from build_stage1_notebook import budget_eval_cells

    source = next(cell.source for cell in budget_eval_cells()
                  if cell.cell_type == "code" and "RUN_SMOKE_EVAL = False" in cell.source)
    calls = []
    context = {"Path": Path, "eval_runs": [(5, tmp_path / "epoch_5", "sft_s1k")],
               "cfg": {"stage": "stage1"}, "EVAL_ENV": "/env", "CODE_ROOT": str(tmp_path),
               "TRAIN_ROOT": str(tmp_path), "BASELINE_ROOT": str(tmp_path / "results"),
               "EVAL_PROFILE": "sample1_budget", "EVAL_LABEL": "sample1_budget_thinking_18432_answer_2048",
               "BUDGET_ARGS": ["--budget_config", "/drive/budget.yaml"],
               "evaluation_command": lambda checkpoint, run_name, smoke=False: [
                   "python", "--model", str(checkpoint), "--profile", "sample1_budget",
                   "--budget_config", "/drive/budget.yaml"] + (["--smoke"] if smoke else []),
               "progress_totals": {k: 30 for k in ["aime_2024", "aime_2025", "aime_2026", "amc23", "math_500"]},
               "run_logged": lambda command, **kwargs: calls.append((command, kwargs))}
    exec(source, context)
    assert not calls
    exec(source.replace("RUN_SMOKE_EVAL = False", "RUN_SMOKE_EVAL = True"), context)
    command, kwargs = calls[0]
    assert len(calls) == 1 and "--smoke" in command
    assert command[command.index("--model") + 1] == str(tmp_path / "epoch_5")
    assert command[command.index("--profile") + 1] == "sample1_budget"
    assert command[command.index("--budget_config") + 1] == "/drive/budget.yaml"
    assert sum(kwargs["progress_totals"].values()) == 5


@pytest.mark.parametrize("maximum", [8192, 16384, 18432])
def test_custom_budget_configuration_and_output_isolation(maximum):
    from eval.profiles import profile_root

    config = yaml.safe_load(Path("configs/eval_english.yaml").read_text())
    execution = yaml.safe_load(Path("configs/eval_execution.yaml").read_text())
    profile, _ = load_profile("sample1_budget", config, execution)
    budget = deepcopy(profile["budget_forcing"])
    budget.update(max_reasoning_tokens=maximum, answer_budget_tokens=20480 - maximum)
    custom, _ = load_profile("sample1_budget", config, execution, budget_override=budget)
    assert custom["budget_forcing"] == budget
    assert profile_root("/out", "sample1_budget", budget) == Path(
        f"/out/profiles/sample1_budget/thinking_{maximum}_answer_{20480 - maximum}")
    with pytest.raises(ValueError, match="require"):
        load_profile("sample1", config, execution, budget_override=budget)


@pytest.mark.parametrize("maximum", [0, -1, 20480, 1.5])
def test_notebook_rejects_invalid_custom_cap_before_reading_checkpoint(monkeypatch, maximum, tmp_path):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from build_stage1_notebook import budget_eval_cells

    source = next(cell.source for cell in budget_eval_cells()
                  if cell.cell_type == "code" and "run_identities = set()" in cell.source)
    with pytest.raises(ValueError, match="MAX_THINKING_TOKENS"):
        exec(source, {"Path": Path, "CODE_ROOT": str(Path.cwd()), "EVAL_PROFILE": "sample1_budget",
                      "MAX_THINKING_TOKENS": maximum, "TRAIN_ROOT": str(tmp_path),
                      "MODEL_KIND": "sft", "TEMPERATURE": 1.0, "CONFIG": "configs/stage1_sft.yaml"})
