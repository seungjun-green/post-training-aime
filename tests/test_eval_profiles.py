import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from test_batched_eval import fixture, install_core

from common.io import read_jsonl
from eval import run_batched_eval as runner
from eval.profiles import (
    bind_profile_protocol,
    bind_profile_runtime,
    load_profile,
    profile_root,
    validate_profile_config,
)


@pytest.mark.parametrize("profile,total,temperature,math_n,queue", [
    ("greedy", 630, 0.0, 1, 64), ("sample8", 3040, 1.0, 4, 16),
])
def test_profiles_only_change_requested_sampling_and_queue(profile, total, temperature, math_n, queue):
    original = yaml.safe_load(Path("configs/eval_english.yaml").read_text())
    original_execution = yaml.safe_load(Path("configs/eval_execution.yaml").read_text())
    before = deepcopy(original)
    config, execution = load_profile(profile, original, original_execution)
    assert original == before
    assert config["temperature"] == temperature
    assert config["datasets"]["math_500"]["n"] == math_n
    assert sum(s["source_rows"] * s["n"] for s in config["datasets"].values()) == total
    assert config["pass_k"] == ([1] if profile == "greedy" else [1, 4, 8])
    assert execution["max_pending_problems"] == queue
    assert config["max_new_tokens"] == 20480
    for key in original:
        if key not in {"temperature", "top_p", "pass_k", "datasets"}:
            assert config[key] == original[key]
    for name, spec in original["datasets"].items():
        assert {k: v for k, v in config["datasets"][name].items() if k != "n"} == {
            k: v for k, v in spec.items() if k != "n"
        }


def test_greedy_sampler_receives_one_answer_and_temperature_zero(monkeypatch):
    from eval.batched_generation import generate_problems

    engine, _, config, execution = fixture()
    config["temperature"], config["top_p"], config["pass_k"] = 0.0, 1.0, [1]
    config["datasets"]["test"]["n"] = 1
    validate_profile_config(config)
    core = install_core(monkeypatch, engine)
    outputs = list(generate_problems(engine, [{"key": 0, "problem": "Q", "n": 1, "seed": 42}], execution))
    assert core.calls[0][2].temperature == 0.0
    assert core.calls[0][2].n == 1
    assert len(outputs[0][2]) == 1


@pytest.mark.parametrize("temperature,n,ks", [
    (-1, 1, [1]), (float("nan"), 1, [1]), (0, 8, [1]), (1, 0, [1]), (1, 4, [8]),
])
def test_invalid_profiles_rejected(temperature, n, ks):
    _, _, config, _ = fixture()
    config.update(temperature=temperature, pass_k=ks)
    config["datasets"]["test"]["n"] = n
    with pytest.raises(ValueError):
        validate_profile_config(config)


@pytest.mark.parametrize("profile", ["greedy", "sample8"])
@pytest.mark.parametrize("first_stage", ["stage0", "stage1"])
def test_either_stage_first_smoke_full_resume_and_profile_isolation(tmp_path, monkeypatch, profile, first_stage):
    from eval import engines, english_engines

    engine, _, _, _ = fixture()
    core = install_core(monkeypatch, engine)
    original = yaml.safe_load(Path("configs/eval_english.yaml").read_text())
    datasets = {name: [
        {"id": i, spec["problem_column"]: f"Q {i}", spec["answer_column"]: "42"}
        for i in range(2)
    ] for name, spec in original["datasets"].items()}
    monkeypatch.setattr(runner, "load_eval_sets", lambda *args: deepcopy(datasets))
    monkeypatch.setattr(runner, "git_identity", lambda: "same-commit")
    monkeypatch.setattr(runner, "resolve_model", lambda model, *args: (None, model))
    monkeypatch.setattr(engines, "check_hardware", lambda *args: {"name": "gpu"})

    def create(model, config, revision):
        engine.config = config
        return engine, None

    monkeypatch.setattr(english_engines, "create_engine", create)
    old = tmp_path / "full/results/stage0/baseline_english.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"original": "untouched"}')

    def run(stage, smoke=False, selected_profile=profile, validate=False):
        monkeypatch.setattr(sys, "argv", [
            "eval.run_batched_eval", "--model", stage, "--stage", stage,
            "--run_name", "baseline_english" if stage == "stage0" else "sft_s1k",
            "--profile", selected_profile, "--output_root", str(tmp_path),
        ] + (["--smoke"] if smoke else []) + (["--validate-only"] if validate else []))
        runner.main()

    root = profile_root(tmp_path, profile)
    run(first_stage, validate=True)
    assert not root.exists()  # validation is read-only for either stage
    run("stage0", smoke=True)
    smoke = json.loads((root / "smoke/results/stage0/baseline_english.json").read_text())
    assert smoke["mode"] == "smoke"
    assert all(m["responses"] == 1 for m in smoke["metrics"].values())
    assert (root / "smoke/archives/stage0/baseline_english/smoke_test_result.zip").exists()
    assert not (root / "full").exists()
    run(first_stage)
    manifests = {p: p.read_bytes() for p in (root / "full/results").glob("*.json")}
    run("stage1" if first_stage == "stage0" else "stage0")
    assert all(path.read_bytes() == content for path, content in manifests.items())
    baseline = json.loads((root / "full/results/stage0/baseline_english.json").read_text())
    result_path = root / "full/results/stage1/sft_s1k.json"
    sft = json.loads(result_path.read_text())
    assert sft["config"] == baseline["config"]
    assert sft["protocol_digest"] == baseline["protocol_digest"]
    assert sft["profile"] == baseline["profile"] == profile
    assert sft["metrics"] == baseline["metrics"]
    if profile == "sample8":
        assert set(sft["metrics"]["aime_2024"]["pass@k"]) == {"1", "4", "8"}
        assert set(sft["metrics"]["math_500"]["pass@k"]) == {"1", "4"}
        assert sum(m["responses"] for m in sft["metrics"].values()) == 72
    else:
        assert all(set(m["pass@k"]) == {"1"} for m in sft["metrics"].values())
        assert len(read_jsonl(result_path.with_name("sft_s1k_generations.jsonl"))) == 10
    calls = len(core.calls)
    run("stage1")
    assert len(core.calls) == calls
    other = "greedy" if profile == "sample8" else "sample8"
    run("stage1", selected_profile=other)
    assert (profile_root(tmp_path, other) / "full/results/stage1/sft_s1k.json").is_file()
    assert not (profile_root(tmp_path, other) / "full/results/stage0").exists()
    assert old.read_text() == '{"original": "untouched"}'


def test_profiles_reject_legacy_import_and_unknown_names(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "eval.run_batched_eval", "--model", "checkpoint", "--run_name", "new",
        "--reuse_run_name", "old", "--profile", "greedy", "--output_root", str(tmp_path),
    ])
    with pytest.raises(SystemExit):
        runner.main()
    with pytest.raises(ValueError, match="Unknown"):
        profile_root(tmp_path, "../old")


def test_sft_first_still_rejects_later_protocol_and_runtime_changes(tmp_path):
    engine, _, config, execution = fixture()
    first = bind_profile_protocol(
        tmp_path, "sample8", config, execution, {}, "stage1", {"test": [0]}, False, "code",
    )
    assert bind_profile_protocol(
        tmp_path, "sample8", config, execution, {}, "stage0", {"test": [0]}, False, "code",
    ) == first
    bind_profile_runtime(tmp_path, engine.tokenizer)
    saved = {p: p.read_bytes() for p in (tmp_path / "results").glob("*.json")}
    changed = deepcopy(config)
    changed["temperature"] = 0.5
    with pytest.raises(ValueError, match="protocol changed"):
        bind_profile_protocol(
            tmp_path, "sample8", changed, execution, {}, "stage0", {"test": [0]}, False, "code",
        )
    engine.tokenizer.chat_template = "different"
    with pytest.raises(ValueError, match="Tokenizer"):
        bind_profile_runtime(tmp_path, engine.tokenizer)
    assert all(path.read_bytes() == content for path, content in saved.items())


@pytest.mark.parametrize("profile", ["greedy", "sample8"])
def test_baseline_notebook_passes_profile_and_keeps_default_full_run_off(tmp_path, monkeypatch, profile):
    import nbformat

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from build_notebooks import code, markdown
    from english_eval_notebook import english_eval_cells

    notebook = nbformat.read("notebooks/evaluate_baseline_english.ipynb", as_version=4)
    assert [c.source for c in notebook.cells] == [c.source for c in english_eval_cells(markdown, code)]
    sources = [c.source for c in notebook.cells if c.cell_type == "code"]
    setup = next(s for s in sources if "EVAL_COMMAND =" in s)
    calls = []
    from types import SimpleNamespace

    context = {
        "Path": Path, "sys": sys, "subprocess": SimpleNamespace(check_call=lambda *a, **k: None),
        "CODE_ROOT": str(Path.cwd()), "GPU_ENV": "/env", "MODEL": "base", "EVAL_PROFILE": profile,
        "OUTPUT_ROOT": str(tmp_path),
    }
    exec(setup, context)
    command = context["EVAL_COMMAND"]
    assert command[command.index("--profile") + 1] == profile
    assert command[command.index("--stage") + 1] == "stage0"
    assert context["PROFILE_ROOT"] == tmp_path / "profiles" / profile
    context["run_logged"] = lambda *a, **k: calls.append((a, k))
    full = next(s for s in sources if "RUN_FULL_EVAL = False" in s)
    exec(full, context)
    assert not calls
    exec(full.replace("RUN_FULL_EVAL = False", "RUN_FULL_EVAL = True"), context)
    assert len(calls) == 1 and calls[0][0][0] == command
