import json
from pathlib import Path

import pytest
import yaml

from eval.profiles import load_profile, profile_root


@pytest.mark.parametrize("model_kind", ["base", "sft"])
@pytest.mark.parametrize("temperature", [0.0, 0.6, 1.0])
def test_actual_notebook_model_temperature_smoke_and_full(tmp_path, monkeypatch, model_kind, temperature):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from build_stage1_notebook import budget_eval_cells

    cells = [cell.source for cell in budget_eval_cells() if cell.cell_type == "code"]
    settings = next(cell for cell in cells if 'MODEL_KIND = "sft"' in cell)
    prepare = next(cell for cell in cells if "run_identities = set()" in cell)
    smoke = next(cell for cell in cells if "RUN_SMOKE_EVAL = False" in cell)
    full = next(cell for cell in cells if "RUN_STAGE1_EVAL = False" in cell)
    context = {"Path": Path, "json": json}
    exec(settings, context)
    assert context["CONFIG"] == "configs/stage1_sft.yaml"
    context.update(MODEL_KIND=model_kind, TEMPERATURE=temperature, CODE_ROOT=str(Path.cwd()),
                   TRAIN_ROOT=str(tmp_path / "train"), BASELINE_ROOT=str(tmp_path / "eval"),
                   MAX_THINKING_TOKENS=8192 if temperature == 0.6 else 18432)
    checkpoint = tmp_path / "train/checkpoints/stage1/sft_s1k/epoch_5"
    if model_kind == "sft":
        checkpoint.mkdir(parents=True)
        (checkpoint / "stage1_checkpoint.json").write_text(json.dumps({"epoch": 5, "run_identity": "old-sft"}))
    # Base evaluation must work without any checkpoint directory or a valid thinking cap.
    if model_kind == "base":
        context["MAX_THINKING_TOKENS"] = -1
    calls = []
    context["run_logged"] = lambda command, **kwargs: calls.append((command, kwargs))
    exec(prepare, context)
    exec(smoke, context)
    exec(full, context)
    assert not calls
    exec(smoke.replace("RUN_SMOKE_EVAL = False", "RUN_SMOKE_EVAL = True"), context)
    exec(full.replace("RUN_STAGE1_EVAL = False", "RUN_STAGE1_EVAL = True"), context)
    assert len(calls) == 2
    for index, (command, kwargs) in enumerate(calls):
        assert ("--smoke" in command) == (index == 0)
        assert sum(kwargs["progress_totals"].values()) == (5 if index == 0 else 630)
        assert f"temperature_{float(temperature)}" in str(kwargs["log_path"])
        sampling = yaml.safe_load(Path(command[command.index("--sampling_config") + 1]).read_text())
        assert sampling == {"temperature": temperature, "top_p": 1.0 if temperature == 0 else 0.7}
        assert f"temperature_{float(temperature)}" in str(context["result_root"])
        config = context["resolved_config"]
        assert config["temperature"] == temperature
        assert all(dataset["n"] == 1 for dataset in config["datasets"].values())
        if model_kind == "base":
            assert not checkpoint.exists()
            assert command[command.index("--model") + 1] == "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
            assert command[command.index("--stage") + 1] == "stage0"
            assert command[command.index("--revision") + 1] == context["cfg"]["model"]["revision"]
            assert command[command.index("--run_name") + 1] == "baseline_english"
            assert command[command.index("--profile") + 1] == "sample1"
            assert "--budget_config" not in command and "budget_forcing" not in config
        else:
            assert command[command.index("--model") + 1] == str(checkpoint)
            assert command[command.index("--stage") + 1] == "stage1"
            assert command[command.index("--run_name") + 1] == "sft_s1k"
            assert "--revision" not in command
            assert command[command.index("--profile") + 1] == "sample1_budget"
            budget = yaml.safe_load(Path(command[command.index("--budget_config") + 1]).read_text())
            assert budget["max_reasoning_tokens"] == context["MAX_THINKING_TOKENS"]
            assert budget["answer_budget_tokens"] == 20480 - context["MAX_THINKING_TOKENS"]


@pytest.mark.parametrize("temperature", [-1, float("nan"), float("inf"), "1", True])
def test_invalid_sampling_override_rejected(temperature):
    config = yaml.safe_load(Path("configs/eval_english.yaml").read_text())
    execution = yaml.safe_load(Path("configs/eval_execution.yaml").read_text())
    with pytest.raises(ValueError):
        load_profile("sample1", config, execution,
                     sampling_override={"temperature": temperature, "top_p": 0.7})


def test_temperature_roots_do_not_overlap():
    roots = {profile_root("/out", "sample1", sampling={"temperature": t, "top_p": 0.7})
             for t in [0.0, 0.5, 1.0, 1.5]}
    assert len(roots) == 4


@pytest.mark.parametrize("profile,temperature", [("sample1", 0.0), ("sample1_budget", 0.6)])
def test_cli_reads_sampling_file_and_binds_correct_destination(tmp_path, monkeypatch, profile, temperature):
    import sys
    from eval import run_batched_eval as runner

    sampling = {"temperature": temperature, "top_p": 1.0 if temperature == 0 else 0.7}
    sampling_file = tmp_path / "sampling.yaml"
    sampling_file.write_text(yaml.safe_dump(sampling))
    args = ["eval.run_batched_eval", "--model", "model", "--run_name", "test",
            "--output_root", str(tmp_path), "--profile", profile,
            "--sampling_config", str(sampling_file), "--validate-only"]
    budget = None
    if profile == "sample1_budget":
        budget = yaml.safe_load(Path("configs/eval_profiles.yaml").read_text())[profile]["budget_forcing"]
        budget.update(max_reasoning_tokens=8192, answer_budget_tokens=12288)
        budget_file = tmp_path / "budget.yaml"
        budget_file.write_text(yaml.safe_dump(budget))
        args += ["--budget_config", str(budget_file)]
    observed = {}
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner, "load_eval_sets", lambda *args: {"aime_2024": [{"id": 1}]})

    def bind(root, name, config, *args, **kwargs):
        observed.update(root=root, config=config)
        assert kwargs["create"] is False
        return "validated"

    monkeypatch.setattr(runner, "bind_profile_protocol", bind)
    runner.main()
    assert observed["root"] == profile_root(tmp_path, profile, budget, sampling) / "full"
    assert observed["config"]["temperature"] == temperature
    assert observed["config"].get("budget_forcing") == budget
