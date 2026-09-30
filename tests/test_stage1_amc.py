import json
from pathlib import Path

import pytest
import yaml

from eval.run_english_eval import bind_protocol
from eval.run_stage1_amc import select_amc, validate_baseline

ROOT = Path(__file__).resolve().parents[1]


def test_amc_subset_reuses_config_without_mutating_full_protocol(tmp_path):
    config = yaml.safe_load((ROOT / "configs/eval_english.yaml").read_text())
    suite = json.loads((ROOT / "configs/english_eval_suite.json").read_text())
    original = json.dumps([config, suite], sort_keys=True)
    ids = {name: entry["ids"] for name, entry in suite["datasets"].items()}
    expected = bind_protocol(tmp_path, config, suite, "stage0", ids, False)
    protocol_path = tmp_path / "results/eval_protocol.json"
    frozen = protocol_path.read_bytes()
    assert validate_baseline(tmp_path, config, suite) == expected
    selected_config, selected_suite = select_amc(config, suite)
    assert list(selected_config["datasets"]) == ["amc23"]
    assert selected_config["datasets"]["amc23"] == config["datasets"]["amc23"]
    assert selected_suite["datasets"]["amc23"] == suite["datasets"]["amc23"]
    assert {k: v for k, v in selected_config.items() if k != "datasets"} == {
        k: v for k, v in config.items() if k != "datasets"
    }
    assert json.dumps([config, suite], sort_keys=True) == original
    assert protocol_path.read_bytes() == frozen
    config["temperature"] = 0.5
    with pytest.raises(ValueError, match="protocol changed"):
        validate_baseline(tmp_path, config, suite)


def test_amc_requires_existing_full_baseline(tmp_path):
    config = yaml.safe_load((ROOT / "configs/eval_english.yaml").read_text())
    suite = json.loads((ROOT / "configs/english_eval_suite.json").read_text())
    with pytest.raises(ValueError, match="baseline first"):
        validate_baseline(tmp_path, config, suite)
    assert not (tmp_path / "results").exists()


def test_amc_cli_uses_shared_scoring_and_resumes_without_regeneration(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from eval import engines, english_engines, run_stage1_amc
    from eval.run_eval import evaluate

    config = yaml.safe_load((ROOT / "configs/eval_english.yaml").read_text())
    suite = json.loads((ROOT / "configs/english_eval_suite.json").read_text())
    ids = {name: entry["ids"] for name, entry in suite["datasets"].items()}
    bind_protocol(tmp_path / "full", config, suite, "stage0", ids, False)
    frozen = (tmp_path / "full/results/eval_protocol.json").read_bytes()
    calls = []

    class Engine:
        name = "test-engine"
        tokenizer = SimpleNamespace(chat_template="unchanged")

        def generate(self, problem, n, seed):
            calls.append((problem, n, seed))
            return "shared prompt", [
                {"text": r"\boxed{4}", "token_count": 10, "finish_reason": "stop"}
                for _ in range(n)
            ]

    def load(selected_config, selected_suite, token):
        assert selected_config["datasets"] == {"amc23": config["datasets"]["amc23"]}
        return {"amc23": [{"id": 0, "question": "2+2?", "answer": "4"}]}

    monkeypatch.setattr(run_stage1_amc, "load_eval_sets", load)
    monkeypatch.setattr(run_stage1_amc, "git_identity", lambda: "test-commit")
    monkeypatch.setattr(run_stage1_amc, "resolve_model", lambda *a: (None, "checkpoint-digest"))
    monkeypatch.setattr(run_stage1_amc, "bind_runtime", lambda *a: None)
    monkeypatch.setattr(engines, "check_hardware", lambda config: {"test": True})
    monkeypatch.setattr(english_engines, "create_engine", lambda *a: (Engine(), None))
    monkeypatch.setattr(sys, "argv", [
        "run_stage1_amc", "--model", "local/checkpoint", "--run_name", "epoch1_amc",
        "--output_root", str(tmp_path),
    ])
    assert run_stage1_amc.evaluate is evaluate
    run_stage1_amc.main()
    run_stage1_amc.main()
    assert len(calls) == 1 and calls[0][1] == 32
    destination = tmp_path / "full/results/stage1/epoch1_amc.json"
    result = json.loads(destination.read_text())
    assert result["mode"] == "subset" and list(result["metrics"]) == ["amc23"]
    assert result["metrics"]["amc23"]["avg@32"] == 1.0
    assert (tmp_path / "full/results/eval_protocol.json").read_bytes() == frozen
