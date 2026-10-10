import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from common.english_prompts import render_prompt
from common.io import append_jsonl, read_jsonl, write_json, write_jsonl
from eval import run_batched_eval as runner
from eval.batched_generation import generate_problems
from eval.run_eval import evaluate, problem_seed


class Tokenizer:
    chat_template = "native"
    special_tokens_map = {}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return messages[0]["content"] + "<assistant>"

    def encode(self, prompt, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [1, 2]

    def get_vocab(self):
        return {"test": 0}


def fixture():
    config = yaml.safe_load(Path("configs/eval_english.yaml").read_text())
    config["datasets"] = {"test": {
        "n": 4, "problem_column": "problem", "answer_column": "answer",
    }}
    datasets = {"test": [
        {"id": i, "problem": f"Question {i}", "answer": "42"} for i in range(3)
    ]}
    execution = {"version": "continuous-v1", "max_pending_problems": 2,
                 "status_interval_seconds": 30}
    engine = SimpleNamespace(name="vllm", config=config, tokenizer=Tokenizer())
    return engine, datasets, config, execution


def install_core(monkeypatch, engine, bad_indices=False):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(
        RequestOutputKind=SimpleNamespace(FINAL_ONLY="final"),
    ))

    class Core:
        def __init__(self):
            self.pending = {}
            self.calls = []
            self.step_sizes = []
            self.aborted = []

        def add_request(self, key, prompt, params):
            self.calls.append((key, prompt, params))
            self.pending[key] = params

        def step(self):
            self.step_sizes.append(len(self.pending))
            key = list(self.pending)[-1]  # deliberately finish out of submission order
            params = self.pending.pop(key)
            indices = [0] * params.n if bad_indices else reversed(range(params.n))
            return [SimpleNamespace(request_id=key, finished=True, outputs=[
                SimpleNamespace(index=i, text=rf"\boxed{{{42 if i % 2 else 0}}}",
                                token_ids=[3] * (i + 1), finish_reason="stop")
                for i in indices
            ])]

        def abort_request(self, ids):
            self.aborted.extend(ids)
            for key in ids:
                self.pending.pop(key, None)

    core = Core()
    engine.llm = SimpleNamespace(llm_engine=core)
    return core


def test_continuous_queue_refills_preserves_sampling_and_sample_order(monkeypatch):
    engine, _, config, execution = fixture()
    core = install_core(monkeypatch, engine)
    jobs = [{"key": i, "problem": str(i), "n": n, "seed": 123 + i}
            for i, n in enumerate([32, 4, 32, 4])]
    results = list(generate_problems(engine, jobs, execution))
    assert [r[0]["key"] for r in results] == [1, 2, 3, 0]
    assert core.step_sizes == [2, 2, 2, 1]
    assert not core.pending and not core.aborted
    for (key, prompt, params), job in zip(core.calls, jobs, strict=True):
        assert prompt == {"prompt_token_ids": [1, 2]}
        assert vars(params) == {
            "n": job["n"], "seed": job["seed"], "temperature": config["temperature"],
            "top_p": config["top_p"], "max_tokens": config["max_new_tokens"],
            "output_kind": "final",
        }
    for job, prompt, responses in results:
        assert prompt == render_prompt(engine.tokenizer, job["problem"])
        assert [r["token_count"] for r in responses] == list(range(1, job["n"] + 1))


def test_custom_instruct_prompt_and_per_problem_budget(monkeypatch):
    engine, _, _, execution = fixture()
    core = install_core(monkeypatch, engine)
    engine.config.update(max_model_len=8, max_new_tokens=7,
                         stop_token_ids=[151645, 151643])
    seen = []
    def renderer(tokenizer, problem):
        seen.append(problem)
        return 'custom instruct prompt'
    job = {'key': 0, 'problem': 'Question', 'n': 8, 'seed': 42, 'max_new_tokens': 6}
    result = list(generate_problems(engine, [job], execution, prompt_renderer=renderer))
    assert seen == ['Question'] and result[0][1] == 'custom instruct prompt'
    assert core.calls[0][2].max_tokens == 6
    assert core.calls[0][2].stop_token_ids == [151645, 151643]
    assert core.calls[0][2].ignore_eos is False
    with pytest.raises(ValueError, match='Invalid per-problem'):
        list(generate_problems(engine, [{**job, 'max_new_tokens': 8}], execution))


@pytest.mark.parametrize("bad_indices", [False, True])
def test_generation_cancels_pending_on_interrupt_or_invalid_outputs(monkeypatch, bad_indices):
    engine, _, _, execution = fixture()
    core = install_core(monkeypatch, engine, bad_indices)
    jobs = [{"key": i, "problem": str(i), "n": 4, "seed": i} for i in range(3)]
    stream = generate_problems(engine, jobs, execution)
    if bad_indices:
        with pytest.raises(ValueError, match="indices"):
            next(stream)
    else:
        next(stream)
        stream.close()
    assert core.aborted == ["eval-0"]
    assert not core.pending


def test_scores_match_original_and_resume_skips_complete_problems(tmp_path, monkeypatch, capsys):
    engine, datasets, config, execution = fixture()
    core = install_core(monkeypatch, engine)
    journal = tmp_path / "problems.jsonl"
    records, scores = runner.evaluate_batched(engine, datasets, config, execution, journal)
    assert len(core.calls) == 3
    assert [entry["records"][0]["id"] for entry in read_jsonl(journal)] == [1, 2, 0]
    assert [r["id"] for r in records] == [i for i in range(3) for _ in range(4)]
    assert scores["test"]["pass@k"] == {"1": 0.5, "4": 1.0}
    assert "test: 3/3 problems" in capsys.readouterr().out
    # Score the identical generated responses through the original evaluator.
    by_problem = {row["problem"]: [r for r in records if r["id"] == row["id"]]
                  for row in datasets["test"]}

    def generate(problem, n, seed):
        group = by_problem[problem]
        return group[0]["prompt"], [{"text": r["response"], "token_count": r["token_count"],
                                     "finish_reason": r["finish_reason"]} for r in group]

    original = evaluate(SimpleNamespace(generate=generate), datasets, config, tmp_path / "old.jsonl")
    assert original == scores
    with journal.open("ab") as stream:
        stream.write(b'{"records":[')  # simulate a killed append of the next problem
    assert runner.evaluate_batched(engine, datasets, config, execution, journal) == (records, scores)
    assert len(core.calls) == 3


def test_interrupted_scoring_regenerates_whole_problem_without_duplicates(tmp_path, monkeypatch):
    engine, datasets, config, execution = fixture()
    core = install_core(monkeypatch, engine)
    journal = tmp_path / "problems.jsonl"
    scorer = runner.score_response
    calls = 0

    def interrupt(*args):
        nonlocal calls
        calls += 1
        if calls == 6:  # first problem durable; second only partly scored
            raise KeyboardInterrupt
        return scorer(*args)

    monkeypatch.setattr(runner, "score_response", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.evaluate_batched(engine, datasets, config, execution, journal)
    assert len(read_jsonl(journal)) == 1
    assert not core.pending
    monkeypatch.setattr(runner, "score_response", scorer)
    core = install_core(monkeypatch, engine)
    records, _ = runner.evaluate_batched(engine, datasets, config, execution, journal)
    assert len(core.calls) == 2
    assert len(records) == 12
    assert len({(r["id"], r["sample_index"]) for r in records}) == 12


def source_fixture(tmp_path, monkeypatch):
    engine, datasets, config, execution = fixture()
    install_core(monkeypatch, engine)
    records, _ = runner.evaluate_batched(engine, datasets, config, execution, tmp_path / "source.jsonl")
    metadata = {key: key for key in ["model", "model_revision", "model_digest", "stage",
                                    "protocol_digest", "dataset_suite", "selected_ids",
                                    "packages", "hardware"]}
    metadata.update(config=config, mode="full")
    write_json(tmp_path / "old_manifest.json", {**metadata, "run_name": "old", "git_commit": "old"})
    write_json(tmp_path / "old_engine.json", {"engine": "vllm"})
    write_jsonl(tmp_path / "old_generations.jsonl", records[:5])
    return engine, datasets, config, execution, metadata, records


def test_legacy_reuse_copies_complete_only_leaves_source_intact(tmp_path, monkeypatch):
    engine, datasets, config, execution, metadata, records = source_fixture(tmp_path, monkeypatch)
    source = tmp_path / "old_generations.jsonl"
    with source.open("ab") as stream:
        stream.write(b'{"id":')
    original = source.read_bytes()
    groups, provenance = runner.legacy_source(tmp_path, "old", metadata, datasets, config)
    assert list(groups) == [("test", "0")]
    assert provenance["completed_problems"] == 1
    assert provenance["ignored_incomplete_responses"] == 1
    core = install_core(monkeypatch, engine)
    result, _ = runner.evaluate_batched(
        engine, datasets, config, execution, tmp_path / "new.jsonl", groups,
    )
    assert len(core.calls) == 2
    assert result[:4] == records[:4]
    assert source.read_bytes() == original


@pytest.mark.parametrize("field", ["model_digest", "config", "protocol_digest", "packages", "mode"])
def test_reuse_rejects_incompatible_source(tmp_path, monkeypatch, field):
    _, datasets, config, _, metadata, _ = source_fixture(tmp_path, monkeypatch)
    metadata[field] = "different"
    with pytest.raises(ValueError, match=field):
        runner.legacy_source(tmp_path, "old", metadata, datasets, config)


def test_corrupt_and_duplicate_journals_fail_closed(tmp_path, monkeypatch):
    engine, datasets, config, execution, _, records = source_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.complete_groups(records + records[:1], datasets, config)
    bad = deepcopy(records)
    bad[0]["problem_seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        runner.complete_groups(bad, datasets, config)
    with pytest.raises(ValueError, match="Incomplete"):
        runner.complete_groups(records[:1], datasets, config)
    journal = tmp_path / "bad.jsonl"
    journal.write_text('{bad}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        runner.read_source_records(journal)
    journal.unlink()
    append_jsonl(journal, {"records": records[:4]})
    append_jsonl(journal, {"records": records[:4]})
    with pytest.raises(ValueError, match="Duplicate"):
        runner.evaluate_batched(engine, datasets, config, execution, journal)


def test_missing_legacy_is_optional_but_orphaned_generations_fail(tmp_path):
    assert runner.legacy_source(tmp_path, "old", {}, {}, {}) == ({}, None)
    (tmp_path / "old_generations.jsonl").write_text("")
    with pytest.raises(ValueError, match="without their manifest"):
        runner.legacy_source(tmp_path, "old", {}, {}, {})


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_execution_config_rejects_invalid_queue(value):
    _, _, _, execution = fixture()
    execution["max_pending_problems"] = value
    with pytest.raises(ValueError, match="positive integer"):
        runner.validate_execution(execution)


def test_original_scientific_fingerprint_is_unchanged():
    from eval.run_english_eval import code_fingerprint

    assert code_fingerprint() == "da1d344e22a75f76b8e9012b6b78f97b0ede6f41b9e2fb545562e0c2e90b1db4"


def test_problem_seed_matches_string_and_native_ids():
    assert problem_seed(42, "test", 7) == problem_seed(42, "test", "7")


def test_cli_records_execution_exports_and_resumes_without_changing_baseline(tmp_path, monkeypatch):
    from eval import engines, english_engines
    from eval.run_english_eval import bind_protocol
    from eval.run_eval import bind_runtime

    engine, datasets, config, execution = fixture()
    core = install_core(monkeypatch, engine)
    config_path, execution_path, suite_path = [tmp_path / name for name in [
        "config.yaml", "execution.yaml", "suite.json",
    ]]
    config_path.write_text(yaml.safe_dump(config))
    execution_path.write_text(yaml.safe_dump(execution))
    suite_path.write_text("{}")
    root = tmp_path / "full"
    bind_protocol(root, config, {}, "stage0", {"test": [0, 1, 2]}, False)
    bind_runtime(root, engine.tokenizer, "stage0")
    baseline = {p: p.read_bytes() for p in (root / "results").glob("*.json")}
    monkeypatch.setattr(runner, "load_eval_sets", lambda *args: datasets)
    monkeypatch.setattr(runner, "git_identity", lambda: "new-commit")
    monkeypatch.setattr(runner, "resolve_model", lambda *args: (None, "weights"))
    monkeypatch.setattr(engines, "check_hardware", lambda *args: {"name": "gpu"})
    monkeypatch.setattr(english_engines, "create_engine", lambda *args: (engine, None))
    monkeypatch.setattr(sys, "argv", [
        "eval.run_batched_eval", "--model", "checkpoint", "--run_name", "sft_batched",
        "--reuse_run_name", "old", "--config", str(config_path),
        "--suite", str(suite_path), "--execution_config", str(execution_path),
        "--output_root", str(tmp_path),
    ])
    runner.main()
    result_path = root / "results/stage1/sft_batched.json"
    result = json.loads(result_path.read_text())
    assert result["execution"] == execution
    assert result["config"] == config
    assert result["legacy_source"] is None
    assert result["metrics"]["test"]["responses"] == 12
    assert len(read_jsonl(root / "results/stage1/sft_batched_generations.jsonl")) == 12
    assert len(core.calls) == 3
    runner.main()
    assert len(core.calls) == 3
    assert json.loads(result_path.read_text()) == result
    assert all(path.read_bytes() == contents for path, contents in baseline.items())
    execution["max_pending_problems"] += 1
    execution_path.write_text(yaml.safe_dump(execution))
    with pytest.raises(ValueError, match="identity/source changed"):
        runner.main()


def test_explicit_stop_tokens_reach_vllm_with_eos_enabled(monkeypatch):
    engine, _, _, execution = fixture()
    engine.config['stop_token_ids'] = [151643, 151645]
    core = install_core(monkeypatch, engine)
    list(generate_problems(engine, [{'key': 0, 'problem': 'Compute 1+1', 'n': 8, 'seed': 42}], execution))
    params = core.calls[0][2]
    assert params.stop_token_ids == [151643, 151645]
    assert params.ignore_eos is False


def test_sampling_stop_diagnostics_are_recorded_when_requested(monkeypatch):
    engine, _, _, execution = fixture()
    engine.config['record_stop_diagnostics'] = True
    install_core(monkeypatch, engine)
    result = list(generate_problems(engine, [{'key': 0, 'problem': 'Compute 1+1', 'n': 8, 'seed': 42}], execution))
    for response in result[0][2]:
        assert response['stop_reason'] is None
        assert response['last_token_id'] == 3
