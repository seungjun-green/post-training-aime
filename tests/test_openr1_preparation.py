import ast
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from datasets import Dataset, IterableDataset, load_dataset

from common.io import digest, write_json
from pipeline import openr1_preparation as prep


class Tokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs == dict(
            add_special_tokens=False, truncation=False, return_attention_mask=False
        )
        return {"input_ids": [list(range(len(text.split()))) for text in texts]}


def row(**overrides):
    return {
        "problem": "Compute the total cost of six red notebooks and three blue pencils today",
        "generations": [
            "An exact original <think> reasoning </think> answer",
            "Second correct answer",
        ],
        "correctness_math_verify": [True, True],
        "finish_reasons": ["stop", "stop"],
        "is_reasoning_complete": [True, True],
        "source": "olympiads",
        "uuid": "source-uuid",
        **overrides,
    }


def test_trace_arrays_remain_aligned_and_length_boundary_is_inclusive():
    original = row(
        problem="one two",
        generations=["three four", "five six seven", "wrong"],
        correctness_math_verify=[True, True, False],
        finish_reasons=["stop"] * 3,
        is_reasoning_complete=[True] * 3,
    )
    kept, counts = prep.filter_problem(original, 12, Tokenizer(), 4, "drop")
    assert kept["generations"] == ["three four"]
    assert kept["generation_indices"] == [0]
    assert kept["problem_generation_tokens"] == [4]
    assert kept["problem"] == original["problem"]
    assert kept["uuid"] == original["uuid"]
    assert counts["overlength_traces"] == counts["incorrect_or_unverified_traces"] == 1
    assert original["generations"] == ["three four", "five six seven", "wrong"]


@pytest.mark.parametrize(
    "reason,complete,reason_key",
    [
        (None, True, "missing_completion_evidence_traces"),
        ("length", True, "truncated_traces"),
        ("stop", False, "incomplete_reasoning_traces"),
        ("abort", True, "other_finish_reason_traces"),
    ],
)
def test_strict_completion_filter(reason, complete, reason_key):
    value = row(
        finish_reasons=None if reason is None else [reason] * 2,
        is_reasoning_complete=[complete] * 2,
    )
    kept, counts = prep.filter_problem(value, 0, Tokenizer(), 20480, "drop")
    assert kept is None and counts[reason_key] == 2


def test_one_valid_trace_preserves_original_generation_index():
    original = row(finish_reasons=[None, "stop"])
    kept, _ = prep.filter_problem(original, 0, Tokenizer(), 20480, "drop")
    assert kept["generation_indices"] == [1]
    assert kept["generations"] == original["generations"][1:]
    assert kept["completion_evidence"] == ["finish_reason_stop"]


def test_unverified_and_misaligned_arrays_are_not_treated_as_valid():
    kept, _ = prep.filter_problem(
        row(correctness_math_verify=[None, False]), 0, Tokenizer(), 20480, "drop"
    )
    assert kept is None
    with pytest.raises(ValueError, match="Misaligned"):
        prep.filter_problem(row(finish_reasons=["stop"]), 0, Tokenizer(), 20480, "drop")


def test_overlap_removes_whole_problem_and_all_its_traces():
    problem = "Find the sum of all integer bases b greater than nine for which seventeen divides ninety seven"
    exact, _ = prep.filter_problem(row(problem=problem), 0, Tokenizer(), 20480, "drop")
    unrelated, _ = prep.filter_problem(row(), 1, Tokenizer(), 20480, "drop")
    clean, report, _ = prep.remove_overlap(
        [exact, unrelated],
        {
            "aime_2025": [{"id": "eval0", "problem": problem}],
            "math_500": [{"id": "eval1", "problem": problem}],
        },
    )
    assert clean == [unrelated]
    assert report["removed_count"] == 1
    assert report["removal_counts_by_eval_set"] == {"aime_2025": 1, "math_500": 1}
    assert len(report["removed"][0]["matches"]) == 2
    assert all(m["coverage"] == 1 for m in report["removed"][0]["matches"])


def setup_run(tmp_path, monkeypatch):
    import datasets
    import transformers

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")

    reference = {
        "id": "eval0",
        "problem": "Find the number of integer points inside a circle of radius seven centered at the origin",
    }
    suite = {
        "datasets": {
            "aime_2024": {
                "repo": "test/eval",
                "config": "default",
                "split": "test",
                "revision": "a" * 40,
                "rows": 1,
                "content_digest": digest([reference]),
                "id_column": "id",
                "ids": ["eval0"],
                "problem_column": "problem",
            }
        }
    }
    settings = {
        "tokenizer": "Qwen/Qwen2.5-3B",
        "batch_size": 1,
        "max_tokens": 20480,
        "missing_finish": "drop",
        "eval_suite": suite,
    }
    manifest = {
        "settings": settings,
        "source_revision": "b" * 40,
        "tokenizer_revision": "c" * 40,
        "signature": "test-signature",
    }
    write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "source_card.md").write_text("Original source card")
    originals = [row(), row(problem=reference["problem"]), row(finish_reasons=None)]

    def loader(repo, **kwargs):
        if repo == prep.SOURCE:
            assert kwargs["name"] == "default" and kwargs["streaming"] is True
            assert kwargs["revision"] == "b" * 40
            return IterableDataset.from_generator(lambda: iter(originals))
        assert repo == "test/eval" and kwargs["revision"] == "a" * 40
        return Dataset.from_list([reference])

    monkeypatch.setattr(datasets, "load_dataset", loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    return originals


def test_end_to_end_filters_parquet_and_uploads_only_verified_files(tmp_path, monkeypatch):
    import huggingface_hub

    originals = setup_run(tmp_path, monkeypatch)
    result = prep.filter_traces(tmp_path, "test-token")
    assert result["counts"]["source_problems"] == 3
    assert result["counts"]["kept_traces"] == 4
    output = prep.decontaminate_filtered(tmp_path, "test-token")
    assert output["counts"] == {
        "kept_problems": 1,
        "kept_traces": 2,
        "removed_problems": 1,
        "removed_traces": 2,
    }
    # Use the real loader on the final Parquet to catch schema/list/nullability errors.
    ds = load_dataset(
        "parquet", data_files=str(tmp_path / "export/data/train-00000.parquet"), split="train"
    )
    assert len(ds) == 1 and ds[0]["generations"] == originals[0]["generations"]
    assert ds[0]["finish_reasons"] == ["stop", "stop"]
    assert "messages" not in ds.column_names
    assert prep.filter_traces(tmp_path, "test-token") == result
    assert prep.decontaminate_filtered(tmp_path, "test-token") == output

    class API:
        def __init__(self, **kwargs):
            pass

        def repo_exists(self, repo, **kwargs):
            assert repo == "user/dp_removed_OpenR1-Math-220k"
            return False

        def create_repo(self, repo, **kwargs):
            assert kwargs == dict(repo_type="dataset", private=True)

        def upload_folder(self, **kwargs):
            assert set(kwargs["allow_patterns"]) == {
                str(Path(n).relative_to("export")) for n in output["files"]
            }
            return SimpleNamespace(oid="uploaded-revision")

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    receipt = prep.publish(tmp_path, "user", True, "test-token")
    assert receipt["revision"] == "uploaded-revision"
    (tmp_path / "export/data/train-00000.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached output changed"):
        prep.publish(tmp_path, "user", True, "test-token")


def test_filter_resumes_completed_batches_without_retokenizing(tmp_path, monkeypatch):
    setup_run(tmp_path, monkeypatch)
    real_filter = prep.filter_problem
    seen = []

    def interrupted(row, index, *args):
        seen.append(index)
        if index == 1:
            raise RuntimeError("Interrupted")
        return real_filter(row, index, *args)

    monkeypatch.setattr(prep, "filter_problem", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted"):
        prep.filter_traces(tmp_path, "test-token")
    assert seen == [0, 1]
    seen.clear()

    def resumed(row, index, *args):
        seen.append(index)
        return real_filter(row, index, *args)

    monkeypatch.setattr(prep, "filter_problem", resumed)
    result = prep.filter_traces(tmp_path, "test-token")
    assert seen == [1, 2]
    assert result["counts"]["source_problems"] == 3


def test_notebook_is_clean_thin_launcher_with_user_choices():
    notebook = nbformat.read("notebooks/prepare_openr1_math_220k.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert 'TOKENIZER = "Qwen/Qwen2.5-3B"' in code
    assert 'MISSING_FINISH = "drop"' in code and "MAX_TOKENS = 20480" in code
    assert "BUNDLE" not in code and "@param" not in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebook", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            assert not cell.outputs and cell.execution_count is None


def test_initialization_pins_once_and_settings_changes_get_a_new_folder(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    card = tmp_path / "original-card.md"
    card.write_text("Original source metadata")

    class API:
        def __init__(self, **kwargs):
            assert kwargs == {"token": "SECRET"}

        def dataset_info(self, repo):
            calls.append(repo)
            return SimpleNamespace(sha="a" * 40)

        def model_info(self, repo):
            calls.append(repo)
            return SimpleNamespace(sha="b" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: card)
    first = prep.initialize_run(tmp_path, "SECRET")
    assert prep.initialize_run(tmp_path, "SECRET") == first
    assert len(calls) == 2
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["source_revision"] == "a" * 40
    assert manifest["tokenizer_revision"] == "b" * 40
    assert manifest["settings"]["tokenizer"] == "Qwen/Qwen2.5-3B"
    assert "SECRET" not in (first / "manifest.json").read_text()
    second = prep.initialize_run(tmp_path, "SECRET", max_tokens=20000)
    assert second != first and (first / "manifest.json").exists()


def test_publication_refuses_a_different_existing_run(tmp_path, monkeypatch):
    import huggingface_hub

    setup_run(tmp_path, monkeypatch)
    prep.filter_traces(tmp_path, "test-token")
    prep.decontaminate_filtered(tmp_path, "test-token")
    old = tmp_path / "old-provenance.json"
    write_json(old, {"signature": "another-run"})
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: old)
    api = SimpleNamespace(repo_exists=lambda *a, **k: True)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **k: api)
    with pytest.raises(ValueError, match="different preparation run"):
        prep.publish(tmp_path, "user", True, "test-token")
