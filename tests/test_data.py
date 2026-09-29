from copy import deepcopy

import pytest

from common.io import append_jsonl, latest_by_id, read_jsonl
from common.math_text import last_boxed
from pipeline.datasets import DAPO_PREFIX, adapt_rows, core_dapo_problem
from pipeline.decontamination import decontaminate, normalize, prepare_decontamination
from pipeline.publish import assemble_row


def test_boxed_nested_escaped_and_malformed():
    assert last_boxed(r"old \boxed{7}, final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{\{1,2\}}") == r"\{1,2\}"
    assert last_boxed(r"\boxed{7} later \boxed{broken") is None
    assert last_boxed("Answer: 7") is None


def test_normalization():
    assert normalize(r"$\left(X\right)\, +\quad Y$") == "x y"
    assert normalize(r"\(A\) \[B\]\;\!") == "a b"
    assert normalize("Words, words!") == "words words"
    assert normalize(r"\rightarrow") == "rightarrow"


def test_exact_eight_short_and_training_only():
    data = {
        "eval": [
            {"id": 0, "problem": "One two three four five six seven eight"},
            {"id": 1, "problem": "short phrase"},
        ],
        "train": [
            {"id": "a", "problem": "prefix One two three four five six seven eight suffix"},
            {"id": "b", "problem": "only one two three four five six seven"},
            {"id": "c", "problem": "a short phrase appears"},
        ],
    }
    before = deepcopy(data)
    registry = {"eval": {"role": "eval"}, "train": {"role": "train"}}
    clean, report, _ = decontaminate(data, registry)
    assert data == before
    assert clean["eval"] == before["eval"]
    assert [r["id"] for r in clean["train"]] == ["b"]
    assert report["train"]["removed_count"] == 2
    assert report["train"]["removed"][0]["matches"][0]["eval_id"] == 0


def test_cached_decontamination_detects_corruption(tmp_path):
    data = {
        "eval": [{"id": 0, "problem": "short phrase"}],
        "train": [{"id": "keep", "problem": "something unrelated"}],
    }
    registry = {"eval": {"role": "eval"}, "train": {"role": "train"}}
    expected = prepare_decontamination(data, registry, tmp_path)
    assert prepare_decontamination(data, registry, tmp_path) == expected
    (tmp_path / "decontamination_v2/train_decontaminated.jsonl").write_text("")
    with pytest.raises(ValueError, match="checksum"):
        prepare_decontamination(data, registry, tmp_path)


def test_adapters_keep_original_columns_and_stable_ids():
    original = {
        "prompt": DAPO_PREFIX + "What is $2+2$?",
        "solution": "4",
        "extra_info": {"index": "abc"},
        "source_prompt": [{"role": "user", "content": "original"}],
    }
    before = deepcopy(original)
    spec = {
        "id_column": "extra_info.index",
        "problem_column": "prompt",
        "answer_column": "solution",
        "split": "train",
    }
    row = adapt_rows("dapo_math_17k", [original], spec)[0]
    assert row["problem"] == "What is $2+2$?"
    assert row["id"] == "abc"
    assert original == before == row["original"]
    output = assemble_row(row, {"translations": {"ko_problem": "문제 $2+2$?"}}, "dapo_math_17k")
    assert all(output[k] == v for k, v in original.items())
    assert output["problem"] == row["problem"]
    with pytest.raises(ValueError, match="Unknown DAPO"):
        core_dapo_problem("Solve the following math problem NEW WRAPPER")


def test_generated_id_uses_original_index():
    spec = {
        "id_column": None,
        "problem_column": "question",
        "answer_column": "solution",
        "split": "train",
    }
    rows = adapt_rows(
        "s1", [{"question": "a", "solution": "0"}, {"question": "b", "solution": "1"}], spec
    )
    assert rows[1]["id"] == "s1:train:1"


def test_journal_recovers_only_interrupted_tail(tmp_path):
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"id": 1, "x": "original"})
    with path.open("ab") as f:
        f.write(b'{"id":2,"x":"interrupted')
    assert latest_by_id(path) == {"1": {"id": 1, "x": "original"}}
    append_jsonl(path, {"id": 1, "x": "retry"})
    assert latest_by_id(path)["1"]["x"] == "retry"
    path.write_text('{bad}\n{"id":1}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        read_jsonl(path, repair_tail=True)


def test_download_recovers_missing_card_directory_and_cached_card(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace

    import datasets
    import huggingface_hub

    from pipeline.datasets import download_sources

    root = tmp_path / "drive" / "project"
    downloaded = tmp_path / "hub_readme.md"
    downloaded.write_text("# Source card\n한국어\n", encoding="utf-8")
    dataset = datasets.Dataset.from_list([{"question": "Find 1+1", "solution": "2"}])
    registry = {
        "test": {
            "repo": "test/source",
            "revision": "pinned",
            "config": "default",
            "split": "train",
            "expected_rows": 1,
            "problem_column": "question",
            "answer_column": "solution",
            "id_column": None,
            "translate": ["question"],
            "license": None,
        }
    }
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda **kwargs: SimpleNamespace(
            dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="pinned", card_data=None)
        ),
    )
    downloads = []

    def download(*args, **kwargs):
        downloads.append(kwargs)
        if len(downloads) == 1:
            # Reproduce the destination disappearing after the initial mkdir.
            shutil.rmtree(root / "sources")
        return str(downloaded)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    initial = download_sources(registry, root)
    card = root / "sources/test_source_card.md"
    assert card.read_text(encoding="utf-8") == downloaded.read_text(encoding="utf-8")
    assert not card.with_suffix(".md.tmp").exists()

    card.unlink()

    def unexpected_reload(*args, **kwargs):
        pytest.fail("Complete dataset rows should be reused when only their card is missing")

    monkeypatch.setattr(datasets, "load_dataset", unexpected_reload)
    assert download_sources(registry, root) == initial
    assert card.exists()
    assert len(downloads) == 2
    assert download_sources(registry, root) == initial
    assert len(downloads) == 2
