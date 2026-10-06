from copy import deepcopy
from pathlib import Path

import nbformat
import pytest
from datasets import Dataset

from common.io import digest
from pipeline.publish_kimi_answer import merge_answer


def inputs():
    rows = [{"question": "q1", "answer": "1"}, {"question": "q2", "answer": "2"}]
    source = Dataset.from_list(rows)
    current = Dataset.from_list([{**r, "deepseek-v4-pro_answer": "older answer"} for r in rows])
    exported = [{**r, "deepseek-v4-pro_answer": a, "deepseek-v4-pro_reasoning": "separate thinking"}
                for r, a in zip(rows, ["Planning\nEvaluation\nReflection\nExploration\nFinal answer: 1", None])]
    statuses = [{"source_index": 0, "status": "complete", "finish_reason": "stop",
                 "response_model": "deepseek-v4-pro"}, {"source_index": 1, "status": "failed"}]
    config = {"source": {"expected_rows": 2, "content_digest": digest(rows)},
              "new_column": "kimi-style-reasoning-answer", "answer_column": "deepseek-v4-pro_answer",
              "model": "deepseek-v4-pro"}
    return source, current, exported, statuses, config


def test_preserve_existing_and_copy_only_answer_with_null_and_rerun():
    source, current, exported, statuses, config = inputs()
    merged, report = merge_answer(source, current, exported, statuses, config)
    assert list(merged.remove_columns(config["new_column"])) == list(current)
    assert list(merged[config["new_column"]]) == [r[config["answer_column"]] for r in exported]
    assert report["answers"] == report["null_answers"] == 1
    again, _ = merge_answer(source, merged, exported, statuses, config)
    assert list(again) == list(merged)


@pytest.mark.parametrize("change", ["order", "missing", "truncated", "conflict", "status_order"])
def test_reject_unsafe_merges(change):
    source, current, exported, statuses, config = inputs()
    exported = deepcopy(exported)
    if change == "order":
        exported.reverse()
    elif change == "missing":
        exported.pop()
    elif change == "truncated":
        statuses[0]["finish_reason"] = "length"
    elif change == "conflict":
        current = current.add_column(config["new_column"], ["different", None])
    elif change == "status_order":
        statuses.reverse()
    with pytest.raises(ValueError):
        merge_answer(source, current, exported, statuses, config)


def test_notebook_valid_and_cells_compile():
    notebook = nbformat.read(Path(__file__).resolve().parents[1] / "notebooks/upload_s1_kimi_answer_to_huggingface.ipynb", 4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert "@param" not in cell.source
            compile(cell.source, "notebook", "exec")


def test_shards_reject_other_splits_and_missing_parts():
    from pipeline.publish_kimi_answer import train_parquet_files

    valid = ["data/train-00000-of-00001.parquet"]
    assert train_parquet_files(["README.md", *valid]) == valid
    for bad in [[], ["data/train-00000-of-00002.parquet"],
                [*valid, "data/test-00000-of-00001.parquet"]]:
        with pytest.raises(ValueError):
            train_parquet_files(bad)


def test_datasetdict_metadata_path_repairs_stale_features(tmp_path):
    """Exercise the actual pinned datasets uploader's metadata builder and reader."""
    import json

    import fsspec
    from datasets import Features, Value, load_dataset
    from datasets.arrow_dataset import _get_updated_dataset_card
    from datasets.splits import SplitInfo

    (tmp_path / "data").mkdir()
    data = Dataset.from_dict({"question": ["q"], "kimi-style-reasoning-answer": ["worked answer"]})
    data.to_parquet(tmp_path / "data/train-00000-of-00001.parquet")
    old_info = {"features": {"question": {"dtype": "string", "_type": "Value"}},
                "splits": {"train": {"name": "train", "num_examples": 1, "num_bytes": 10}},
                "download_size": 10, "dataset_size": 10}
    (tmp_path / "dataset_infos.json").write_text(json.dumps({"default": old_info}))
    (tmp_path / "README.md").write_text('''---
license: mit
dataset_info:
  features:
    - name: question
      dtype: string
  splits:
    - name: train
      num_examples: 1
      num_bytes: 10
  download_size: 10
  dataset_size: 10
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
---
Keep this dataset description.
''')
    with pytest.raises(Exception, match="generating the dataset"):
        load_dataset(str(tmp_path), split="train", cache_dir=str(tmp_path / "cache-before"))
    fs = fsspec.filesystem("dir", path=str(tmp_path))
    kwargs = dict(fs=fs, config_name="default", splits_info=[SplitInfo(name="train", num_examples=1, num_bytes=30)],
                  features=data.features, data_dir="data", set_default=None,
                  uploaded_sizes=[30], deleted_sizes=[10])
    # Dataset.push_to_hub's existing-info branch reproduces the stale schema.
    stale, _ = _get_updated_dataset_card(**kwargs, remove_other_splits=False)
    assert len(stale.data.to_dict()["dataset_info"]["features"]) == 1
    # DatasetDict.push_to_hub uses this branch and refreshes README + legacy metadata.
    card, legacy = _get_updated_dataset_card(**kwargs, remove_other_splits=True)
    assert "Keep this dataset description." in str(card)
    assert card.data.license == "mit"
    assert Features.from_dict(legacy["default"]["features"]) == Features(
        {"question": Value("string"), "kimi-style-reasoning-answer": Value("string")})
    (tmp_path / "README.md").write_text(str(card))
    (tmp_path / "dataset_infos.json").write_text(json.dumps(legacy))
    restored = load_dataset(str(tmp_path), split="train", cache_dir=str(tmp_path / "cache-after"))
    assert list(restored) == list(data)
