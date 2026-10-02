import ast
import copy
from pathlib import Path

import nbformat
import pytest
from datasets import Dataset

from common.io import digest
from pipeline.publish_s1_regeneration import merge_export


@pytest.fixture
def inputs():
    rows = [{"question": "Find α\n+ β", "solution": "a", "grade": "No"},
            {"question": "Other", "solution": "b", "grade": "Yes"}]
    source = Dataset.from_list(rows)
    exported = [{**rows[0], "pro_reasoning": "think\n" * 30000, "pro_answer": "answer"},
                {**rows[1], "pro_reasoning": None, "pro_answer": None}]
    statuses = [{"source_index": 0, "status": "complete", "finish_reason": "stop",
                 "response_model": "pro"},
                {"source_index": 1, "status": "incomplete", "finish_reason": "length"}]
    config = {"source": {"expected_rows": 2, "content_digest": digest(rows)},
              "model": "pro", "new_columns": ["pro_reasoning", "pro_answer"]}
    return source, source, exported, statuses, config


def test_preserves_text_nulls_extra_columns_and_resumes(inputs):
    source, current, exported, statuses, config = inputs
    current = current.add_column("unrelated", ["keep", "also keep"])
    merged, summary = merge_export(source, current, exported, statuses, config)
    assert summary["complete"] == summary["incomplete"] == 1
    assert merged[0]["pro_reasoning"] == exported[0]["pro_reasoning"]
    assert merged[1]["pro_reasoning"] is None
    assert merged[0]["unrelated"] == "keep"
    assert merged.features["pro_reasoning"].dtype == "string"
    again, _ = merge_export(source, merged, exported, statuses, config)
    assert list(again) == list(merged)
    filled = copy.deepcopy(exported)
    filled[1].update(pro_reasoning="new think", pro_answer="new answer")
    statuses[1].update(status="complete", finish_reason="stop", response_model="pro")
    completed, summary = merge_export(source, merged, filled, statuses, config)
    assert summary["complete"] == 2
    with pytest.raises(ValueError, match="replace an existing"):
        merge_export(source, completed, exported, inputs[3][:1] + [
            {"source_index": 1, "status": "incomplete"}], config)


@pytest.mark.parametrize("fault", ["order", "text", "status", "truncated", "missing",
                                   "model", "unknown_column"])
def test_rejects_invalid_exports(inputs, fault):
    source, current, exported, statuses, config = inputs
    if fault == "order":
        exported.reverse()
    elif fault == "text":
        exported[0]["solution"] = "changed"
    elif fault == "status":
        statuses[1]["source_index"] = 0
    elif fault == "truncated":
        statuses[0]["finish_reason"] = "length"
    elif fault == "missing":
        exported[0]["pro_answer"] = None
    elif fault == "model":
        statuses[0]["response_model"] = "other-model"
    elif fault == "unknown_column":
        exported[0]["extra"] = "unexpected"
    with pytest.raises(ValueError):
        merge_export(source, current, exported, statuses, config)


def test_notebook_is_valid_and_uploads_only_after_validation():
    notebook = nbformat.read("notebooks/upload_s1_deepseek_to_huggingface.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for source in code_cells:
        ast.parse(source)
    assert "merge_export(" in code_cells[3]
    assert "push_to_hub(" in code_cells[4]
    assert "receipt[\"verified\"] = True" in code_cells[5]
    # Bundled code is the same implementation exercised by the tests.
    import base64
    import io
    from zipfile import ZipFile
    module = ast.parse(code_cells[1])
    encoded = next(ast.literal_eval(node.value) for node in module.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "BUNDLE" for t in node.targets))
    with ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        for filename in archive.namelist():
            assert archive.read(filename) == Path(filename).read_bytes()


def test_actual_notebook_validate_upload_verify_cells(inputs, tmp_path, monkeypatch):
    """Exercise publication/receipt/reload flow without credentials or network writes."""
    import datasets
    import huggingface_hub
    import json
    import shutil
    from types import SimpleNamespace
    from common.io import write_jsonl

    source, current, exported, statuses, config = inputs
    config["source"].update(repo="test/source", config="default", split="train", revision="pinned")
    config["target"] = {"repo": "test/source", "config": "default", "split": "train",
                        "revision": "main"}
    config["files"] = {"dataset": str(tmp_path / "incoming.jsonl"),
                       "status": str(tmp_path / "status.jsonl"),
                       "receipt_directory": str(tmp_path / "receipts")}
    write_jsonl(config["files"]["dataset"], exported)
    write_jsonl(config["files"]["status"], statuses)
    state = {"head": "before"}

    class FakeApi:
        def __init__(self, token):
            assert token == "mock-token"

        def dataset_info(self, repo, revision):
            return SimpleNamespace(sha=state["head"])

        def upload_file(self, **kwargs):
            backup = tmp_path / "remote.zip"
            shutil.copyfile(kwargs["path_or_fileobj"], backup)
            state.update(backup=backup, head="backup-commit")
            return SimpleNamespace(oid="backup-commit")

    def fake_push(dataset, repo, **kwargs):
        assert kwargs["config_name"] == "default" and kwargs["split"] == "train"
        state.update(dataset=dataset, head="dataset-commit")
        return SimpleNamespace(oid="dataset-commit")

    def fake_load(repo, **kwargs):
        return {"pinned": source, "before": current,
                "backup-commit": state.get("dataset")}[kwargs["revision"]]

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setattr(Dataset, "push_to_hub", fake_push)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: state["backup"])
    notebook = nbformat.read("notebooks/upload_s1_deepseek_to_huggingface.ipynb", as_version=4)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    namespace = {"CONFIG": config, "HF_TOKEN": "mock-token"}
    for cell in code_cells[3:]:
        exec(compile(cell.replace("/content/", str(tmp_path) + "/"), "notebook-cell", "exec"),
             namespace)
    receipt = json.loads(namespace["receipt_path"].read_text())
    assert receipt["verified"] is True
    assert receipt["dataset_revision"] == "dataset-commit"
    assert receipt["revision"] == "backup-commit"
    assert list(state["dataset"]) == exported
    assert "mock-token" not in namespace["receipt_path"].read_text()
