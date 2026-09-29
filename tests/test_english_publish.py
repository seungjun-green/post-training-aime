from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from datasets import Dataset, Features, Value

from common.io import digest
from pipeline.decontamination import prepare_decontamination
from pipeline.english_publish import publish_english, stage_english, upload_name


def fixture(tmp_path):
    def wrap(i, problem):
        original = {"question": problem, "solution": "82", "trace": "Keep 82, not 8²."}
        return {"id": i, "problem": problem, "answer": "82", "original": original}

    problem = (
        "Find the positive integer solution to this particular equation with two unknown values"
    )
    repeated = "Different training question with all original numbers 82 and 92 preserved"
    data = {
        "train": [wrap(0, problem), wrap(1, repeated), wrap(2, repeated)],
        "eval": [wrap(3, problem)],
    }
    registry = {
        name: {
            "repo": f"owner/{name}",
            "config": "en" if name == "train" else "default",
            "split": "train" if name == "train" else "test",
            "role": name,
            "revision": "a" * 40,
            "license": None,
            "problem_column": "question",
            "answer_column": "solution",
        }
        for name in data
    }
    manifest = {
        name: {
            "revision": "a" * 40,
            "content_digest": digest(rows),
            "features": Features({k: Value("string") for k in rows[0]["original"]}).to_dict(),
        }
        for name, rows in data.items()
    }
    (tmp_path / "sources").mkdir()
    for name in data:
        (tmp_path / "sources" / f"{name}_source_card.md").write_text("Original source card")
    clean, report = prepare_decontamination(data, registry, tmp_path)
    return data, clean, registry, manifest, report


def test_original_rows_and_schema_survive_parquet_without_train_dedup(tmp_path):
    args = fixture(tmp_path)
    data, clean, registry, sources, report = args
    frozen = deepcopy(data)
    manifest = stage_english(*args, tmp_path)
    assert len(clean["train"]) == 2  # Both repeated training problems stay.
    assert clean["eval"] == data["eval"] and data == frozen
    for name, spec in registry.items():
        root = Path(manifest["folder"]) / upload_name(spec)
        loaded = Dataset.from_parquet(
            str(root / "data" / f"{spec['split']}-00000-of-00001.parquet"),
            cache_dir=str(tmp_path / "cache"),
        )
        assert loaded.to_list() == [r["original"] for r in clean[name]]
        assert loaded.features.to_dict() == sources[name]["features"]
        card = (root / "README.md").read_text()
        metadata = yaml.safe_load(card.split("---")[1])
        assert metadata["language"] == ["en"] and "license" not in metadata
        assert metadata["configs"][0]["config_name"] == spec["config"]
        assert (root / "SOURCE_README.md").read_text() == "Original source card"


def test_eval_mutation_and_missing_source_card_prevent_export(tmp_path):
    args = fixture(tmp_path)
    bad = deepcopy(args[1])
    bad["eval"][0]["original"]["question"] = "edited"
    with pytest.raises(ValueError, match="Evaluation rows changed"):
        stage_english(args[0], bad, *args[2:], tmp_path)
    (tmp_path / "sources/eval_source_card.md").unlink()
    with pytest.raises(FileNotFoundError):
        stage_english(*args, tmp_path)
    assert not (tmp_path / "english_exports").exists()


def test_publishing_names_atomic_commits_and_file_integrity(tmp_path):
    manifest = stage_english(*fixture(tmp_path), tmp_path)
    calls = []

    class API:
        def whoami(self):
            calls.append("auth")

        def create_repo(self, repo, **kw):
            assert repo.startswith("Seungjun/dp_removed_") and kw["private"]

        def update_repo_settings(self, repo, **kw):
            assert kw["private"]

        def upload_folder(self, **kw):
            assert kw["repo_type"] == "dataset"
            assert "README.md" in kw["allow_patterns"]
            assert "decontamination_report.json" in kw["allow_patterns"]
            assert any(p.endswith(".parquet") for p in kw["allow_patterns"])
            calls.append(kw["repo_id"])
            return SimpleNamespace(oid="commit-" + str(len(calls)))

    result = publish_english(manifest, "Seungjun", True, None, api=API())
    assert len(result) == 2
    assert result["train"]["rows"] == 2 and result["eval"]["removed_rows"] == 0
    assert result["eval"]["revision"] == "commit-3"
    path = Path(manifest["folder"]) / "dp_removed_eval/README.md"
    path.write_text("tampered")
    calls.clear()
    with pytest.raises(ValueError, match="Staged file changed"):
        publish_english(manifest, "Seungjun", True, None, api=API())
    assert calls == []


def test_exact_source_repo_names():
    registry = yaml.safe_load(Path("configs/datasets.yaml").read_text())
    assert [upload_name(spec) for spec in registry.values()] == [
        "dp_removed_s1K-1.1",
        "dp_removed_DAPO-Math-17k-Processed",
        "dp_removed_aime_2024",
        "dp_removed_aime25",
        "dp_removed_aime_2026",
        "dp_removed_amc23",
        "dp_removed_MATH-500",
    ]
