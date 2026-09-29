from zipfile import ZipFile

import pytest

from pipeline.publish import archive_outputs, dataset_card, publish


def test_source_license_not_invented_and_2026_retained():
    spec = {
        "repo": "source/data",
        "upload_name": "data-ko",
        "revision": "abc",
        "config": "default",
        "split": "test",
        "license": None,
        "role": "eval",
    }
    checks = {"datasets": {"test": {"accepted": 30, "flagged_count": 0}}}
    card = dataset_card("test", spec, {"TRANSLATION_MODEL": "requested"}, {}, checks)
    assert "Not specified by the source" in card
    assert "\nlicense:" not in card
    spec["license"] = "cc-by-nc-sa-4.0"
    assert "license: cc-by-nc-sa-4.0" in dataset_card(
        "test", spec, {"TRANSLATION_MODEL": "requested"}, {}, checks
    )


def test_no_upload_in_smoke_mode():
    with pytest.raises(ValueError, match="never upload"):
        publish({}, {}, {"SMOKE_TEST": True}, {}, {}, {}, "not-a-real-token")


def test_archive_excludes_previous_archives_and_secrets(tmp_path):
    (tmp_path / "smoke_test").mkdir()
    (tmp_path / "smoke_test/rows.jsonl").write_text("{}\n")
    (tmp_path / ".env").write_text("not-a-real-secret")
    first = archive_outputs(tmp_path, "smoke_test")
    second = archive_outputs(tmp_path, "smoke_test")
    with ZipFile(second) as archive:
        assert archive.namelist() == ["smoke_test/rows.jsonl"]
    assert first.exists()
