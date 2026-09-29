import json
from pathlib import Path

import pytest
import yaml

from common.io import append_jsonl, digest, latest_by_id, write_json
from pipeline.decontamination import (
    decontaminate,
    normalize,
    prepare_decontamination,
    print_decontamination_report,
)
from pipeline.decontamination_cache import refresh_translation_cache
from pipeline.translation import prepare_run, run_signature

REGISTRY = {"train": {"role": "train"}, "eval": {"role": "eval"}}


def pair(training, evaluation, **kwargs):
    data = {
        "train": [{"id": "t", "problem": training}],
        "eval": [{"id": "e", "problem": evaluation}],
    }
    return decontaminate(data, REGISTRY, **kwargs)


def test_punctuation_dropped_after_latex_cleanup():
    assert normalize(r"$\left(\frac{m}{n}\right)\quad + \; [x], !!! ___$") == "frac m n x"


def test_boilerplate_only_kept():
    shared = ", where m and n are relatively prime positive integers. find m + n"
    train = " ".join("triangle" + str(i) for i in range(35)) + shared
    evaluation = " ".join("sequence" + str(i) for i in range(35)) + shared
    clean, report, _ = pair(train, evaluation)
    assert len(clean["train"]) == 1
    assert report["train"]["removed_count"] == 0


def test_fraction_only_kept():
    train = r"A triangle has one angle whose sine equals \frac{1}{2}. Calculate its perimeter."
    evaluation = r"A coin has probability \frac{1}{2} of landing heads. Determine the expected number of tosses."
    clean, _, _ = pair(train, evaluation)
    assert len(clean["train"]) == 1


PROBLEM = (
    "A student writes forty distinct positive integers on a board in increasing order and "
    "then erases every third number before computing the sum of the remaining entries "
    "and multiplying that sum by seven to obtain a final total. Determine the largest "
    "possible value of the smallest original integer if the total equals nine thousand."
)


def test_one_word_changed_removed():
    clean, report, _ = pair(PROBLEM.replace("seven", "eight"), PROBLEM)
    assert not clean["train"]
    match = report["train"]["removed"][0]["matches"][0]
    assert 0.7 <= match["coverage"] < 1


def test_embedded_eval_removed_and_eval_unchanged():
    data = {
        "train": [
            {
                "id": "t",
                "problem": "Solve carefully and show steps. " + PROBLEM + " Return the result.",
            }
        ],
        "eval": [{"id": "e", "problem": PROBLEM}],
    }
    before = digest(data)
    clean, report, _ = decontaminate(data, REGISTRY)
    assert not clean["train"]
    assert clean["eval"] is data["eval"]
    assert report["train"]["removed"][0]["matches"][0]["coverage"] == 1
    assert digest(data) == before


def test_distinct_grams_not_occurrences_and_near_miss_interval():
    # Five distinct evaluation bigrams, three present in training, despite repetition.
    clean, report, near = pair("a b c d a b c d", "a b c d e f", ngram_size=2)
    assert len(clean["train"]) == 1
    assert report["train"]["near_miss_count"] == 1
    assert near[0]["best_coverage"] == 0.6
    assert near[0]["matches"][0]["matched_ngram_count"] == 3
    assert near[0]["matches"][0]["eval_ngram_count"] == 5
    clean, report, near = pair(
        "a b c d", "a b c d e f", ngram_size=2, coverage_threshold=0.6, near_miss_min=0.5
    )
    assert not clean["train"] and not near
    assert report["train"]["removed"][0]["best_coverage"] == 0.6


def test_coverage_never_pooled_across_eval_problems():
    data = {
        "train": [{"id": "t", "problem": "a b c d j k l m"}],
        "eval": [{"id": "e1", "problem": "a b c d e f"}, {"id": "e2", "problem": "j k l m n o"}],
    }
    clean, _, near = decontaminate(data, REGISTRY, ngram_size=2)
    assert len(clean["train"]) == 1
    assert len(near[0]["matches"]) == 2
    assert near[0]["best_coverage"] == 0.6


def test_all_qualifying_pairs_recorded_and_counts_deduplicated():
    data = {
        "train": [{"id": "t", "problem": PROBLEM}],
        "eval": [{"id": "e1", "problem": PROBLEM}, {"id": "e2", "problem": PROBLEM}],
    }
    _, report, _ = decontaminate(data, REGISTRY)
    assert len(report["train"]["removed"][0]["matches"]) == 2
    assert report["train"]["removal_counts_by_eval_set"] == {"eval": 1}


def test_short_eval_substring_still_removes():
    clean, report, _ = pair("Instructions: find x now please", "find x")
    assert not clean["train"]
    assert report["train"]["removed"][0]["matches"][0]["method"] == "short_exact_substring"


def test_versioned_caches_leave_previous_outputs_untouched(tmp_path):
    old = tmp_path / "decontamination"
    old.mkdir()
    (old / "decontamination_report.json").write_text('{"normalization_version": 1}')
    (old / "train_decontaminated.jsonl").write_text("old rows\n")
    snapshot = {p.name: p.read_bytes() for p in old.iterdir()}
    data = {"train": [{"id": "t", "problem": PROBLEM}], "eval": [{"id": "e", "problem": PROBLEM}]}
    first = prepare_decontamination(data, REGISTRY, tmp_path)
    assert first[1]["cache_directory"] == "decontamination_v2"
    first_path = tmp_path / "decontamination_v2/decontamination_report.json"
    first_bytes = first_path.read_bytes()
    changed = prepare_decontamination(data, REGISTRY, tmp_path, coverage_threshold=0.9)
    assert changed[1]["cache_directory"].startswith("decontamination_v2/")
    assert changed == prepare_decontamination(data, REGISTRY, tmp_path, coverage_threshold=0.9)
    changed_n = prepare_decontamination(data, REGISTRY, tmp_path, ngram_size=9)
    changed_near = prepare_decontamination(data, REGISTRY, tmp_path, near_miss_min=0.4)
    assert len({r[1]["input_digest"] for r in [first, changed, changed_n, changed_near]}) == 4
    assert first_path.read_bytes() == first_bytes
    assert {p.name: p.read_bytes() for p in old.iterdir()} == snapshot


def test_sanity_warning_prints_real_pairs_not_forced_zero(tmp_path, capsys):
    registry = {
        "s1k_1.1": {"role": "train"},
        "aime_2025": {"role": "eval"},
        "aime_2026": {"role": "eval"},
    }
    data = {
        "s1k_1.1": [{"id": "training", "problem": PROBLEM}],
        "aime_2025": [{"id": "future", "problem": PROBLEM}],
        "aime_2026": [],
    }
    _, report = prepare_decontamination(data, registry, tmp_path)
    print_decontamination_report(report)
    output = capsys.readouterr().out
    assert "aime_2025 removed 1 rows" in output
    assert "aime_2026 removed 0 rows" in output
    assert "WARNING" in output and "coverage=1.0000" in output
    assert "TRAINING PROBLEM" in output and "EVALUATION PROBLEM" in output
    assert "10 removed rows" in output


def test_translation_cohort_migration_reuses_ids_without_changing_translator(tmp_path):
    config = yaml.safe_load(Path("configs/translation.yaml").read_text())
    config.update(PROJECT_ROOT=str(tmp_path), SMOKE_TEST_N=1)
    data = {
        "train": [
            {"id": "kept", "problem": "old retained row"},
            {"id": "restored", "problem": "previously excluded row"},
        ],
        "eval": [{"id": "eval", "problem": "unique eval row"}],
    }
    prior = {**data, "train": data["train"][:1]}
    write_json(
        tmp_path / "decontamination/decontamination_report.json",
        {
            "datasets": {"train": {"removed": [{"id": "restored"}]}},
            "output_digests": {"train": digest(prior["train"])},
        },
    )
    # Previously completed full translations and a checked smoke run.
    for mode in ["smoke_test", "translations"]:
        write_json(
            tmp_path / mode / "manifest.json",
            {
                "signature": run_signature(config, REGISTRY, prior),
                "checks_completed": True,
                "mode": mode,
                "config": config,
                "selected_ids": {"train": ["kept"], "eval": ["eval"]},
            },
        )
        for name, rows in prior.items():
            for row in rows:
                append_jsonl(
                    tmp_path / mode / f"{name}.jsonl",
                    {
                        "id": row["id"],
                        "source_digest": digest(row),
                        "translation_model": config["TRANSLATION_MODEL"],
                        "translations": {"ko_problem": "이미 번역됨"},
                        "attempt": 1,
                    },
                )
    report = {"datasets": {}, "cache_directory": "decontamination_v2"}
    refresh_translation_cache(data, data, REGISTRY, config, report)
    records = latest_by_id(tmp_path / "translations/train.jsonl")
    assert records["kept"]["translations"]["ko_problem"] == "이미 번역됨"
    assert "restored" not in records  # Newly eligible rows still need translation.
    manifest = json.loads((tmp_path / "translations/manifest.json").read_text())
    assert manifest["selected_ids"]["train"] == ["kept", "restored"]
    assert manifest["signature"] == run_signature(config, REGISTRY, data)
    assert not manifest["checks_completed"]
    assert list((tmp_path / "translations/decontamination_history").glob("*/manifest.json"))
    assert prepare_run(config, REGISTRY, data)[0]
    config["SMOKE_TEST"] = False
    with pytest.raises(ValueError, match="completed smoke test"):
        prepare_run(config, REGISTRY, data)
    config["TRANSLATION_MODEL"] = "different"
    with pytest.raises(ValueError, match="beyond decontamination"):
        refresh_translation_cache(data, data, REGISTRY, config, report)


def test_first_full_run_reuses_completed_smoke_rows_by_id(tmp_path):
    config = yaml.safe_load(Path("configs/translation.yaml").read_text())
    config.update(PROJECT_ROOT=str(tmp_path), SMOKE_TEST_N=1)
    data = {
        "train": [{"id": "t", "problem": "training example"}],
        "eval": [{"id": "e", "problem": "evaluation example"}],
    }
    selected, folder, manifest = prepare_run(config, REGISTRY, data)
    for name, rows in selected.items():
        for row in rows:
            append_jsonl(
                folder / f"{name}.jsonl",
                {
                    "id": row["id"],
                    "source_digest": digest(row),
                    "translation_model": config["TRANSLATION_MODEL"],
                    "attempt": 1,
                    "translations": {"ko_problem": "이미 번역한 문제"},
                },
            )
    manifest["checks_completed"] = True
    write_json(folder / "manifest.json", manifest)
    config["SMOKE_TEST"] = False
    refresh_translation_cache(
        data, data, REGISTRY, config, {"cache_directory": "decontamination_v2"}
    )
    _, full_folder, _ = prepare_run(config, REGISTRY, data)
    assert latest_by_id(full_folder / "train.jsonl")["t"]["translations"] == {
        "ko_problem": "이미 번역한 문제"
    }
    assert json.loads((folder / "manifest.json").read_text())["checks_completed"]
