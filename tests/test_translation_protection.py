import json
from pathlib import Path

import httpx
import pytest
import yaml

from common.io import digest, read_jsonl
from pipeline.deepseek_translation import MODELS, DeepSeekTranslator, model_config, run_model
from pipeline.translation_protection import ProtectedText, preservation_issues


@pytest.mark.parametrize(
    "source",
    [
        "effectively 82 copies of S0, in a 3x3 grid; 8*(8/92) = 82 / 92; 8n / 9n",
        r"Find $m - \sqrt{n}$ and \frac{m}{n}, then \boxed{\frac{1}{2}}.",
        r"Use \(82\), \[92\], $$8^2$$ and $\displaystyle x^{2}$.",
        "Code: ```python\nprint(82)\n``` and `x = 92` [asy]draw((0,0)--(1,2));[/asy]",
        "Keep -1.25, 1,000, 6.02e-23, 8n and S0.",
        "Malformed source stays malformed: $x^{2 and \\frac{m}{n",
    ],
)
def test_lossless_roundtrip(source):
    protected = ProtectedText.from_source(source)
    assert protected.restore(protected.masked) == source
    assert "82" not in protected.replacements.values() or "82" not in protected.masked


def test_regression_lost_superscripts_are_not_repaired():
    source = "effectively 82 copies of S0, total area 8*(8/92) = 82 / 92."
    protected = ProtectedText.from_source(source)
    translated = protected.masked.replace("effectively", "사실상").replace("copies of", "복사본")
    output = protected.restore(translated)
    assert "82" in output and "8^2" not in output and "92" in output
    corrupted = output.replace("82", "8^2").replace("92", "9^2")
    assert any("numeric_literals_changed" in s for s in preservation_issues(source, corrupted))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s, p: s.replace(next(iter(p)), "8^2"),
        lambda s, p: s + next(iter(p)),
        lambda s, p: s + "⟪KEEP_abcdef_999⟫",
        lambda s, p: s.replace("KEEP_", "KEEP "),
        lambda s, p: s + "⟪KEEP_broken⟫",
        lambda s, p: s.replace(next(iter(p)), r"\(" + next(iter(p)) + r"\)"),
        lambda s, p: s.replace(next(iter(p)), next(iter(p)) + "^2"),
        lambda s, p: s.replace(next(iter(p)), next(iter(p)) + "²"),
        lambda s, p: s.replace(next(iter(p)), next(iter(p)) + "^n"),
    ],
)
def test_corrupted_or_wrapped_placeholders_fail(mutation):
    protected = ProtectedText.from_source("There are 82 copies.")
    with pytest.raises(ValueError):
        protected.restore(mutation(protected.masked, protected.replacements))


def test_reordering_for_korean_grammar_and_duplicate_values():
    protected = ProtectedText.from_source("For 82 copies use $d$ and another 82 copies.")
    a, b, c = protected.replacements
    assert (
        protected.restore(f"{b}를 사용해 {a}개와 {c}개의 복사본")
        == "$d$를 사용해 82개와 82개의 복사본"
    )
    with pytest.raises(ValueError, match="placeholder_integrity"):
        protected.restore(f"{b} {a} {a}")


def test_existing_latex_reformatting_and_invented_numbers_fail():
    assert preservation_issues(r"Find $\displaystyle x^2$.", r"\(x^2\)를 구하세요.")
    assert preservation_issues("Eight copies", "8개의 복사본")
    assert not preservation_issues("Eight copies", "여덟 개의 복사본")
    assert not preservation_issues(r"Use \frac{m}{n}.", r"\frac{m}{n}를 사용하세요.")
    assert preservation_issues("Compute 82 / 92.", "82 × 92를 계산하세요.")


def config_for(tmp_path):
    base = yaml.safe_load(Path("configs/translation_deepseek.yaml").read_text())
    base.update(PROJECT_ROOT=str(tmp_path), SMOKE_TEST_N=1, MAX_RETRIES=0)
    return base


def response(output, finish="stop"):
    return httpx.Response(
        200,
        json={
            "model": MODELS[0],
            "choices": [{"finish_reason": finish, "message": {"content": output}}],
        },
    )


async def test_real_request_masks_source_then_restores_and_logs(tmp_path):
    source = "There are 82 copies of $S_0$."

    def handler(request):
        body = json.loads(request.content)
        assert "82" not in body["messages"][1]["content"]
        assert "$S_0$" not in body["messages"][1]["content"]
        return response(body["messages"][1]["content"].replace("There are", "있다"))

    config = model_config(config_for(tmp_path), MODELS[0], True)
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler)
    ) as client:
        translator = DeepSeekTranslator(client, config, tmp_path / "usage.jsonl")
        output, _ = await translator.chunk(source)
    assert output == "있다 82 copies of $S_0$."
    assert read_jsonl(tmp_path / "usage.jsonl")[0]["protected_occurrences"] == 2


@pytest.mark.parametrize("recover", [False, True])
async def test_integrity_failure_quality_retry_and_exclusion(tmp_path, recover):
    source = "There are 82 copies."
    row = {"id": "a", "problem": source, "original": {"problem": source}}
    data = {"train": [row]}
    registry = {"train": {"role": "train", "translate": ["problem"]}}
    report = {"datasets": {"train": {"removed": []}}, "output_digests": {"train": digest([row])}}
    calls = []

    def handler(request):
        calls.append(1)
        masked = json.loads(request.content)["messages"][1]["content"]
        return response("한국어 " + masked if recover and len(calls) == 2 else "복사본 8^2개")

    result = await run_model(
        config_for(tmp_path),
        registry,
        data,
        data,
        report,
        MODELS[0],
        True,
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    assert len(calls) == 2
    assert len(result["accepted"]["train"]) == int(recover)
    checks = result["checks"]["datasets"]["train"]
    assert checks["retried_count"] == 1 and checks["flagged_count"] == int(not recover)
    if not recover:
        assert any("placeholder_integrity" in r for r in checks["reasons"])


async def test_truncation_splits_original_and_shields_children(tmp_path):
    calls = []

    def handler(request):
        masked = json.loads(request.content)["messages"][1]["content"]
        calls.append(masked)
        return (
            response("partial ⟪KEEP_", "length")
            if len(calls) == 1
            else response("한국어 " + masked)
        )

    config = model_config(config_for(tmp_path), MODELS[0], True)
    async with httpx.AsyncClient(
        base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler)
    ) as client:
        output, chunks = await DeepSeekTranslator(client, config, tmp_path / "usage.jsonl").chunk(
            "First 82\n\nSecond 92"
        )
    assert output == "한국어 First 82\n\n한국어 Second 92"
    assert len(chunks) == 2 and len(calls) == 3


def test_protection_version_changes_cache_folder(tmp_path, monkeypatch):
    import pipeline.deepseek_translation as module

    base = config_for(tmp_path)
    before = model_config(base, MODELS[0], True)["PROJECT_ROOT"]
    monkeypatch.setattr(module, "PROTECTION_VERSION", 999)
    assert model_config(base, MODELS[0], True)["PROJECT_ROOT"] != before
