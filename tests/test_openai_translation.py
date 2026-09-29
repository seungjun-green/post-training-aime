import ast
import json
from pathlib import Path
from zipfile import ZipFile

import httpx
import nbformat
import pytest
import yaml

from common.io import digest, read_jsonl
from pipeline.openai_translation import BASE_URL, MODEL, OpenAITranslator, model_config, run_model
from pipeline.publish import archive_outputs
from pipeline.text_translation_checks import hard_checks, literal_sequence
from pipeline.translation import TranslationFailure


def settings(tmp_path, **overrides):
    config = yaml.safe_load(Path("configs/translation_openai.yaml").read_text())
    config.update(PROJECT_ROOT=str(tmp_path), SMOKE_TEST_N=1, MAX_RETRIES=0, BACKOFF_SECONDS=0)
    config.update(overrides)
    return config


def reply(text="한국어", status="completed", reason=None, content=None, **extras):
    return {
        "id": "resp-test",
        "model": MODEL,
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content
                if content is not None
                else [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
        **extras,
    }


def client(handler):
    return httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))


def fixture_data(name="train"):
    rows = [
        {"id": str(i), "problem": text, "original": {"problem": text}}
        for i, text in enumerate(["Find 82.", "Compute 92."])
    ]
    data = {name: rows}
    reg = {name: {"role": "train", "translate": ["problem"]}}
    report = {"datasets": {name: {"removed": []}}, "output_digests": {name: digest(rows)}}
    return data, reg, report


async def test_exact_model_unmasked_payload_and_response_usage(tmp_path):
    source = r"There are 82 copies of $S_0$. Find \boxed{92}."

    def handler(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.6-sol" and body["input"] == source
        assert body["reasoning"] == {"effort": "none"} and body["store"] is False
        assert "temperature" not in body and "KEEP_" not in str(body)
        return httpx.Response(200, json=reply("한국어 " + source))

    config = model_config(settings(tmp_path), True)
    async with client(handler) as session:
        output, chunks = await OpenAITranslator(session, config, tmp_path / "smoke").chunk(source)
    assert output == "한국어 " + source and chunks[0]["response_model"] == MODEL
    usage = read_jsonl(tmp_path / "smoke/api_usage.jsonl")
    assert len(usage) == 1 and usage[0]["usage"]["total_tokens"] == 30


@pytest.mark.parametrize("target", ["8^2개", "8²개", "82개와 82개"])
def test_numeric_mutations_fail(target):
    assert any("numbers_changed" in reason for reason in hard_checks("82 copies", target))


def test_number_multisets_allow_order_and_translate_ordinals():
    assert not hard_checks(
        "7th and 8th grades have 520 and 650 pupils.", "650명과 520명이 8학년과 7학년에 있다."
    )
    assert hard_checks("-82", "82")
    assert hard_checks("zero", "0")


def test_latex_exact_order_and_final_boxed_answer():
    source = r"Find $x$ then \[y\] and \frac{m}{n}; \boxed{\frac{1}{2}}."
    assert literal_sequence(source) == [r"$x$", r"\[y\]", r"\frac{m}{n}", r"\boxed{\frac{1}{2}}"]
    assert not hard_checks(source, "한국어 " + source)
    assert hard_checks(source, source.replace("$x$", r"\(x\)"))
    assert hard_checks(r"$x$ before $y$", r"$y$ 다음 $x$")
    assert "final_boxed_answer_changed" in hard_checks(r"\boxed{1}", r"\boxed{2}")
    assert hard_checks("[asy]draw((0,0)--(1,1));[/asy]", "[asy]draw((0,0)--(1,1))[/asy]")


async def test_only_failed_chunk_retries_with_original_source_and_feedback(tmp_path):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        source = body["input"]
        output = "8^2개" if source == "Second 82" and len(calls) == 2 else "한국어 " + source
        return httpx.Response(200, json=reply(output))

    config = model_config(settings(tmp_path), True)
    async with client(handler) as session:
        tr = OpenAITranslator(session, config, tmp_path / "smoke")
        await tr.chunk("First 92")
        output, chunks = await tr.chunk("Second 82")
        await tr.chunk("First 92")
    assert [c["input"] for c in calls] == ["First 92", "Second 82", "Second 82"]
    assert "numbers_changed" in calls[-1]["instructions"]
    assert output == "한국어 Second 82" and chunks[0]["quality_attempt"] == 2
    failed = read_jsonl(tmp_path / "smoke/failed_chunks.jsonl")
    assert len(failed) == 1 and failed[0]["response"]["output"][0]["content"][0]["text"] == "8^2개"


async def test_truncation_discard_and_resume_after_interruption(tmp_path):
    config = model_config(settings(tmp_path), True)
    calls = []

    def handler(request):
        source = json.loads(request.content)["input"]
        calls.append(source)
        if "\n\n" in source:
            return httpx.Response(200, json=reply("BAD PARTIAL", "incomplete", "max_output_tokens"))
        if source == "Second 92" and calls.count(source) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=reply("한국어 " + source))

    async with client(handler) as session:
        with pytest.raises(RuntimeError, match="retries exhausted"):
            await OpenAITranslator(session, config, tmp_path / "smoke").chunk(
                "First 82\n\nSecond 92"
            )
        # A new translator simulates a restarted notebook runtime.
        output, chunks = await OpenAITranslator(session, config, tmp_path / "smoke").chunk(
            "First 82\n\nSecond 92"
        )
    assert output == "한국어 First 82\n\n한국어 Second 92" and len(chunks) == 2
    assert calls.count("First 82") == 1
    assert "BAD PARTIAL" not in output


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 503])
async def test_service_errors_stop_and_do_not_expose_secrets(tmp_path, status):
    async with client(lambda _: httpx.Response(status, text="SECRET")) as session:
        with pytest.raises(RuntimeError) as error:
            await OpenAITranslator(
                session, model_config(settings(tmp_path), True), tmp_path / "smoke"
            ).chunk("source")
    assert "SECRET" not in str(error.value)


async def test_quota_failure_stops_without_backoff_retries(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})

    async with client(handler) as session:
        with pytest.raises(RuntimeError, match="HTTP 429"):
            await OpenAITranslator(
                session, model_config(settings(tmp_path, MAX_RETRIES=6), True), tmp_path / "smoke"
            ).chunk("source")
    assert len(calls) == 1


async def test_rate_limit_retries_then_succeeds(tmp_path):
    values = iter(
        [
            httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}}),
            httpx.Response(200, json=reply()),
        ]
    )
    async with client(lambda _: next(values)) as session:
        result, _ = await OpenAITranslator(
            session, model_config(settings(tmp_path, MAX_RETRIES=1), True), tmp_path / "smoke"
        ).chunk("source")
    assert result == "한국어"


@pytest.mark.parametrize(
    "data",
    [
        reply(content=[{"type": "refusal", "refusal": "Cannot translate"}]),
        reply("partial", "incomplete", "content_filter"),
        reply(""),
    ],
)
async def test_refusal_or_incomplete_is_never_accepted(tmp_path, data):
    async with client(lambda _: httpx.Response(200, json=data)) as session:
        with pytest.raises(TranslationFailure):
            await OpenAITranslator(
                session, model_config(settings(tmp_path), True), tmp_path / "smoke"
            ).chunk("source")
    assert read_jsonl(tmp_path / "smoke/failed_chunks.jsonl")


async def test_soft_flags_do_not_retry_or_exclude_and_full_reuses_smoke(tmp_path):
    data, reg, report = fixture_data("aime_2025")
    calls = []

    def handler(request):
        assert request.headers["Authorization"] == "Bearer SECRET"
        source = json.loads(request.content)["input"]
        calls.append(source)
        # Intentionally no Hangul and extreme length; numbers still intact.
        return httpx.Response(200, json=reply(source + " wording" * 20))

    base = settings(tmp_path)
    transport = httpx.MockTransport(handler)
    smoke = await run_model(base, reg, data, data, report, True, "SECRET", transport=transport)
    checks = smoke["checks"]["datasets"]["aime_2025"]
    assert checks["accepted"] == 1 and checks["review_count"] == 1 and checks["retried_count"] == 0
    await run_model(base, reg, data, data, report, True, "SECRET", transport=transport)
    assert len(calls) == 1
    full = await run_model(
        base, reg, data, data, report, False, "SECRET", reviewed=True, transport=transport
    )
    assert len(calls) == 2 and len(full["accepted"]["aime_2025"]) == 2
    queue = read_jsonl(full["folder"] / "human_review_queue.jsonl")
    assert len(queue) == 2 and all(r["manual_eval_review"] and r["accepted"] for r in queue)
    archive = archive_outputs(full["config"]["PROJECT_ROOT"], "translations")
    with ZipFile(archive) as z:
        assert "translations/human_review.html" in z.namelist()
        assert all("SECRET" not in z.read(name).decode() for name in z.namelist())
    assert full["usage"]["responses_recorded"] == 1


async def test_hard_failure_excluded_without_whole_row_retry(tmp_path):
    data, reg, report = fixture_data()
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=reply("8^2개"))

    result = await run_model(
        settings(tmp_path),
        reg,
        data,
        data,
        report,
        True,
        "key",
        transport=httpx.MockTransport(handler),
    )
    assert len(calls) == 2 and not result["accepted"]["train"]
    checks = result["checks"]["datasets"]["train"]
    assert checks["flagged_count"] == checks["retried_count"] == 1
    assert not any("hangul_missing" in r for r in checks["reasons"])


async def test_full_requires_matching_completed_smoke_before_api(tmp_path):
    data, reg, report = fixture_data()

    async def run(base, reviewed):
        return await run_model(base, reg, data, data, report, False, "key", reviewed=reviewed)

    with pytest.raises(ValueError, match="Review"):
        await run(settings(tmp_path), False)
    with pytest.raises(ValueError, match="smoke test"):
        await run(settings(tmp_path), True)
    await run_model(
        settings(tmp_path),
        reg,
        data,
        data,
        report,
        True,
        "key",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=reply("한국어 " + json.loads(r.content)["input"]))
        ),
    )
    with pytest.raises(ValueError, match="smoke test"):
        await run(settings(tmp_path, REASONING_EFFORT="low"), True)


def test_no_model_substitution_or_incompatible_cache_reuse(tmp_path):
    with pytest.raises(ValueError, match="no fallback"):
        model_config(settings(tmp_path, TRANSLATION_MODEL="gpt-5.6"), True)
    assert (
        model_config(settings(tmp_path), True)["PROJECT_ROOT"]
        != model_config(settings(tmp_path, HARD_CHECK_RETRIES=2), True)["PROJECT_ROOT"]
    )


async def test_notebook_full_cell_off_and_key_from_userdata():
    nb = nbformat.read("notebooks/translate_datasets_openai.ipynb", as_version=4)
    full = next(
        c.source for c in nb.cells if c.cell_type == "code" and "RUN_FULL_TRANSLATION =" in c.source
    )
    scope = {}
    await eval(compile(full, "full-cell", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), scope)
    assert scope["full_result"] is None
    all_code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "userdata.get('OPENAI_API_KEY')" in all_code and "UPLOAD_DATASETS = False" in all_code
    assert "DEEPSEEK_API_KEY" not in all_code and "smoke=True" in all_code
