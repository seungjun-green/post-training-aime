import json
from pathlib import Path
from zipfile import ZipFile

import httpx
import pytest
import yaml

from common.io import digest, read_jsonl
from pipeline.deepseek_translation import (
    MODELS,
    DeepSeekTranslator,
    comparison_archive,
    model_config,
    run_model,
)
from pipeline.translation import TranslationFailure


def settings(tmp_path):
    config = yaml.safe_load(Path("configs/translation_deepseek.yaml").read_text())
    config.update(PROJECT_ROOT=str(tmp_path), SMOKE_TEST_N=1, MAX_RETRIES=1, BACKOFF_SECONDS=0)
    return config


def fixture_data():
    def row(identifier, text):
        return {"id": identifier, "problem": text, "answer": "42", "original": {"problem": text}}

    data = {
        "train": [row("a", "Find the answer 42."), row("b", "Compute the value 42.")],
        "eval": [row("c", "Determine the result 42.")],
    }
    registry = {n: {"role": n, "translate": ["problem"]} for n in data}
    report = {
        "datasets": {"train": {"removed": []}},
        "output_digests": {n: digest(rs) for n, rs in data.items()},
    }
    return registry, data, report


def answer(model="deepseek-flash", text="정답을 구하세요.", finish="stop", **message):
    return {
        "model": model,
        "id": "response-1",
        "system_fingerprint": "fp-test",
        "choices": [{"finish_reason": finish, "message": {"content": text, **message}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "prompt_cache_hit_tokens": 5},
    }


def make_client(handler):
    return httpx.AsyncClient(
        base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler)
    )


async def test_payload_whitespace_keepalive_and_usage_logging(tmp_path):
    sent = []

    def handler(request):
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        sent.append(json.loads(request.content))
        return httpx.Response(200, content="\n\n " + json.dumps(answer()))

    config = model_config(settings(tmp_path), MODELS[0], True)
    async with make_client(handler) as client:
        text, chunks = await DeepSeekTranslator(client, config, tmp_path / "usage.jsonl").chunk(
            "English"
        )
    assert text == "정답을 구하세요."
    assert sent[0]["thinking"] == {"type": "disabled"}
    assert sent[0]["max_tokens"] == 16000
    assert "seed" not in sent[0] and "chat_template_kwargs" not in sent[0]
    assert chunks[0]["stop_reason"] == "end_turn"
    usage = read_jsonl(tmp_path / "usage.jsonl")[0]
    assert usage["system_fingerprint"] == "fp-test"
    assert usage["usage"]["prompt_cache_hit_tokens"] == 5


async def test_length_discards_partial_and_accounts_all_responses(tmp_path):
    values = iter(
        [
            answer(text="TRUNCATED", finish="length"),
            answer(text="첫 문단"),
            answer(text="둘째 문단"),
        ]
    )
    config = model_config(settings(tmp_path), MODELS[0], True)
    async with make_client(lambda _: httpx.Response(200, json=next(values))) as client:
        output, chunks = await DeepSeekTranslator(client, config, tmp_path / "usage.jsonl").chunk(
            "First\n\nSecond"
        )
    assert output == "첫 문단\n\n둘째 문단" and len(chunks) == 2
    assert len(read_jsonl(tmp_path / "usage.jsonl")) == 3


async def test_usage_journal_recovers_interrupted_tail(tmp_path):
    usage_path = tmp_path / "usage.jsonl"
    usage_path.write_text('{"partial":')
    config = model_config(settings(tmp_path), MODELS[0], True)
    async with make_client(lambda _: httpx.Response(200, json=answer())) as client:
        await DeepSeekTranslator(client, config, usage_path).chunk("source")
    assert len(read_jsonl(usage_path)) == 1


@pytest.mark.parametrize(
    "message",
    [
        answer(finish="content_filter"),
        answer(refusal="refused"),
        answer(reasoning_content="thinking"),
        answer(text=""),
    ],
)
async def test_refusals_empty_and_reasoning_cannot_pass(tmp_path, message):
    config = model_config(settings(tmp_path), MODELS[0], True)
    async with make_client(lambda _: httpx.Response(200, json=message)) as client:
        with pytest.raises(TranslationFailure):
            await DeepSeekTranslator(client, config, tmp_path / "usage.jsonl").chunk("source")


@pytest.mark.parametrize("status,count", [(401, 1), (402, 1), (400, 1), (429, 2), (503, 2)])
async def test_http_errors_stop_without_leaking_body_or_key(tmp_path, status, count):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="SECRET should never be printed")

    config = model_config(settings(tmp_path), MODELS[0], True)
    async with make_client(handler) as client:
        with pytest.raises(RuntimeError) as error:
            await DeepSeekTranslator(client, config, tmp_path / "usage.jsonl").chunk("source")
    assert "SECRET" not in str(error.value)
    assert len(calls) == count


@pytest.mark.parametrize("finish", ["insufficient_system_resource", "aborted"])
async def test_interrupted_generation_retries_without_accepting_partial(tmp_path, finish):
    values = iter([answer(text="BAD PARTIAL", finish=finish), answer()])
    config = model_config(settings(tmp_path), MODELS[0], True)
    async with make_client(lambda _: httpx.Response(200, json=next(values))) as client:
        output, _ = await DeepSeekTranslator(client, config, tmp_path / "usage.jsonl").chunk(
            "source"
        )
    assert output == "정답을 구하세요."
    assert len(read_jsonl(tmp_path / "usage.jsonl")) == 2


async def test_both_smokes_full_selection_resume_and_secret_isolation(tmp_path):
    config = settings(tmp_path)
    registry, data, report = fixture_data()
    calls = []

    def handler(request):
        assert request.headers["Authorization"] == "Bearer SECRET"
        body = json.loads(request.content)
        calls.append(body["model"])
        return httpx.Response(
            200, json=answer(model=body["model"], text="한국어 " + body["messages"][1]["content"])
        )

    transport = httpx.MockTransport(handler)
    results = {}
    for model in MODELS:
        results[model] = await run_model(
            config, registry, data, data, report, model, True, "SECRET", transport=transport
        )
    assert len(calls) == 4
    assert results[MODELS[0]]["selected"] == results[MODELS[1]]["selected"]
    assert results[MODELS[0]]["folder"] != results[MODELS[1]]["folder"]
    archive = comparison_archive(results, registry, tmp_path)
    with ZipFile(archive) as z:
        page = z.read("smoke_comparison.html").decode()
        assert all(m in page for m in MODELS)
        assert all("SECRET" not in z.read(name).decode() for name in z.namelist())
    # Same smoke reruns make no new requests.
    await run_model(
        config, registry, data, data, report, MODELS[0], True, "SECRET", transport=transport
    )
    assert len(calls) == 4
    full = await run_model(
        config,
        registry,
        data,
        data,
        report,
        MODELS[1],
        False,
        "SECRET",
        reviewed=True,
        transport=transport,
    )
    assert calls == [MODELS[0]] * 2 + [MODELS[1]] * 3
    assert len(full["accepted"]["train"]) == 2
    assert full["manifest"]["config"]["TRANSLATION_MODEL"] == MODELS[1]


async def test_full_gate_requires_chosen_model_matching_settings(tmp_path):
    config = settings(tmp_path)
    registry, data, report = fixture_data()
    calls = []

    def handler(request):
        calls.append(1)
        body = json.loads(request.content)
        return httpx.Response(200, json=answer(text="한국어 " + body["messages"][1]["content"]))

    transport = httpx.MockTransport(handler)
    with pytest.raises(ValueError, match="Review"):
        await run_model(
            config, registry, data, data, report, MODELS[0], False, "SECRET", transport=transport
        )
    with pytest.raises(ValueError, match="smoke test"):
        await run_model(
            config,
            registry,
            data,
            data,
            report,
            MODELS[0],
            False,
            "SECRET",
            reviewed=True,
            transport=transport,
        )
    assert not calls
    await run_model(
        config, registry, data, data, report, MODELS[0], True, "SECRET", transport=transport
    )
    with pytest.raises(ValueError, match="smoke test"):
        await run_model(
            config,
            registry,
            data,
            data,
            report,
            MODELS[1],
            False,
            "SECRET",
            reviewed=True,
            transport=transport,
        )
    config["TEMPERATURE"] = 0.8
    with pytest.raises(ValueError, match="smoke test"):
        await run_model(
            config,
            registry,
            data,
            data,
            report,
            MODELS[0],
            False,
            "SECRET",
            reviewed=True,
            transport=transport,
        )
    assert len(calls) == 2
