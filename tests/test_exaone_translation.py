import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from common.io import digest, read_jsonl, write_json
from pipeline.exaone_runtime import server_command
from pipeline.exaone_translation import MODEL, ExaoneTranslator, model_config, pin_model, run_model
from pipeline.translation import TranslationFailure


def settings(tmp_path, **extra):
    c = yaml.safe_load(Path("configs/translation_exaone.yaml").read_text())
    c.update(
        PROJECT_ROOT=str(tmp_path),
        MODEL_REVISION="a" * 40,
        MAX_RETRIES=0,
        SMOKE_TEST_N=1,
        BACKOFF_SECONDS=0,
    )
    c.update(extra)
    return c


def reply(text, finish="stop", **message):
    return {
        "model": MODEL,
        "id": "local-test",
        "choices": [{"finish_reason": finish, "message": {"content": text, **message}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def client(handler):
    def wrapped(request):
        assert request.url.host == "127.0.0.1"
        assert "authorization" not in request.headers
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            return httpx.Response(200, json={"count": 100})
        return handler(request)

    return httpx.AsyncClient(
        base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(wrapped)
    )


async def test_unmasked_local_translation_chunk_retry_and_restart_cache(tmp_path):
    seen = []
    source = r"There are 82 copies of $S_0$. Find \boxed{92}."

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["model"] == MODEL
        assert body["messages"][1]["content"] == source
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        text = source.replace("82", "8²") if len(seen) == 1 else "한국어 " + source
        return httpx.Response(200, json=reply(text))

    config = model_config(settings(tmp_path), True)
    async with client(handler) as session:
        first = ExaoneTranslator(session, config, tmp_path / "smoke")
        output, metadata = await first.chunk(source)
        assert output == "한국어 " + source and metadata[0]["quality_attempt"] == 2
        restarted = ExaoneTranslator(session, config, tmp_path / "smoke")
        assert (await restarted.chunk(source))[0] == output
    assert len(seen) == 2
    assert "numbers_changed" in seen[1]["messages"][0]["content"]
    assert len(read_jsonl(tmp_path / "smoke/failed_chunks.jsonl")) == 1


async def test_truncation_discards_partial_and_translates_subchunks(tmp_path):
    def handler(request):
        text = json.loads(request.content)["messages"][1]["content"]
        return httpx.Response(
            200, json=reply("partial", "length") if "\n\n" in text else reply("한국어 " + text)
        )

    async with client(handler) as session:
        translator = ExaoneTranslator(session, settings(tmp_path), tmp_path / "smoke")
        result, chunks = await translator.chunk("Find 82.\n\nFind 92.")
    assert result == "한국어 Find 82.\n\n한국어 Find 92." and len(chunks) == 2


async def test_context_budget_splits_before_generation(tmp_path):
    generated = []

    def handler(request):
        body = json.loads(request.content)
        text = body["messages"][1]["content"]
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 17000 if "\n\n" in text else 100})
        generated.append(text)
        return httpx.Response(200, json=reply("한국어 " + text))

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handler)
    ) as session:
        await ExaoneTranslator(session, settings(tmp_path), tmp_path / "smoke").chunk("82\n\n92")
    assert generated == ["82", "92"]


@pytest.mark.parametrize("kind", ["reasoning", "refusal", "http", "model"])
async def test_errors_not_accepted(tmp_path, kind):
    def handler(request):
        if kind == "http":
            return httpx.Response(500)
        result = reply(
            "한국어 82",
            **(
                {"reasoning_content": "thinking"}
                if kind == "reasoning"
                else {"refusal": "refused"}
                if kind == "refusal"
                else {}
            ),
        )
        if kind == "model":
            result["model"] = "wrong"
        return httpx.Response(200, json=result)

    async with client(handler) as session:
        with pytest.raises(RuntimeError if kind in {"http", "model"} else TranslationFailure):
            await ExaoneTranslator(session, settings(tmp_path), tmp_path / "smoke").chunk("82")
    assert not read_jsonl(tmp_path / "chunk_cache.jsonl")


def test_pins_and_settings_identity(tmp_path):
    calls = []

    def info(model, revision):
        calls.append((model, revision))
        return SimpleNamespace(sha="b" * 40)

    base = settings(tmp_path, MODEL_REVISION=None)
    c = pin_model(base, api=SimpleNamespace(model_info=info))
    assert pin_model(base, api=object()) == c and len(calls) == 1
    smoke = model_config(c, True)
    assert model_config(c, False)["PROJECT_ROOT"] == smoke["PROJECT_ROOT"]
    for key, value in [
        ("MODEL_REVISION", "c" * 40),
        ("TEMPERATURE", 0.7),
        ("MAX_OUTPUT_TOKENS", 8000),
    ]:
        assert model_config(dict(c, **{key: value}), True)["PROJECT_ROOT"] != smoke["PROJECT_ROOT"]


def test_runtime_is_pinned_local_text_only(tmp_path):
    c = settings(tmp_path)
    cmd = server_command("/gpu/bin/python", c)
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--revision") + 1] == "a" * 40
    assert cmd[cmd.index("--limit-mm-per-prompt") + 1] == '{"image": 0}'
    assert cmd[cmd.index("--dtype") + 1] == "bfloat16"
    assert "--trust-remote-code" not in cmd


async def test_full_requires_matching_smoke_then_reuses_rows(tmp_path):
    c = settings(tmp_path)
    write_json(
        tmp_path / "server_identity.json",
        {k: c[k] for k in ["TRANSLATION_MODEL", "MODEL_REVISION", "MAX_MODEL_LEN"]},
    )
    rows = [
        {"id": str(i), "problem": text, "original": {"problem": text}}
        for i, text in enumerate(["Find 82.", "Find 92."])
    ]
    data = {"train": rows}
    registry = {"train": {"role": "train", "translate": ["problem"]}}
    report = {"datasets": {"train": {"removed": []}}, "output_digests": {"train": digest(rows)}}
    requests = []

    def handler(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        text = json.loads(request.content)["messages"][1]["content"]
        requests.append(text)
        return httpx.Response(200, json=reply("한국어 " + text))

    transport = httpx.MockTransport(handler)
    with pytest.raises(ValueError):
        await run_model(c, registry, data, data, report, False, reviewed=False, transport=transport)
    with pytest.raises((ValueError, FileNotFoundError)):
        await run_model(c, registry, data, data, report, False, reviewed=True, transport=transport)
    smoke = await run_model(c, registry, data, data, report, True, transport=transport)
    full = await run_model(
        c, registry, data, data, report, False, reviewed=True, transport=transport
    )
    assert len(requests) == 2 and len(full["accepted"]["train"]) == 2
    assert smoke["config"]["PROJECT_ROOT"] == full["config"]["PROJECT_ROOT"]
    assert (full["folder"] / "human_review.html").exists()
