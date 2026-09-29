import json
from types import SimpleNamespace

import pytest
import yaml

from common.io import digest, latest_by_id
from pipeline.translation import (
    TranslationFailure,
    Translator,
    bisect_chunk,
    check_all,
    check_row,
    prepare_run,
    select_smoke,
    split_chunks,
    translate_all,
)


def settings(tmp_path):
    config = yaml.safe_load(open("configs/translation.yaml"))
    config.update(PROJECT_ROOT=str(tmp_path), SMOKE_TEST_N=1, MAX_CONCURRENCY=2)
    return config


def sample():
    return {
        "id": "id-1",
        "problem": "Find the answer 42.",
        "answer": "42",
        "original": {"problem": "Find the answer 42."},
    }


def test_chunks_respect_math_and_reconstruct():
    text = "First paragraph.\n\n$$x\n\n+ y$$\n\nLast paragraph."
    chunks = split_chunks(text, 20)
    assert all(len(c) <= 20 for c in chunks)
    assert "$$x\n\n+ y$$" in chunks
    assert "\n\n".join(chunks) == text
    assert "\n\n".join(bisect_chunk(text)) == text
    with pytest.raises(TranslationFailure):
        split_chunks("$$" + "x" * 50 + "$$", 20)
    with pytest.raises(TranslationFailure):
        bisect_chunk("one indivisible paragraph")


class FakeTranslator(Translator):
    def __init__(self, config, answers):
        super().__init__(None, config)
        self.answers = iter(answers)
        self.calls = []

    async def request(self, text):
        self.calls.append(text)
        value, stop = next(self.answers)
        return SimpleNamespace(
            stop_reason=stop,
            content=[SimpleNamespace(type="text", text=value)],
            model="fake-test-only",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


async def test_max_tokens_discards_output_and_resplits(tmp_path):
    translator = FakeTranslator(
        settings(tmp_path),
        [("TRUNCATED", "max_tokens"), ("첫 문단", "end_turn"), ("둘째 문단", "end_turn")],
    )
    output, chunks = await translator.chunk("first paragraph\n\nsecond paragraph")
    assert output == "첫 문단\n\n둘째 문단"
    assert len(chunks) == 2 and all(c["stop_reason"] == "end_turn" for c in chunks)
    assert len(translator.calls) == 3


async def test_resume_and_one_quality_retry(tmp_path):
    config = settings(tmp_path)
    registry = {"test": {"translate": ["problem"]}}
    rows = {"test": [sample()]}
    folder = tmp_path / "smoke_test"
    translator = FakeTranslator(
        config, [("English unchanged", "end_turn"), ("정답 42를 구하세요.", "end_turn")]
    )
    await translate_all(rows, registry, folder, translator)
    await translate_all(rows, registry, folder, translator)
    assert len(translator.calls) == 1
    manifest = {"mode": "smoke_test", "signature": "test"}
    accepted, report = await check_all(rows, registry, folder, manifest, translator)
    assert report["datasets"]["test"]["flagged_count"] == 0
    assert report["datasets"]["test"]["retried_count"] == 1
    assert len(accepted["test"]) == 1
    assert latest_by_id(folder / "test.jsonl")["id-1"]["attempt"] == 2
    await check_all(rows, registry, folder, manifest, translator)
    assert len(translator.calls) == 2


async def test_persistent_failure_excluded_and_reported(tmp_path):
    config = settings(tmp_path)
    translator = FakeTranslator(config, [("English unchanged", "end_turn")] * 2)
    registry, rows = {"test": {"translate": ["problem"]}}, {"test": [sample()]}
    folder = tmp_path / "smoke_test"
    await translate_all(rows, registry, folder, translator)
    accepted, report = await check_all(
        rows, registry, folder, {"mode": "smoke_test", "signature": "x"}, translator
    )
    assert accepted == {"test": []}
    assert report["datasets"]["test"]["flagged_count"] == 1
    assert (tmp_path / "checks/test_flagged.jsonl").exists()


def test_boxed_answer_check_is_exact_and_nested(tmp_path):
    config = settings(tmp_path)
    row = {"id": "x", "original": {"deepseek_attempt": r"Answer is \boxed{\frac{1}{2}}"}}
    record = {
        "translations": {"ko_deepseek_attempt": r"정답은 \boxed{\frac{2}{4}}"},
        "source_digest": digest(row),
        "chunks": {"deepseek_attempt": [{"stop_reason": "end_turn"}]},
    }
    assert "deepseek_attempt: final_answer_not_preserved" in check_row(
        row, record, {"translate": ["deepseek_attempt"]}, config, "s1k_1.1"
    )


def test_full_run_requires_matching_completed_smoke(tmp_path):
    config = settings(tmp_path)
    registry = {"test": {"translate": ["problem"]}}
    data = {"test": [sample()]}
    config["SMOKE_TEST"] = False
    with pytest.raises(ValueError, match="smoke test"):
        prepare_run(config, registry, data)
    config["SMOKE_TEST"] = True
    _, folder, manifest = prepare_run(config, registry, data)
    manifest["checks_completed"] = True
    (folder / "manifest.json").write_text(json.dumps(manifest))
    config["SMOKE_TEST"] = False
    assert prepare_run(config, registry, data)[1].name == "translations"
    config["TRANSLATION_MODEL"] = "changed"
    with pytest.raises(ValueError, match="same sources/model"):
        prepare_run(config, registry, data)


def test_smoke_includes_longest_retained_trace(tmp_path):
    config = settings(tmp_path)
    config.update(CHUNK_CHARS=20, SMOKE_TEST_N=2)
    rows = [
        {"id": i, "original": {"deepseek_thinking_trajectory": "x" * length}}
        for i, length in enumerate([10, 50, 30])
    ]
    a = select_smoke({"s1k_1.1": rows}, config)
    assert a == select_smoke({"s1k_1.1": rows}, config)
    assert a["s1k_1.1"][0]["id"] == 1


async def test_api_retry_policy_and_concurrency(tmp_path, monkeypatch):
    import asyncio

    import httpx
    from anthropic import AuthenticationError, RateLimitError

    config = settings(tmp_path)
    config.update(BACKOFF_SECONDS=0, MAX_RETRIES=2)
    calls = []
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))

    class Stream:
        async def __aenter__(self):
            calls.append(1)
            if len(calls) < 3:
                raise RateLimitError("rate limit", response=response, body=None)
            return self

        async def __aexit__(self, *args):
            return False

        async def get_final_message(self):
            return "success"

    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: Stream()))
    assert await Translator(client, config).request("text") == "success"
    assert len(calls) == 3

    def unauthorized(**kwargs):
        raise AuthenticationError(
            "bad key", response=httpx.Response(401, request=response.request), body=None
        )

    client.messages.stream = unauthorized
    with pytest.raises(AuthenticationError):
        await Translator(client, config).request("text")

    active, peak = 0, 0

    class ConcurrentStream(Stream):
        async def __aenter__(self):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.001)
            return self

        async def __aexit__(self, *args):
            nonlocal active
            active -= 1

    client.messages.stream = lambda **kw: ConcurrentStream()
    translator = Translator(client, config)
    await asyncio.gather(*(translator.request("x") for _ in range(10)))
    assert peak == 2
