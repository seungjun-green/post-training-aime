import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from common.io import append_jsonl
from pipeline.qwen_runtime import server_command
from pipeline.qwen_translation import (
    QwenTranslator,
    configure_run,
    require_smoke_review,
    write_smoke_review,
)
from pipeline.translation import TranslationFailure, check_all, prepare_run, translate_all


def settings(tmp_path):
    config = yaml.safe_load(Path("configs/translation_qwen.yaml").read_text())
    config.update(
        PROJECT_ROOT=str(tmp_path), MODEL_REVISION="a" * 40, BACKOFF_SECONDS=0, MAX_RETRIES=1
    )
    return config


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return list(range(len(messages[-1]["content"])))


def response(output="정답 42를 구하세요.", finish="stop", **message):
    return {
        "model": "Qwen/Qwen3-32B",
        "choices": [{"finish_reason": finish, "message": {"content": output, **message}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 10},
    }


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://localhost")


async def test_local_request_and_retry_seed_provenance(tmp_path):
    bodies = []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response())

    async with client(handler) as http:
        translator = QwenTranslator(http, settings(tmp_path), Tokenizer())
        row = {"id": "1", "problem": "Find answer 42.", "original": {"problem": "Find answer 42."}}
        spec = {"translate": ["problem"]}
        first = await translator.row(row, spec)
        await translator.row(row, spec)
        await translator.row(row, spec, attempt=2)
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert bodies[0]["seed"] == bodies[1]["seed"] != bodies[2]["seed"]
    assert bodies[0]["max_tokens"] == 16000
    assert bodies[0]["temperature"] == 0.7
    assert first["model_revision"] == "a" * 40
    assert first["chunks"]["problem"][0]["stop_reason"] == "end_turn"


async def test_truncation_discarded_and_paragraphs_retranslated(tmp_path):
    responses = iter([response("TRUNCATED", "length"), response("첫 문단"), response("둘째 문단")])
    async with client(lambda _: httpx.Response(200, json=next(responses))) as http:
        output, chunks = await QwenTranslator(http, settings(tmp_path), Tokenizer()).chunk(
            "first paragraph\n\nsecond paragraph"
        )
    assert output == "첫 문단\n\n둘째 문단"
    assert len(chunks) == 2


async def test_context_guard_splits_before_sending_and_never_truncates(tmp_path):
    config = settings(tmp_path)
    config.update(MAX_OUTPUT_TOKENS=10, MAX_MODEL_LEN=30)
    sent = []

    def handler(request):
        text = json.loads(request.content)["messages"][-1]["content"]
        sent.append(text)
        return httpx.Response(200, json=response())

    async with client(handler) as http:
        translator = QwenTranslator(http, config, Tokenizer())
        await translator.chunk("first paragraph\n\nsecond paragraph")
        assert sent == ["first paragraph", "second paragraph"]
        with pytest.raises(TranslationFailure, match="no safe paragraph"):
            await translator.chunk("x" * 40)
    assert len(sent) == 2


@pytest.mark.parametrize(
    "answer",
    [
        response("", "content_filter"),
        response("<think>unexpected</think>"),
        response("", refusal="blocked"),
        response(reasoning_content="unexpected"),
        response(""),
    ],
)
async def test_failed_responses_never_accepted(tmp_path, answer):
    async with client(lambda _: httpx.Response(200, json=answer)) as http:
        with pytest.raises(TranslationFailure):
            await QwenTranslator(http, settings(tmp_path), Tokenizer()).chunk("text")


@pytest.mark.parametrize("status, calls", [(400, 1), (401, 1), (500, 2), (429, 2)])
async def test_infrastructure_failure_stops_run(tmp_path, status, calls):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, text="server unavailable")

    async with client(handler) as http:
        with pytest.raises(RuntimeError):
            await QwenTranslator(http, settings(tmp_path), Tokenizer()).chunk("text")
    assert len(requests) == calls


def test_pinned_revision_and_settings_isolation(tmp_path):
    config = settings(tmp_path)
    calls = []

    class API:
        def model_info(self, *args, **kwargs):
            calls.append(1)
            return SimpleNamespace(sha="b" * 40)

    first = configure_run(config, api=API())
    assert first["MODEL_REVISION"] == "b" * 40
    assert Path(first["PROJECT_ROOT"]).parent == tmp_path / "runs"
    # A full run/resume/decontamination change retains compatible Qwen outputs.
    config.update(
        SMOKE_TEST=False, SMOKE_REVIEWED=True, MAX_CONCURRENCY=1, DECONTAM_COVERAGE_THRESHOLD=0.8
    )
    assert configure_run(config, api=API())["PROJECT_ROOT"] == first["PROJECT_ROOT"]
    assert len(calls) == 1
    config["TEMPERATURE"] = 0.3
    assert configure_run(config, api=API())["PROJECT_ROOT"] != first["PROJECT_ROOT"]
    assert not (tmp_path / "smoke_test").exists()


async def test_smoke_full_gate_and_resume_reuse(tmp_path):
    config = settings(tmp_path)
    rows = {
        "test": [
            {"id": "1", "problem": "Find answer 42.", "original": {"problem": "Find answer 42."}}
        ]
    }
    registry = {"test": {"translate": ["problem"]}}
    config.update(SMOKE_TEST=False)
    with pytest.raises(ValueError, match="Review"):
        require_smoke_review(config)
    config["SMOKE_REVIEWED"] = True
    require_smoke_review(config)
    with pytest.raises(ValueError, match="smoke test"):
        prepare_run(config, registry, rows)
    config["SMOKE_TEST"] = True
    selected, folder, manifest = prepare_run(config, registry, rows)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=response())

    async with client(handler) as http:
        translator = QwenTranslator(http, config, Tokenizer())
        await translate_all(selected, registry, folder, translator)
        await translate_all(selected, registry, folder, translator)
        accepted, report = await check_all(selected, registry, folder, manifest, translator)
    assert len(calls) == 1 and report["datasets"]["test"]["accepted"] == 1
    config["SMOKE_TEST"] = False
    assert prepare_run(config, registry, rows)[1].name == "translations"


def test_review_escapes_source_and_server_is_local(tmp_path):
    config = settings(tmp_path)
    row = {
        "id": "<id>",
        "problem": "<script>bad</script>",
        "original": {"problem": "<script>bad</script>"},
    }
    append_jsonl(tmp_path / "test.jsonl", {"id": "<id>", "translations": {"ko_problem": "번역"}})
    review = write_smoke_review({"test": [row]}, {"test": {"translate": ["problem"]}}, tmp_path)
    assert "<script>" not in review.read_text()
    assert "&lt;script&gt;" in review.read_text()
    command = server_command("python", config)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--revision") + 1] == "a" * 40
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "--trust-remote-code" not in command


def test_qwen_notebook_uses_separate_root_and_smoke_default():
    import nbformat

    nb = nbformat.read("notebooks/translate_datasets_qwen.ipynb", as_version=4)
    config = nb.cells[2].source
    assert "SMOKE_TEST = True" in config and "SMOKE_REVIEWED = False" in config
    assert "LG-Korea-AIME-Qwen" in config
    all_code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "ANTHROPIC_API_KEY" not in all_code and "OPENAI_API_KEY" not in all_code
    assert "require_smoke_review(CONFIG)" in all_code
