import json
from pathlib import Path

import httpx
import pytest
import yaml

from common.io import read_jsonl
from pipeline.api_diagnostics import error_details, http_error_response, install_error_details
from pipeline.openai_translation import BASE_URL, OpenAITranslator, model_config


def test_recovers_suppressed_context_and_redacts_key():
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        headers={"Authorization": "Bearer private-key-value"},
    )
    response = httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "message": "Unsupported reasoning.effort: none; private-key-value sk-proj-EXAMPLE",
                "code": "unsupported_value",
                "param": "reasoning.effort",
                "type": "invalid_request_error",
            }
        },
        headers={"x-request-id": "req-test"},
    )
    try:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise RuntimeError("Earlier generic wrapper") from None
    except RuntimeError as exc:
        details = error_details(http_error_response(exc))
    assert details["param"] == "reasoning.effort" and details["request_id"] == "req-test"
    assert "private-key-value" not in str(details) and "sk-proj-EXAMPLE" not in str(details)
    assert details["message"].count("[REDACTED]") == 2


@pytest.mark.parametrize(
    "body",
    [b"SECRET HTML", b"[]", b'{"error": "SECRET"}', b'{"error": {"message": {"SECRET": 1}}}'],
)
def test_malformed_errors_do_not_dump_raw_body(body):
    assert "SECRET" not in str(error_details(httpx.Response(400, content=body)))


async def test_wrapper_preserves_cache_identity_payload_and_prints_actual_error(
    tmp_path, monkeypatch
):
    base = yaml.safe_load(Path("configs/translation_openai.yaml").read_text())
    base.update(PROJECT_ROOT=str(tmp_path), MAX_RETRIES=0)
    before = model_config(base, True)
    # Register restoration before the installer changes the class.
    monkeypatch.setattr(OpenAITranslator, "request", OpenAITranslator.request)
    install_error_details(OpenAITranslator)
    wrapped = OpenAITranslator.request
    install_error_details(OpenAITranslator)
    assert wrapped is OpenAITranslator.request
    assert model_config(base, True) == before

    def handler(request):
        assert json.loads(request.content)["input"] == "Original 82"
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid service_tier argument",
                    "param": "service_tier",
                    "code": "invalid_value",
                    "type": "invalid_request_error",
                }
            },
            headers={"x-request-id": "req-400"},
        )

    async with httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(RuntimeError, match="Invalid service_tier argument") as captured:
            await OpenAITranslator(client, before, tmp_path / "smoke").chunk("Original 82")
    assert "service_tier" in str(captured.value) and "req-400" in str(captured.value)
    log = read_jsonl(tmp_path / "smoke/api_errors.jsonl")
    assert len(log) == 1 and log[0]["param"] == "service_tier"
    assert "Original 82" not in str(log)
