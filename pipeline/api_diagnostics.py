"""Expose redacted HTTP error details without changing translation/cache identity."""

import json
import re
from functools import wraps

import httpx

from common.io import append_jsonl, digest, read_jsonl


def http_error_response(error):
    """Recover the HTTP cause even when an earlier wrapper used `from None`."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, httpx.HTTPStatusError):
            return error.response
        error = error.__cause__ or error.__context__
    return None


def error_details(response):
    """Whitelist useful API fields; never print headers or arbitrary response bodies."""
    secrets = []
    try:
        auth = response.request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            secrets.append(auth[7:])
    except RuntimeError:
        pass

    def redact(value):
        if value is None:
            return None
        if not isinstance(value, (str, int, float, bool)):
            return "[non-scalar error field omitted]"
        text = str(value)
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
        return text[:2000]

    try:
        payload = response.json()
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if not isinstance(error, dict):
            error = {}
    except (ValueError, UnicodeDecodeError):
        error = {}
    details = {"http_status": response.status_code}
    details.update({key: redact(error.get(key)) for key in ("message", "type", "code", "param")})
    details["request_id"] = redact(response.headers.get("x-request-id"))
    if not details["message"]:
        details["message"] = "No structured API error message returned; raw body omitted."
    return details


def install_error_details(translator_class):
    """Wrap diagnostics only, preserving existing model settings and cache folder."""
    original = translator_class.request
    if getattr(original, "_api_diagnostics_installed", False):
        return

    @wraps(original)
    async def request(self, text, feedback=None):
        try:
            return await original(self, text, feedback)
        except RuntimeError as exc:
            response = http_error_response(exc)
            if response is None:
                raise
            details = error_details(response)
            path = self.folder / "api_errors.jsonl"
            read_jsonl(path, repair_tail=True)
            append_jsonl(path, {**details, "source_digest": digest(text)})
            raise RuntimeError(
                f"OpenAI HTTP {response.status_code}: "
                + json.dumps(details, ensure_ascii=False)
                + " Completed chunks are saved."
            ) from None

    request._api_diagnostics_installed = True
    translator_class.request = request
