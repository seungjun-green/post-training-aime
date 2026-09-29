"""Local Qwen adapter; retain the shared journals, chunking and quality checks."""

import asyncio
import html
import json
import re
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import httpx

from common.io import digest, latest_by_id, write_json
from pipeline.datasets import source_fields
from pipeline.translation import SYSTEM_PROMPT, TranslationFailure, Translator, bisect_chunk


def configure_run(config, token=None, api=None):
    """Pin model revisions and isolate incompatible generation settings in Drive."""
    from huggingface_hub import HfApi

    config = dict(config)
    base = Path(config["PROJECT_ROOT"])
    requested = config["MODEL_REVISION"] or "main"
    pin = base / "model_pins" / (digest([config["TRANSLATION_MODEL"], requested]) + ".json")
    if pin.exists():
        revision = json.loads(pin.read_text())["revision"]
    else:
        revision = (
            (api or HfApi(token=token))
            .model_info(config["TRANSLATION_MODEL"], revision=requested)
            .sha
        )
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("The model revision must resolve to an immutable HF commit")
        write_json(pin, {"model": config["TRANSLATION_MODEL"], "revision": revision})
    config["MODEL_REVISION"] = revision
    keys = [
        "TRANSLATION_MODEL",
        "MODEL_REVISION",
        "MAX_OUTPUT_TOKENS",
        "CHUNK_CHARS",
        "SEED",
        "TEMPERATURE",
        "TOP_P",
        "TOP_K",
        "MAX_MODEL_LEN",
        "LENGTH_RATIO_MIN",
        "LENGTH_RATIO_MAX",
    ]
    project = Path(__file__).resolve().parents[1]
    identity = {
        "settings": {k: config[k] for k in keys},
        "enable_thinking": False,
        "dtype": "bfloat16",
        "implementation": {
            str(p.relative_to(project)): digest(p.read_text())
            for p in [
                Path(__file__),
                project / "pipeline/qwen_runtime.py",
                project / "pipeline/translation.py",
                project / "common/math_text.py",
                project / "requirements-eval.lock",
            ]
        },
    }
    # Mode, concurrency and decontamination settings do not change translation identity.
    root = base / "runs" / digest(identity)[:20]
    config["PROJECT_ROOT"] = str(root)
    config["TRANSLATION_BACKEND"] = "local-vllm-qwen"
    write_json(root / "qwen_runtime.json", identity)
    write_json(root / "translation_config.json", config)
    return config


def require_smoke_review(config):
    if not config["SMOKE_TEST"] and not config["SMOKE_REVIEWED"]:
        raise ValueError("Review smoke_review.html, then set SMOKE_REVIEWED=True for the full run")


class QwenTranslator(Translator):
    def __init__(self, client, config, tokenizer):
        super().__init__(client, config)
        self.tokenizer = tokenizer
        self.row_attempt = ContextVar("qwen_row_attempt", default=1)

    async def row(self, row, spec, attempt=1):
        token = self.row_attempt.set(attempt)
        try:
            record = await super().row(row, spec, attempt=attempt)
            record["model_revision"] = self.config["MODEL_REVISION"]
            record["translation_backend"] = "local-vllm-qwen"
            return record
        finally:
            self.row_attempt.reset(token)

    def messages(self, text):
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    async def chunk(self, text):
        tokens = self.tokenizer.apply_chat_template(
            self.messages(text), tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if len(tokens) + self.config["MAX_OUTPUT_TOKENS"] > self.config["MAX_MODEL_LEN"]:
            try:
                parts = bisect_chunk(text)
            except TranslationFailure as exc:
                raise TranslationFailure(
                    "Input plus output budget exceeds model context; no safe paragraph boundary"
                ) from exc
            outputs, leaves = [], []
            for part in parts:
                output, metadata = await self.chunk(part)
                outputs.append(output)
                leaves.extend(metadata)
            return "\n\n".join(outputs), leaves
        output, leaves = await super().chunk(text)
        for leaf in leaves:
            leaf.update(
                backend="local-vllm-qwen",
                model_revision=self.config["MODEL_REVISION"],
                enable_thinking=False,
            )
        return output, leaves

    async def request(self, text):
        # Stable per-text seeds make scheduling/restarts independent of RNG call order.
        seed = int(digest([self.config["SEED"], text, self.row_attempt.get()])[:8], 16) % (2**31)
        body = {
            "model": self.config["TRANSLATION_MODEL"],
            "messages": self.messages(text),
            "max_tokens": self.config["MAX_OUTPUT_TOKENS"],
            "temperature": self.config["TEMPERATURE"],
            "top_p": self.config["TOP_P"],
            "top_k": self.config["TOP_K"],
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        for attempt in range(self.config["MAX_RETRIES"] + 1):
            try:
                async with self.semaphore:
                    response = await self.client.post("/v1/chat/completions", json=body)
                    response.raise_for_status()
                data = response.json()
                if data.get("model") != self.config["TRANSLATION_MODEL"]:
                    raise RuntimeError(
                        "Local server returned a different model; stop and restart it"
                    )
                choice = data["choices"][0]
                message = choice["message"]
                finish = choice["finish_reason"]
                if message.get("refusal"):
                    raise TranslationFailure(f"Local model refusal: {message['refusal']}")
                if message.get("reasoning_content") or message.get("reasoning"):
                    raise TranslationFailure("Unexpected generated reasoning in non-thinking mode")
                output = message.get("content") or ""
                if "<think>" in output or "</think>" in output:
                    raise TranslationFailure("Unexpected thinking tags in translation")
                if finish not in {"stop", "length"}:
                    raise TranslationFailure(f"Unexpected finish_reason: {finish}")
                usage = data.get("usage", {})
                return SimpleNamespace(
                    stop_reason="end_turn" if finish == "stop" else "max_tokens",
                    content=[SimpleNamespace(type="text", text=output)],
                    model=data["model"],
                    usage=SimpleNamespace(
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                    ),
                )
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                if status is not None and status != 429 and status < 500:
                    raise RuntimeError(
                        f"Local server rejected the request ({status}): {exc.response.text[:1000]}"
                    ) from exc
                if attempt == self.config["MAX_RETRIES"]:
                    # An unavailable/OOM server must stop the job, not flag thousands of rows.
                    raise RuntimeError(
                        "Local Qwen server unavailable; inspect qwen_server.log and resume"
                    ) from exc
                cap = min(
                    self.config["BACKOFF_MAX_SECONDS"], self.config["BACKOFF_SECONDS"] * 2**attempt
                )
                await asyncio.sleep(cap * (0.5 + self.rng.random() / 2))


def write_smoke_review(selected, registry, folder):
    """A self-contained, escaped side-by-side review of every smoke field."""
    sections = []
    for name, rows in selected.items():
        records = latest_by_id(folder / f"{name}.jsonl")
        for row in rows:
            record = records[str(row["id"])]
            fields = []
            for field, original in source_fields(row, registry[name]).items():
                translated = record["translations"].get("ko_" + field, "[MISSING]")
                fields.append(
                    f"<h4>{html.escape(field)}</h4><div class='pair'>"
                    f"<pre>{html.escape(str(original))}</pre>"
                    f"<pre>{html.escape(translated)}</pre></div>"
                )
            sections.append(
                f"<details><summary>{html.escape(name)} / "
                f"{html.escape(str(row['id']))}</summary>{''.join(fields)}</details>"
            )
    destination = folder / "smoke_review.html"
    destination.write_text(
        "<!doctype html><meta charset='utf-8'><title>Qwen smoke review</title>"
        "<style>body{font:16px system-ui;margin:24px}summary{cursor:pointer;padding:12px}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;background:#f5f5f5}"
        "</style><h1>Qwen smoke review</h1><p>English left; Korean right. "
        "Review all fields alongside checks_report.json before enabling the full run.</p>"
        + "".join(sections),
        encoding="utf-8",
    )
    return destination
