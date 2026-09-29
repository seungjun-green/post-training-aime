"""Local EXAONE transport with shared unmasked chunk caching and hard/soft checks.

OpenAITranslator supplies only provider-independent persistence/checking methods.
This subclass overrides its request method; all requests go to loopback vLLM.
"""

import asyncio
import json
import re
import shutil
from pathlib import Path

import httpx

from common.io import append_jsonl, digest, read_jsonl, write_json
from pipeline.decontamination_cache import refresh_translation_cache
from pipeline.openai_translation import FULL_TEXT_PROMPT, OpenAITranslator
from pipeline.text_translation_review import check_all
from pipeline.translation import prepare_run, translate_all

MODEL = "LGAI-EXAONE/EXAONE-4.5-33B"


def pin_model(config, token=None, api=None):
    from huggingface_hub import HfApi

    config = dict(config)
    if config["TRANSLATION_MODEL"] != MODEL:
        raise ValueError(f"This runtime is configured for {MODEL}")
    requested = config.get("MODEL_REVISION") or "main"
    path = Path(config["PROJECT_ROOT"]) / "model_pins" / (digest([MODEL, requested]) + ".json")
    if path.exists():
        revision = json.loads(path.read_text())["revision"]
    else:
        revision = (api or HfApi(token=token)).model_info(MODEL, revision=requested).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Model revision must resolve to an immutable Hugging Face commit")
    write_json(path, {"model": MODEL, "revision": revision})
    config["MODEL_REVISION"] = revision
    return config


def model_config(base, smoke):
    config = dict(base)
    if config["TRANSLATION_MODEL"] != MODEL or not re.fullmatch(
        r"[0-9a-f]{40}", config.get("MODEL_REVISION") or ""
    ):
        raise ValueError("Run pin_model before translation")
    if not 0 < config["MAX_OUTPUT_TOKENS"] < config["MAX_MODEL_LEN"]:
        raise ValueError("Output budget must be positive and smaller than the context limit")
    project = Path(__file__).resolve().parents[1]
    identity = {
        "settings": {
            k: config[k]
            for k in [
                "TRANSLATION_MODEL",
                "MODEL_REVISION",
                "TEMPERATURE",
                "TOP_P",
                "MAX_OUTPUT_TOKENS",
                "MAX_MODEL_LEN",
                "CHUNK_CHARS",
                "HARD_CHECK_RETRIES",
                "SEED",
                "SMOKE_TEST_N",
                "LENGTH_RATIO_MIN",
                "LENGTH_RATIO_MAX",
            ]
        },
        "prompt": FULL_TEXT_PROMPT,
        "enable_thinking": False,
        "dtype": "bfloat16",
        "implementation": {
            p: digest((project / p).read_text())
            for p in [
                "pipeline/exaone_translation.py",
                "pipeline/exaone_runtime.py",
                "pipeline/openai_translation.py",
                "pipeline/text_translation_checks.py",
                "pipeline/text_translation_review.py",
                "pipeline/translation.py",
                "common/math_text.py",
                "requirements-exaone.lock",
            ]
        },
    }
    config.update(SMOKE_TEST=smoke, TRANSLATION_BACKEND="local-vllm-exaone")
    config["PROJECT_ROOT"] = str(Path(base["PROJECT_ROOT"]) / "runs" / digest(identity)[:20])
    return config


class ExaoneTranslator(OpenAITranslator):
    async def request(self, text, feedback=None):
        instructions = FULL_TEXT_PROMPT
        if feedback:
            instructions += "\nPrevious translation failed these literal checks:\n" + "\n".join(
                feedback
            )
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content": text}]
        body = {
            "model": MODEL,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": self.config["MAX_OUTPUT_TOKENS"],
            "temperature": self.config["TEMPERATURE"],
            "top_p": self.config["TOP_P"],
            "seed": int(digest([self.config["SEED"], text, feedback])[:8], 16) % (2**31),
            "stream": False,
        }
        for attempt in range(self.config["MAX_RETRIES"] + 1):
            try:
                async with self.semaphore:
                    # Use the pinned server tokenizer and exactly the same chat template.
                    response = await self.client.post(
                        "/tokenize",
                        json={
                            "model": MODEL,
                            "messages": messages,
                            "add_generation_prompt": True,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    response.raise_for_status()
                    count = response.json()["count"]
                    if count + body["max_tokens"] > self.config["MAX_MODEL_LEN"]:
                        # Shared chunk code safely bisects at paragraph boundaries.
                        return {
                            "status": "incomplete",
                            "output": [],
                            "incomplete_details": {"reason": "max_output_tokens"},
                            "local_reason": "input_plus_output_exceeds_context",
                        }
                    response = await self.client.post("/v1/chat/completions", json=body)
                    response.raise_for_status()
                    raw = response.json()
                if raw.get("model") != MODEL:
                    raise RuntimeError("Unexpected server model; restart the EXAONE server")
                choice = raw["choices"][0]
                message = choice["message"]
                output = message.get("content") or ""
                finish = choice["finish_reason"]
                usage = raw.get("usage") or {}
                append_jsonl(
                    self.usage_path,
                    {
                        "source_digest": digest(text),
                        "response_model": raw["model"],
                        "model_revision": self.config["MODEL_REVISION"],
                        "response_id": raw.get("id"),
                        "finish_reason": finish,
                        "request_attempt": attempt + 1,
                        "usage": {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    },
                )
                refusal = message.get("refusal") or finish == "content_filter"
                unexpected_thinking = (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or "<think>" in output
                    or "</think>" in output
                )
                status = "completed" if finish == "stop" else "incomplete"
                if unexpected_thinking:
                    status = "unexpected_reasoning"
                return {
                    "id": raw.get("id"),
                    "model": raw["model"],
                    "status": status,
                    "incomplete_details": {"reason": "max_output_tokens"}
                    if finish == "length"
                    else {"reason": finish},
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "refusal", "refusal": str(refusal)}
                                if refusal
                                else {"type": "output_text", "text": output}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                    },
                }
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                if status is not None and status not in {408, 429} and status < 500:
                    raise RuntimeError(
                        f"Local EXAONE HTTP {status}: {exc.response.text[:1500]}"
                    ) from exc
                if attempt == self.config["MAX_RETRIES"]:
                    raise RuntimeError(
                        "EXAONE server unavailable; inspect exaone_server.log and resume"
                    ) from exc
                await asyncio.sleep(
                    min(
                        self.config["BACKOFF_MAX_SECONDS"],
                        self.config["BACKOFF_SECONDS"] * 2**attempt,
                    )
                )

    async def row(self, row, spec, attempt=1):
        record = await super().row(row, spec, attempt)
        record.update(
            model_revision=self.config["MODEL_REVISION"], translation_backend="local-vllm-exaone"
        )
        return record


async def run_model(base, registry, data, clean, report, smoke, reviewed=False, transport=None):
    if not smoke and not reviewed:
        raise ValueError("Review the smoke output before enabling full translation")
    config = model_config(base, smoke)
    root = Path(config["PROJECT_ROOT"])
    write_json(root / "decontamination_v2" / digest(report) / "decontamination_report.json", report)
    refresh_translation_cache(data, clean, registry, config, report)
    selected, folder, manifest = prepare_run(config, registry, clean)
    write_json(root / "translation_config.json", config)
    write_json(folder / "translation_config.json", config)
    source = Path(base["PROJECT_ROOT"]) / "sources"
    for path in [source / "manifest.json"] + [
        source / f"{name}_source_card.md" for name in registry
    ]:
        if path.exists():
            destination = root / "sources" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    print(
        f"{MODEL}: {'SMOKE' if smoke else 'FULL'} | {sum(map(len, selected.values()))} rows | {folder}",
        flush=True,
    )
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{config['SERVER_PORT']}",
        timeout=config["REQUEST_TIMEOUT_SECONDS"],
        trust_env=False,
        transport=transport,
    ) as client:
        # Prevent resuming with a different model revision/settings on the same port.
        expected = {k: base[k] for k in ["TRANSLATION_MODEL", "MODEL_REVISION", "MAX_MODEL_LEN"]}
        state = json.loads((Path(base["PROJECT_ROOT"]) / "server_identity.json").read_text())
        if state != expected:
            raise RuntimeError("Restart the server after changing model/context settings")
        translator = ExaoneTranslator(client, config, folder)
        await translate_all(selected, registry, folder, translator)
        accepted, checks = check_all(selected, registry, folder, manifest, translator)
    usage = read_jsonl(folder / "api_usage.jsonl", repair_tail=True)
    totals = {
        k: sum((row.get("usage") or {}).get(k, 0) or 0 for row in usage)
        for k in ["input_tokens", "output_tokens", "total_tokens"]
    }
    totals["responses_recorded"] = len(usage)
    write_json(folder / "usage_summary.json", totals)
    for name, result in checks["datasets"].items():
        print(name, result)
    print("Review:", folder / "human_review.html")
    return dict(
        config=config,
        selected=selected,
        folder=folder,
        manifest=manifest,
        accepted=accepted,
        checks=checks,
        usage=totals,
    )
