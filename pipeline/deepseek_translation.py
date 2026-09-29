"""DeepSeek API translation and isolated, resumable two-model smoke runs."""

import asyncio
import html
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import httpx

from common.io import append_jsonl, digest, latest_by_id, read_jsonl, write_json
from pipeline.datasets import source_fields
from pipeline.decontamination_cache import refresh_translation_cache
from pipeline.translation import (
    SYSTEM_PROMPT,
    TranslationFailure,
    Translator,
    check_all,
    check_row,
    prepare_run,
    translate_all,
)
from pipeline.translation_protection import (
    PROTECTION_VERSION,
    ProtectedText,
    preservation_issues,
)

MODELS = ("deepseek-flash", "deepseek-v4-pro")
BASE_URL = "https://api.deepseek.com"
PROTECTED_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + r"""
- Tokens shaped like ⟪KEEP_<id>_<index>⟫ are opaque source literals. Copy every
  token exactly once. Do not modify, split, explain, expand, or add math delimiters
  around them. They may move with their surrounding phrase for Korean grammar.
- Translate only the surrounding prose. Never infer or repair apparent typos,
  lost superscripts, or mathematical mistakes. Do not introduce numeric notation
  for numbers written as words; translate those words as words.
- Terminology: diameter = 지름; radius = 반지름. Keep them distinct.
  Ordered pair = 순서쌍; ordered triple = 순서 있는 세쌍.
"""
)


class TransientGenerationError(Exception):
    pass


class DeepSeekTranslator(Translator):
    def __init__(self, client, config, usage_path):
        super().__init__(client, config)
        self.usage_path = usage_path
        # Repair an interrupted final append before any new response is journaled.
        read_jsonl(self.usage_path, repair_tail=True)

    async def request(self, text):
        try:
            protected = ProtectedText.from_source(text)
        except ValueError as exc:
            raise TranslationFailure(str(exc)) from exc
        body = {
            "model": self.config["TRANSLATION_MODEL"],
            "messages": [
                {"role": "system", "content": PROTECTED_SYSTEM_PROMPT},
                {"role": "user", "content": protected.masked},
            ],
            "thinking": {"type": "disabled"},
            "temperature": self.config["TEMPERATURE"],
            "max_tokens": self.config["MAX_OUTPUT_TOKENS"],
            "stream": False,
        }
        for attempt in range(self.config["MAX_RETRIES"] + 1):
            started = time.monotonic()
            try:
                async with self.semaphore:
                    # A total deadline also bounds responses that send keep-alive whitespace.
                    async with asyncio.timeout(self.config["REQUEST_TIMEOUT_SECONDS"]):
                        response = await self.client.post("/chat/completions", json=body)
                        response.raise_for_status()
                        data = response.json()
                choice = data["choices"][0]
                message, finish = choice["message"], choice["finish_reason"]
                append_jsonl(
                    self.usage_path,
                    {
                        "requested_model": body["model"],
                        "response_model": data.get("model"),
                        "response_id": data.get("id"),
                        "system_fingerprint": data.get("system_fingerprint"),
                        "usage": data.get("usage", {}),
                        "finish_reason": finish,
                        "request_attempt": attempt + 1,
                        "source_digest": digest(text),
                        "protection_version": PROTECTION_VERSION,
                        "protected_occurrences": len(protected.replacements),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                if finish in {"insufficient_system_resource", "aborted"}:
                    raise TransientGenerationError(finish)
                if message.get("refusal"):
                    raise TranslationFailure(f"API refusal: {message['refusal']}")
                if message.get("reasoning_content"):
                    raise TranslationFailure("API returned reasoning despite disabled thinking")
                if finish not in {"stop", "length"}:
                    raise TranslationFailure(f"Unexpected finish_reason: {finish}")
                output = message.get("content") or ""
                # Truncated responses are discarded by the shared chunk splitter.
                # Only a complete response can be checked and restored.
                if finish == "stop":
                    try:
                        output = protected.restore(output)
                    except ValueError as exc:
                        raise TranslationFailure(str(exc)) from exc
                usage = data.get("usage") or {}
                return SimpleNamespace(
                    stop_reason="end_turn" if finish == "stop" else "max_tokens",
                    content=[SimpleNamespace(type="text", text=output)],
                    model=data.get("model", body["model"]),
                    usage=SimpleNamespace(
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                    ),
                )
            except (
                httpx.TransportError,
                httpx.HTTPStatusError,
                TimeoutError,
                TransientGenerationError,
                json.JSONDecodeError,
            ) as exc:
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                if status is not None and status not in {408, 429} and status < 500:
                    # Do not echo bodies/headers that could expose a credential.
                    raise RuntimeError(
                        f"DeepSeek HTTP {status}; check key, balance, model access and request settings. "
                        "Saved rows are safe; resolve the error before resuming."
                    ) from None
                if attempt == self.config["MAX_RETRIES"]:
                    raise RuntimeError(
                        "DeepSeek retries exhausted; saved rows are safe. Rerun to resume."
                    ) from None
                cap = min(
                    self.config["BACKOFF_MAX_SECONDS"], self.config["BACKOFF_SECONDS"] * 2**attempt
                )
                await asyncio.sleep(cap * (0.5 + self.rng.random() / 2))

    async def row(self, row, spec, attempt=1):
        record = await super().row(row, spec, attempt)
        # Recheck after joining chunks; shared quality retries/exclusion consume
        # these errors just like other translation failures.
        for field, original in source_fields(row, spec).items():
            output = record["translations"].get("ko_" + field)
            if output is not None:
                record["errors"].extend(
                    f"{field}: {reason}" for reason in preservation_issues(original, output)
                )
        record["protection_version"] = PROTECTION_VERSION
        return record


def model_config(base_config, model, smoke):
    if model not in MODELS:
        raise ValueError(f"Choose one of {MODELS}")
    config = dict(base_config)
    project = Path(__file__).resolve().parents[1]
    identity = {
        "model": model,
        "base_url": BASE_URL,
        "thinking": "disabled",
        "protection_version": PROTECTION_VERSION,
        "settings": {
            k: config[k]
            for k in [
                "MAX_OUTPUT_TOKENS",
                "CHUNK_CHARS",
                "TEMPERATURE",
                "SEED",
                "SMOKE_TEST_N",
                "LENGTH_RATIO_MIN",
                "LENGTH_RATIO_MAX",
            ]
        },
        "implementation": {
            str(p.relative_to(project)): digest(p.read_text())
            for p in [
                Path(__file__),
                project / "pipeline/translation.py",
                project / "common/math_text.py",
                project / "pipeline/translation_protection.py",
            ]
        },
    }
    config.update(
        TRANSLATION_MODEL=model,
        SMOKE_TEST=smoke,
        TRANSLATION_BACKEND="deepseek-api",
        THINKING="disabled",
    )
    config["PROJECT_ROOT"] = str(
        Path(base_config["PROJECT_ROOT"]) / "models" / model / digest(identity)[:20]
    )
    return config


def prepare_model_run(base_config, registry, data, clean, report, model, smoke, reviewed=False):
    if not smoke and not reviewed:
        raise ValueError(
            "Review both smoke outputs and select a model before enabling the full run"
        )
    config = model_config(base_config, model, smoke)
    root = Path(config["PROJECT_ROOT"])
    # Preserve historical reports for the shared verified-by-ID cohort migration.
    write_json(root / "decontamination_v2" / digest(report) / "decontamination_report.json", report)
    refresh_translation_cache(data, clean, registry, config, report)
    selected, folder, manifest = prepare_run(config, registry, clean)
    write_json(root / "translation_config.json", config)
    write_json(folder / "translation_config.json", config)
    # Publisher/archives expect source cards and source manifest in this model's root.
    source = Path(base_config["PROJECT_ROOT"]) / "sources"
    for path in [source / "manifest.json"] + [source / f"{n}_source_card.md" for n in registry]:
        if path.exists():
            destination = root / "sources" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    return config, selected, folder, manifest


async def run_model(
    base_config,
    registry,
    data,
    clean,
    report,
    model,
    smoke,
    api_key,
    reviewed=False,
    transport=None,
):
    config, selected, folder, manifest = prepare_model_run(
        base_config, registry, data, clean, report, model, smoke, reviewed
    )
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required")
    print(
        f"{model}: {'SMOKE' if smoke else 'FULL'} | "
        f"{sum(map(len, selected.values()))} rows | {folder}",
        flush=True,
    )
    # The credential lives only in this client's headers, never config/manifests.
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=config["REQUEST_TIMEOUT_SECONDS"],
        transport=transport,
    ) as client:
        translator = DeepSeekTranslator(client, config, folder / "api_usage.jsonl")
        await translate_all(selected, registry, folder, translator)
        accepted, checks = await check_all(selected, registry, folder, manifest, translator)
    for name, result in checks["datasets"].items():
        print(model, name, result)
    usage = read_jsonl(folder / "api_usage.jsonl", repair_tail=True)
    totals = {
        key: sum((r.get("usage") or {}).get(key, 0) or 0 for r in usage)
        for key in [
            "prompt_tokens",
            "completion_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        ]
    }
    totals["responses_recorded"] = len(usage)
    write_json(folder / "usage_summary.json", totals)
    print("Recorded API usage (includes returned retries/truncations):", totals)
    return {
        "config": config,
        "selected": selected,
        "folder": folder,
        "manifest": manifest,
        "accepted": accepted,
        "checks": checks,
        "usage": totals,
    }


def comparison_archive(results, registry, base_root):
    """Compare exactly the same source rows, with per-model checks beside every field."""
    if set(results) != set(MODELS):
        raise ValueError("Both model smoke tests must finish before creating the comparison")
    selected = results[MODELS[0]]["selected"]
    if digest(selected) != digest(results[MODELS[1]]["selected"]):
        raise ValueError("The smoke models did not use identical source rows")
    signatures = {m: results[m]["manifest"]["signature"] for m in MODELS}
    folder = Path(base_root) / "comparisons" / digest(signatures)[:20]
    folder.mkdir(parents=True, exist_ok=True)
    records = {
        m: {n: latest_by_id(results[m]["folder"] / f"{n}.jsonl") for n in selected} for m in MODELS
    }
    sections = []
    for name, rows in selected.items():
        for row in rows:
            columns = []
            for field, original in source_fields(row, registry[name]).items():
                columns.append(
                    f"<h3>{html.escape(field)}</h3><div class='grid'>"
                    f"<section><h4>English</h4><pre>{html.escape(str(original))}</pre></section>"
                )
                for model in MODELS:
                    record = records[model][name][str(row["id"])]
                    reasons = check_row(row, record, registry[name], results[model]["config"], name)
                    status = (
                        "; ".join(reasons)
                        if reasons
                        else "Passed automatic checks; review meaning."
                    )
                    text = record["translations"].get("ko_" + field, "[MISSING]")
                    columns.append(
                        f"<section><h4>{model}</h4><p>{html.escape(status)}</p>"
                        f"<pre>{html.escape(text)}</pre></section>"
                    )
                columns.append("</div>")
            sections.append(
                f"<details><summary>{html.escape(name)} / {html.escape(str(row['id']))}"
                f"</summary>{''.join(columns)}</details>"
            )
    review = folder / "smoke_comparison.html"
    review.write_text(
        "<!doctype html><meta charset='utf-8'><title>DeepSeek smoke comparison</title>"
        "<style>body{font:16px system-ui;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:12px}"
        "summary{cursor:pointer;padding:16px}h4{position:sticky;top:0;background:white;padding:8px}</style>"
        "<h1>English / DeepSeek Flash / DeepSeek Pro</h1>"
        "<p>Review mathematical meaning, equations, omissions and reasoning structure. "
        "These API model names are aliases: response model IDs and fingerprints are logged, "
        "but the hosted weights cannot be pinned.</p>" + "".join(sections),
        encoding="utf-8",
    )
    write_json(
        folder / "comparison_summary.json",
        {
            m: {
                "checks": results[m]["checks"],
                "usage": results[m]["usage"],
                "run_folder": str(results[m]["folder"]),
            }
            for m in MODELS
        },
    )
    archive = folder / "deepseek_smoke_comparison.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        z.write(review, review.name)
        z.write(folder / "comparison_summary.json", "comparison_summary.json")
        for model in MODELS:
            for path in sorted(results[model]["folder"].rglob("*")):
                if path.is_file():
                    z.write(path, Path(model) / path.relative_to(results[model]["folder"]))
    return archive
