"""Async translation, safe paragraph chunks, durable row journals and quality checks."""

import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

from common.io import append_jsonl, digest, latest_by_id, write_json, write_jsonl
from common.math_text import is_hangul, last_boxed, protected_spans
from pipeline.datasets import source_fields

SYSTEM_PROMPT = r"""You are a professional translator of mathematics from English to Korean.
Translate the user's text into natural, fluent Korean.

Rules:
- Preserve all LaTeX, math expressions, numbers, variable names, and code exactly as written.
- Do not solve, simplify, correct, summarize, add, or remove anything.
- Keep the original structure: line breaks, paragraphs, lists, and \boxed{...} expressions.
- For reasoning text, translate faithfully, including hesitations and self-corrections
  (e.g. "Wait" → "잠깐").
- Output only the Korean translation, with no preface or notes."""


class TranslationFailure(ValueError):
    pass


def paragraph_units(text):
    spans = protected_spans(text)
    boundaries = [
        m.start()
        for m in re.finditer(r"\n\n", text)
        if not any(start <= m.start() < end for start, end in spans)
    ]
    units, start = [], 0
    for boundary in boundaries:
        if boundary < start:
            continue
        units.append(text[start:boundary])
        start = boundary + 2
    units.append(text[start:])
    return units


def split_chunks(text, limit):
    if limit < 1:
        raise ValueError("CHUNK_CHARS must be positive")
    if len(text) <= limit:
        return [text]
    chunks, current = [], None
    for unit in paragraph_units(text):
        if len(unit) > limit:
            raise TranslationFailure(
                f"A paragraph/protected math block has {len(unit)} characters, exceeding {limit}; "
                "cannot split it at paragraph boundaries without violating Spec 1"
            )
        if current is None:
            current = unit
        elif len(current) + 2 + len(unit) <= limit:
            current += "\n\n" + unit
        else:
            chunks.append(current)
            current = unit
    if current is not None:
        chunks.append(current)
    assert "\n\n".join(chunks) == text
    return chunks


def bisect_chunk(text):
    units = paragraph_units(text)
    if len(units) < 2:
        raise TranslationFailure("max_tokens: no safe paragraph boundary remains for re-splitting")
    positions = [len("\n\n".join(units[:i])) for i in range(1, len(units))]
    index = min(range(len(positions)), key=lambda i: abs(positions[i] - len(text) / 2)) + 1
    return ["\n\n".join(units[:index]), "\n\n".join(units[index:])]


class Translator:
    def __init__(self, client, config):
        self.client, self.config = client, config
        self.semaphore = asyncio.Semaphore(config["MAX_CONCURRENCY"])
        self.rng = random.Random(config["SEED"])

    async def request(self, text):
        from anthropic import APIConnectionError, APIStatusError

        for attempt in range(self.config["MAX_RETRIES"] + 1):
            try:
                async with self.semaphore:
                    # Streaming prevents long 16K-output requests from hitting non-streaming limits.
                    async with self.client.messages.stream(
                        model=self.config["TRANSLATION_MODEL"],
                        max_tokens=self.config["MAX_OUTPUT_TOKENS"],
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": text}],
                    ) as stream:
                        return await stream.get_final_message()
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable:
                    raise  # Invalid model, authentication, permissions: stop, never flag all rows.
                if attempt == self.config["MAX_RETRIES"]:
                    raise TranslationFailure(f"API retries exhausted (status={status})") from exc
                cap = min(
                    self.config["BACKOFF_MAX_SECONDS"], self.config["BACKOFF_SECONDS"] * 2**attempt
                )
                await asyncio.sleep(cap * (0.5 + self.rng.random() / 2))

    async def chunk(self, text):
        response = await self.request(text)
        if response.stop_reason == "max_tokens":
            # Discard truncated output completely. Only successful leaf chunks are retained.
            outputs, leaves = [], []
            for part in bisect_chunk(text):
                translated, metadata = await self.chunk(part)
                outputs.append(translated)
                leaves.extend(metadata)
            return "\n\n".join(outputs), leaves
        if response.stop_reason != "end_turn":
            raise TranslationFailure(f"Unexpected stop_reason: {response.stop_reason}")
        output = "".join(block.text for block in response.content if block.type == "text")
        if not output.strip():
            raise TranslationFailure("Empty translation")
        return output, [
            {
                "source_chars": len(text),
                "output_chars": len(output),
                "stop_reason": response.stop_reason,
                "response_model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        ]

    async def row(self, row, spec, attempt=1):
        translated, metadata, errors = {}, {}, []
        for column, original in source_fields(row, spec).items():
            try:
                if not isinstance(original, str) or not original.strip():
                    raise TranslationFailure(f"Empty/non-string source field: {column}")
                outputs, leaves = [], []
                for chunk in split_chunks(original, self.config["CHUNK_CHARS"]):
                    output, details = await self.chunk(chunk)
                    outputs.append(output)
                    leaves.extend(details)
                translated[f"ko_{column}"] = "\n\n".join(outputs)
                metadata[column] = leaves
            except TranslationFailure as exc:
                errors.append(f"{column}: {exc}")
        return {
            "id": row["id"],
            "source": row,
            "translations": translated,
            "chunks": metadata,
            "errors": errors,
            "attempt": attempt,
            "source_digest": digest(row),
            "translation_model": self.config["TRANSLATION_MODEL"],
        }


def check_row(row, record, spec, config, name):
    reasons = list(record.get("errors", []))
    if record.get("source_digest") != digest(row):
        reasons.append("source_digest_mismatch")
    for column, original in source_fields(row, spec).items():
        translated = record.get("translations", {}).get(f"ko_{column}", "")
        if not any(is_hangul(c) for c in translated):
            reasons.append(f"{column}: hangul_missing")
        ratio = len(translated) / len(original) if isinstance(original, str) and original else 0
        if not config["LENGTH_RATIO_MIN"] <= ratio <= config["LENGTH_RATIO_MAX"]:
            reasons.append(f"{column}: length_ratio={ratio:.4f}")
        if isinstance(original, str) and original.count(r"\boxed{") != translated.count(r"\boxed{"):
            reasons.append(f"{column}: boxed_count_changed")
        chunks = record.get("chunks", {}).get(column, [])
        if not chunks or any(c.get("stop_reason") != "end_turn" for c in chunks):
            reasons.append(f"{column}: truncated_or_missing_chunks")
    if name == "s1k_1.1":
        source = last_boxed(row["original"]["deepseek_attempt"] or "")
        target = last_boxed(record.get("translations", {}).get("ko_deepseek_attempt", ""))
        if source is None or target is None or source != target:
            reasons.append("deepseek_attempt: final_answer_not_preserved")
    return reasons


def select_smoke(data, config):
    selected = {}
    for name, rows in data.items():
        n = min(config["SMOKE_TEST_N"], len(rows))
        if n < 1:
            raise ValueError(f"Smoke test needs at least one row: {name}")
        rng = random.Random(f"{config['SEED']}:{name}")
        if name == "s1k_1.1":
            longest = max(
                rows, key=lambda r: len(r["original"]["deepseek_thinking_trajectory"] or "")
            )
            if len(longest["original"]["deepseek_thinking_trajectory"]) <= config["CHUNK_CHARS"]:
                raise ValueError("No retained s1K trace is long enough to exercise chunking")
            selected[name] = [longest] + rng.sample(
                [r for r in rows if r["id"] != longest["id"]], n - 1
            )
        else:
            selected[name] = rng.sample(rows, n)
    return selected


def run_signature(config, registry, data):
    # Runtime mode and concurrency can change without invalidating translations.
    keys = [
        "TRANSLATION_MODEL",
        "MAX_OUTPUT_TOKENS",
        "CHUNK_CHARS",
        "SEED",
        "LENGTH_RATIO_MIN",
        "LENGTH_RATIO_MAX",
    ]
    return digest(
        {
            "config": {k: config[k] for k in keys},
            "registry": registry,
            "sources": {k: digest(v) for k, v in data.items()},
            "system_prompt": SYSTEM_PROMPT,
            "implementation": digest(
                {
                    "translation": Path(__file__).read_text(),
                    "math_text": (Path(__file__).parents[1] / "common/math_text.py").read_text(),
                }
            ),
        }
    )


def prepare_run(config, registry, data):
    root = Path(config["PROJECT_ROOT"])
    mode = "smoke_test" if config["SMOKE_TEST"] else "translations"
    signature = run_signature(config, registry, data)
    if not config["SMOKE_TEST"]:
        smoke = root / "smoke_test" / "manifest.json"
        if not smoke.exists():
            raise ValueError("Run and review the smoke test before setting SMOKE_TEST=False")
        previous = json.loads(smoke.read_text())
        if previous["signature"] != signature or not previous.get("checks_completed"):
            raise ValueError(
                "A completed smoke test with the same sources/model/settings is required"
            )
    rows = select_smoke(data, config) if config["SMOKE_TEST"] else data
    folder = root / mode
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["signature"] != signature or manifest["selected_ids"] != {
            n: [r["id"] for r in rs] for n, rs in rows.items()
        }:
            raise ValueError("Translation run changed; use a new project root")
    else:
        manifest = {
            "signature": signature,
            "mode": mode,
            "config": config,
            "selected_ids": {n: [r["id"] for r in rs] for n, rs in rows.items()},
            "checks_completed": False,
        }
        write_json(path, manifest)
    return rows, folder, manifest


async def bounded_map(items, function, concurrency):
    queue = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    async def worker():
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await function(item)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def translate_all(rows, registry, folder, translator):
    for name, examples in rows.items():
        path = folder / f"{name}.jsonl"
        existing = latest_by_id(path)
        for row in examples:
            if str(row["id"]) in existing and existing[str(row["id"])]["source_digest"] != digest(
                row
            ):
                raise ValueError(f"Resume source mismatch: {name}/{row['id']}")
        pending = [r for r in examples if str(r["id"]) not in existing]

        async def translate(row):
            record = await translator.row(row, registry[name])
            append_jsonl(path, record)  # Synchronous append: no await/interleaving during write.

        await bounded_map(pending, translate, translator.config["MAX_CONCURRENCY"])
        print(f"{name}: {len(examples)} rows persisted ({len(pending)} new)")


async def check_all(rows, registry, folder, manifest, translator):
    config = translator.config
    checks = Path(config["PROJECT_ROOT"]) / "checks"
    report = {"mode": manifest["mode"], "signature": manifest["signature"], "datasets": {}}
    accepted = {}
    for name, examples in rows.items():
        path = folder / f"{name}.jsonl"
        records = latest_by_id(path)
        if any(str(r["id"]) not in records for r in examples):
            raise ValueError(f"Translation is incomplete: {name}")
        retry = [
            r
            for r in examples
            if check_row(r, records[str(r["id"])], registry[name], config, name)
            and records[str(r["id"])]["attempt"] < 2
        ]

        async def retry_row(row):
            record = await translator.row(row, registry[name], attempt=2)
            append_jsonl(path, record)
            records[str(row["id"])] = record

        await bounded_map(retry, retry_row, config["MAX_CONCURRENCY"])
        flagged, accepted[name], reasons = [], [], Counter()
        for row in examples:
            record = records[str(row["id"])]
            failures = check_row(row, record, registry[name], config, name)
            if failures:
                flagged.append(
                    {"id": row["id"], "reasons": failures, "source": row, "record": record}
                )
                reasons.update(failures)
            else:
                accepted[name].append((row, record))
        write_jsonl(checks / f"{name}_flagged.jsonl", flagged)
        report["datasets"][name] = {
            "total": len(examples),
            "accepted": len(accepted[name]),
            "flagged_count": len(flagged),
            "retried_count": sum(r["attempt"] == 2 for r in records.values()),
            "reasons": dict(reasons),
        }
    write_json(checks / "checks_report.json", report)
    # Keep smoke evidence even after the full run replaces checks/.
    write_json(folder / "checks_report.json", report)
    manifest["checks_completed"] = True
    manifest["checks_digest"] = digest(report)
    write_json(folder / "manifest.json", manifest)
    return accepted, report
