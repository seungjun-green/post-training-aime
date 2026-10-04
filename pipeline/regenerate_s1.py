"""Concurrent DeepSeek reasoning generation; no answer-correctness filtering."""

import asyncio
import copy
import csv
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from common import io
from common.io import append_jsonl, digest, read_jsonl, write_json, write_jsonl


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def validate_source(rows, config):
    spec = config["source"]
    if len(rows) != spec["expected_rows"] or digest(rows) != spec["content_digest"]:
        raise ValueError("Source differs from the pinned decontaminated s1K dataset")
    for row in rows:
        for name in config["columns"].values():
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Missing source text: {name}")


def load_source(config, token):
    from datasets import load_dataset

    spec = config["source"]
    dataset = load_dataset(spec["repo"], name=spec["config"], split=spec["split"],
                           revision=spec["revision"], token=token)
    rows = list(dataset)
    validate_source(rows, config)
    return rows


def retry_after_seconds(value):
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.0


class FatalAPIError(RuntimeError):
    """Authentication, billing or invalid request errors must stop paid work."""


def answer_format_issues(answer, output_format):
    """Check presentation only; no claim of correctness or authentic self-correction."""
    if not isinstance(answer, str) or not answer.strip():
        return ["missing_final_response"]
    issues = []
    marker = re.escape(output_format["final_answer_marker"])
    endings = list(re.finditer(r"(?im)^[ \t]*(?:\*\*)?" + marker + r"(?:\*\*)?[ \t]*", answer))
    if len(endings) != 1:
        issues.append("missing_or_duplicate_final_answer_marker")
    else:
        if not answer[:endings[0].start()].strip():
            issues.append("missing_worked_solution")
        if not answer[endings[0].end():].strip():
            issues.append("missing_final_conclusion")
    # Flag a return to the old four-section template; no reasoning behavior is
    # required to occur a fixed number of times, or at all when unnecessary.
    if re.search(r"(?im)^[ \t]*(?:#{1,6}[ \t]+|\*\*)[ \t]*(?:[1-4][.)][ \t]*)?"
                 r"(?:Planning|Evaluation|Reflection|Exploration)[ \t:]*(?:\*\*)?[ \t]*$", answer):
        issues.append("fixed_cognitive_section_heading")
    return issues


class RegenerationRun:
    def __init__(self, rows, config, output_root):
        self.rows = copy.deepcopy(rows)
        self.config = copy.deepcopy(config)
        validate_source(self.rows, self.config)
        output_format = self.config["output_format"]
        if (set(output_format) != {"style", "final_answer_marker"}
                or output_format["style"] != "natural_reasoning"
                or not isinstance(output_format["final_answer_marker"], str)
                or not output_format["final_answer_marker"].strip()):
            raise ValueError("Expected natural_reasoning format with a final-answer marker")
        if output_format["final_answer_marker"] not in self.config["prompt"]:
            raise ValueError("Final-answer marker must be included in the prompt")
        api = self.config["api"]
        if api["model"] not in {"deepseek-v4-pro", "deepseek-flash"}:
            raise ValueError("Unsupported DeepSeek model")
        if api["reasoning_effort"] not in {"low", "high", "max"}:
            raise ValueError("Use a supported thinking effort")
        if not isinstance(api["concurrency"], int) or api["concurrency"] < 1:
            raise ValueError("concurrency must be a positive integer")
        if not isinstance(api["max_retries"], int) or api["max_retries"] < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if not 1 <= api["max_tokens"] <= 393216 or not 0.95 <= api["top_p"] <= 1:
            raise ValueError("Invalid DeepSeek token ceiling or top_p")
        if api["request_timeout_seconds"] <= 0 or api["min_request_interval_seconds"] < 0:
            raise ValueError("Invalid request timing")
        if not 0 <= api["retry_initial_seconds"] <= api["retry_max_seconds"]:
            raise ValueError("Invalid retry timing")
        count = self.config["smoke"]["examples"]
        if not isinstance(count, int) or not 1 <= count <= len(rows):
            raise ValueError("Invalid smoke sample size")
        self.reasoning_column = api["model"] + "_reasoning"
        self.answer_column = api["model"] + "_answer"
        if any(self.reasoning_column in row or self.answer_column in row for row in rows):
            raise ValueError("Generated columns already exist; refusing to overwrite source fields")
        self.smoke_indices = random.Random(self.config["smoke"]["seed"]).sample(
            range(len(rows)), count)
        # Only content, never local paths, belongs in the resumable identity.
        implementation = digest([Path(__file__).read_text(), Path(io.__file__).read_text()])
        identity = digest({"config": self.config, "implementation": implementation})
        self.root = Path(output_root) / "runs" / api["model"] / identity[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "identity": identity, "config": self.config, "implementation": implementation,
            "smoke_indices": self.smoke_indices,
            "new_columns": [self.reasoning_column, self.answer_column],
            "intended_training_target_column": self.answer_column,
            "raw_api_thinking_column": self.reasoning_column,
            "correctness_filter": False,
            "api_seed_supported": False,
        }
        write_json(self.root / "manifest.json", manifest)
        self.journal = self.root / "attempts.jsonl"
        self.records = read_jsonl(self.journal, repair_tail=True)
        self.latest = {}
        for record in self.records:
            self.latest[record["index"]] = record
        self._running = False

    def succeeded(self, index):
        return self.latest.get(index, {}).get("status") == "complete"

    def save(self, record):
        append_jsonl(self.journal, record)
        self.records.append(record)
        self.latest[record["index"]] = record

    async def _wait_slot(self):
        async with self._rate_lock:
            await asyncio.sleep(max(0.0, self._next_request - time.monotonic()))
            self._next_request = time.monotonic() + self.config["api"][
                "min_request_interval_seconds"]

    async def _request(self, client, index):
        api = self.config["api"]
        body = {
            "model": api["model"], "thinking": {"type": "enabled"},
            "reasoning_effort": api["reasoning_effort"], "top_p": api["top_p"],
            "max_tokens": api["max_tokens"], "stream": False,
            "messages": [
                {"role": "system", "content": self.config["prompt"]},
                {"role": "user", "content": self.rows[index][
                    self.config["columns"]["question"]]},
            ],
        }
        for attempt in range(api["max_retries"] + 1):
            await self._wait_slot()
            started = time.monotonic()
            record = {"index": index, "requested_model": api["model"],
                      "recorded_at": timestamp(), "attempt_in_call": attempt + 1}
            retry_delay = 0.0
            fatal = False
            try:
                async with asyncio.timeout(api["request_timeout_seconds"]):
                    response = await client.post("/chat/completions", json=body)
                    record["http_status"] = response.status_code
                    response.raise_for_status()
                    data = response.json()
                record.update(
                    response_id=data.get("id"), response_model=data.get("model"),
                    system_fingerprint=data.get("system_fingerprint"), usage=data.get("usage"),
                )
                choice = data["choices"][0]
                message = choice["message"]
                reasoning, answer = message.get("reasoning_content"), message.get("content")
                finish = choice.get("finish_reason")
                complete = (
                    finish == "stop" and not message.get("refusal")
                    and isinstance(reasoning, str) and bool(reasoning.strip())
                    and isinstance(answer, str) and bool(answer.strip())
                )
                format_issues = answer_format_issues(
                    answer, self.config["output_format"])
                record.update(
                    reasoning=reasoning, answer=answer, finish_reason=finish,
                    refusal=message.get("refusal"),
                    format_issues=format_issues,
                    status=("invalid_format" if format_issues else "complete")
                    if complete else "incomplete",
                )
                retryable = finish in {"insufficient_system_resource", "aborted"}
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                retryable = status in {408, 409, 429} or status >= 500
                fatal = not retryable
                # Do not log request headers, credentials, or arbitrary server error bodies.
                record.update(status="api_error", error=f"HTTP {status}")
                retry_delay = retry_after_seconds(exc.response.headers.get("retry-after"))
                if status == 429:
                    async with self._rate_lock:
                        self._next_request = max(self._next_request, time.monotonic() + max(
                            retry_delay, api["retry_initial_seconds"]))
            except (httpx.TransportError, TimeoutError) as exc:
                retryable = True
                record.update(status="api_error", error=type(exc).__name__)
            except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                retryable = True
                record.update(status="api_error", error="MalformedResponse:" + type(exc).__name__)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            self.save(record)
            if fatal:
                raise FatalAPIError(f"DeepSeek HTTP {record['http_status']}; check credentials, "
                                    "balance and request settings, then rerun. Completed rows saved.")
            if retryable and attempt == api["max_retries"]:
                raise RuntimeError("DeepSeek transient-error retries exhausted; completed rows "
                                   "are saved. Inspect attempts.jsonl and rerun to resume.")
            if not retryable:
                return
            backoff = min(api["retry_max_seconds"], api["retry_initial_seconds"] * 2 ** attempt)
            # Honour Retry-After even when it exceeds our exponential-backoff ceiling.
            await asyncio.sleep(max(backoff, retry_delay))

    async def generate(self, indices, api_key, *, progress=True, transport=None):
        """Resume successes; retry failed rows on a later invocation. Bound paid concurrency."""
        if self._running:
            raise RuntimeError("This run is already active")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        indices = list(indices)
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(self.rows) for i in indices):
            raise ValueError("Invalid or duplicate source indices")
        self._running = True
        tasks = []
        bar = None
        try:
            from tqdm.auto import tqdm

            api = self.config["api"]
            self._rate_lock = asyncio.Lock()
            self._next_request = 0.0
            pending = iter(i for i in indices if not self.succeeded(i))
            bar = tqdm(total=len(indices), initial=sum(self.succeeded(i) for i in indices),
                       desc="DeepSeek generation", unit="example", disable=not progress)
            async with httpx.AsyncClient(
                base_url=api["base_url"], headers={"Authorization": f"Bearer {api_key}"},
                timeout=api["request_timeout_seconds"], transport=transport,
                limits=httpx.Limits(max_connections=api["concurrency"],
                                    max_keepalive_connections=api["concurrency"]),
            ) as client:
                async def worker():
                    for index in pending:
                        await self._request(client, index)
                        bar.update(1)
                        bar.set_postfix(complete=sum(self.succeeded(i) for i in indices),
                                        refresh=False)

                tasks = [asyncio.create_task(worker()) for _ in range(
                    min(api["concurrency"], len(indices)))]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if bar is not None:
                bar.close()
            self._running = False

    def export(self, indices, mode):
        """Export all requested rows, with null outputs for failures and separate status files."""
        if mode not in {"smoke", "full"}:
            raise ValueError("Unknown export mode")
        indices = list(indices)
        if mode == "smoke" and indices != self.smoke_indices:
            raise ValueError("Smoke export must use the fixed random sample")
        if mode == "full" and indices != list(range(len(self.rows))):
            raise ValueError("Full export must preserve every source row in source order")
        target = self.root / mode
        target.mkdir(parents=True, exist_ok=True)
        statuses = []
        augmented = []
        for index in indices:
            record = self.latest.get(index, {})
            complete = self.succeeded(index)
            augmented.append({
                **self.rows[index],
                self.reasoning_column: record["reasoning"] if complete else None,
                self.answer_column: record["answer"] if complete else None,
            })
            statuses.append({"source_index": index, "status": record.get("status", "pending"),
                             "finish_reason": record.get("finish_reason"),
                             "error": record.get("error"),
                             "format_issues": record.get("format_issues", []),
                             "response_id": record.get("response_id"),
                             "response_model": record.get("response_model")})
        remaining = sum(not self.succeeded(i) for i in indices)
        summary = {"requested": len(indices), "complete": len(indices) - remaining,
                   "remaining": remaining, "source_indices": indices,
                   "correctness_filter": False}
        selected = set(indices)
        usage = [r.get("usage") or {} for r in self.records if r["index"] in selected]
        summary["reported_tokens_including_retries"] = {
            name: sum(u.get(name, 0) or 0 for u in usage)
            for name in ["prompt_tokens", "completion_tokens", "total_tokens"]
        }
        write_json(target / "summary.json", summary)
        write_jsonl(target / "status.jsonl", statuses)
        if mode == "smoke":
            cols = self.config["columns"]
            names = [cols["question"], cols["original_reasoning"], cols["original_answer"],
                     self.reasoning_column, self.answer_column]
            path = target / "comparison.csv"
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(augmented)
            temporary.replace(path)
        else:
            path = target / ("dataset.partial.jsonl" if remaining else "dataset.jsonl")
            write_jsonl(path, augmented)
        return {"path": path, "summary": summary, "status_path": target / "status.jsonl"}
