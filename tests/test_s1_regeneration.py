import ast
import base64
import copy
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import httpx
import nbformat
import pytest
import yaml

from common.io import digest, read_jsonl
from pipeline.regenerate_s1 import FatalAPIError, RegenerationRun, retry_after_seconds


@pytest.fixture
def source():
    return [{"question": f'Problem {i}: "x,y"\nFind α + β.',
             "deepseek_thinking_trajectory": 'Original "reasoning",\nwith Unicode ∑',
             "deepseek_attempt": "Original answer", "deepseek_grade": "No",
             "solution": "Reference should not be sent", "nested": {"id": i}}
            for i in range(24)]


@pytest.fixture
def config(source):
    cfg = yaml.safe_load(Path("configs/regenerate_s1_deepseek.yaml").read_text())
    cfg["source"].update(expected_rows=len(source), content_digest=digest(source))
    cfg["api"].update(min_request_interval_seconds=0, retry_initial_seconds=0,
                       retry_max_seconds=0, max_retries=1, concurrency=3)
    return cfg


def response(index="test", *, finish="stop", reasoning='Plan, calculate.\nCheck "α".',
             answer=r"\boxed{wrong-is-retained}"):
    return httpx.Response(200, json={
        "id": f"response-{index}", "model": "returned-model-version",
        "system_fingerprint": "test", "usage": {"prompt_tokens": 2, "completion_tokens": 5,
                                                   "total_tokens": 7},
        "choices": [{"finish_reason": finish,
                     "message": {"reasoning_content": reasoning, "content": answer}}],
    })


async def test_smoke_then_full_concurrent_resumable_csv_and_source_preservation(
    source, config, tmp_path,
):
    import asyncio

    original = copy.deepcopy(source)
    calls, active, peak = [], 0, 0

    async def handler(request):
        nonlocal active, peak
        body = json.loads(request.content)
        assert body["model"] == "deepseek-v4-pro"
        assert body["thinking"] == {"type": "enabled"}
        assert "temperature" not in body and "seed" not in body
        assert body["messages"][0]["content"] == config["prompt"]
        question = body["messages"][1]["content"]
        assert question in [r["question"] for r in source]
        assert len(body["messages"]) == 2
        calls.append(question)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return response(question)

    transport = httpx.MockTransport(handler)
    run = RegenerationRun(source, config, tmp_path)
    assert len(run.smoke_indices) == len(set(run.smoke_indices)) == 20
    await run.generate(run.smoke_indices, "fake-key", progress=False, transport=transport)
    output = run.export(run.smoke_indices, "smoke")
    assert output["summary"]["complete"] == 20
    assert 1 < peak <= config["api"]["concurrency"]
    with output["path"].open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["question", "deepseek_thinking_trajectory", "deepseek_attempt",
                                     "deepseek-v4-pro_reasoning", "deepseek-v4-pro_answer"]
        data = list(reader)
    assert len(data) == 20
    for row, index in zip(data, run.smoke_indices, strict=True):
        assert row["question"] == source[index]["question"]
        assert row["deepseek_thinking_trajectory"] == source[index]["deepseek_thinking_trajectory"]
        assert row[run.answer_column] == r"\boxed{wrong-is-retained}"
    # New runtime + torn final append: repair only the interrupted tail and reuse completed rows.
    with run.journal.open("ab") as f:
        f.write(b'{"index":')
    resumed = RegenerationRun(source, config, tmp_path)
    assert resumed.root == run.root and resumed.smoke_indices == run.smoke_indices
    await resumed.generate(range(len(source)), "fake-key", progress=False, transport=transport)
    assert len(calls) == len(set(calls)) == len(source)
    full = resumed.export(range(len(source)), "full")
    assert full["path"].name == "dataset.jsonl"
    augmented = read_jsonl(full["path"])
    assert len(augmented) == len(source)
    for old, new in zip(source, augmented, strict=True):
        assert {k: new[k] for k in old} == old
        assert set(new) - set(old) == {run.reasoning_column, run.answer_column}
    assert source == original
    assert full["summary"]["reported_tokens_including_retries"]["total_tokens"] == 24 * 7


async def test_rate_limit_retry_and_raw_usage(source, config, tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return response()

    run = RegenerationRun(source, config, tmp_path)
    await run.generate([0], "fake-key", progress=False, transport=httpx.MockTransport(handler))
    records = read_jsonl(run.journal)
    assert calls == 2 and run.succeeded(0)
    assert records[0]["status"] == "api_error" and records[0]["http_status"] == 429
    assert records[1]["response_model"] == "returned-model-version"
    assert records[1]["usage"]["total_tokens"] == 7
    assert "fake-key" not in run.journal.read_text()
    assert retry_after_seconds("3") == 3
    assert retry_after_seconds("invalid") == 0


@pytest.mark.parametrize("bad", [
    {"finish": "length"}, {"reasoning": None}, {"answer": ""},
])
async def test_incomplete_not_promoted_and_retried_on_resume(source, config, tmp_path, bad):
    run = RegenerationRun(source, config, tmp_path)
    await run.generate([0], "fake-key", progress=False,
                       transport=httpx.MockTransport(lambda request: response(**bad)))
    assert not run.succeeded(0)
    partial = run.export(range(len(source)), "full")
    assert partial["path"].name == "dataset.partial.jsonl"
    rows = read_jsonl(partial["path"])
    assert len(rows) == len(source) and rows[0][run.answer_column] is None
    assert run.records[0]["status"] == "incomplete"
    await run.generate([0], "fake-key", progress=False,
                       transport=httpx.MockTransport(lambda request: response()))
    assert run.succeeded(0) and len(run.records) == 2


@pytest.mark.parametrize("status,error", [(401, FatalAPIError), (402, FatalAPIError),
                                        (400, FatalAPIError), (503, RuntimeError)])
async def test_fatal_or_persistent_service_failure_stops_remaining_work(
    source, config, tmp_path, status, error,
):
    config["api"]["concurrency"] = 1
    run = RegenerationRun(source, config, tmp_path)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return response() if calls == 1 else httpx.Response(status)

    with pytest.raises(error):
        await run.generate(range(24), "fake-key", progress=False,
                           transport=httpx.MockTransport(handler))
    assert run.succeeded(0) and not run.succeeded(1)
    assert calls == (3 if status == 503 else 2)
    assert not run._running


async def test_cancel_keeps_completed_rows_and_cancels_workers(source, config, tmp_path):
    import asyncio

    config["api"]["concurrency"] = 1
    run = RegenerationRun(source, config, tmp_path)
    blocked = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return response()
        blocked.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(run.generate(range(24), "fake-key", progress=False,
                                           transport=httpx.MockTransport(handler)))
    await asyncio.wait_for(blocked.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set() and run.succeeded(0) and not run._running
    assert len(read_jsonl(run.journal)) == 1


def test_changed_config_isolated_source_validation_and_column_collision(source, config, tmp_path):
    first = RegenerationRun(source, config, tmp_path)
    cfg = copy.deepcopy(config)
    cfg["prompt"] += "\nCheck units."
    assert RegenerationRun(source, cfg, tmp_path).root != first.root
    bad = copy.deepcopy(source)
    bad[0]["question"] = "wrong source"
    with pytest.raises(ValueError, match="Source differs"):
        RegenerationRun(bad, config, tmp_path)
    bad = copy.deepcopy(source)
    bad[0][first.answer_column] = "existing"
    cfg["source"]["content_digest"] = digest(bad)
    with pytest.raises(ValueError, match="already exist"):
        RegenerationRun(bad, cfg, tmp_path)


def test_notebook_compiles_bundle_current_and_full_run_defaults_off():
    nb = nbformat.read("notebooks/regenerate_s1_deepseek.ipynb", as_version=4)
    nbformat.validate(nb)
    bundle = None
    smoke_cell = full_cell = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        compile(cell.source, f"cell-{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        assert cell.execution_count is None and cell.outputs == []
        if 'run.export(run.smoke_indices, "smoke")' in cell.source:
            smoke_cell = i
        if "RUN_FULL_GENERATION = False" in cell.source:
            full_cell = i
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "BUNDLE" for t in node.targets
            ):
                bundle = ast.literal_eval(node.value)
    assert smoke_cell is not None and full_cell is not None and smoke_cell != full_cell
    assert bundle
    with ZipFile(io.BytesIO(base64.b64decode(bundle))) as archive:
        for name in archive.namelist():
            assert archive.read(name) == Path(name).read_bytes(), f"Stale bundle: {name}"


async def test_notebook_smoke_and_full_cells_execute_with_mock_api(
    source, config, tmp_path, monkeypatch,
):
    from types import SimpleNamespace

    run = RegenerationRun(source, config, tmp_path)
    generate = run.generate
    calls = []
    downloads = []
    long_text = ('Equation "α, β"\n' * 12000) + "End of reasoning."

    def handler(request):
        calls.append(json.loads(request.content))
        return response(reasoning=long_text)

    async def mock_generate(indices, api_key):
        await generate(indices, api_key, progress=False, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(run, "generate", mock_generate)
    namespace = {"run": run, "source_rows": source, "DEEPSEEK_API_KEY": "fake-key",
                 "files": SimpleNamespace(download=downloads.append)}
    nb = nbformat.read("notebooks/regenerate_s1_deepseek.ipynb", as_version=4)
    smoke = next(c.source for c in nb.cells if c.cell_type == "code"
                 and 'run.export(run.smoke_indices, "smoke")' in c.source)
    full = next(c.source for c in nb.cells if c.cell_type == "code"
                and "RUN_FULL_GENERATION = False" in c.source)
    await eval(compile(smoke, "smoke", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
    assert len(calls) == 20 and len(downloads) == 1
    previous_limit = csv.field_size_limit(len(long_text) * 2)
    try:
        with Path(downloads[0]).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                assert row[run.reasoning_column] == long_text
    finally:
        csv.field_size_limit(previous_limit)
    await eval(compile(full, "full-off", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
    assert len(calls) == 20
    enabled = full.replace("RUN_FULL_GENERATION = False", "RUN_FULL_GENERATION = True")
    await eval(compile(enabled, "full-on", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
    assert len(calls) == 24 and namespace["full_result"]["summary"]["remaining"] == 0
    assert namespace["full_result"]["path"].is_file()
