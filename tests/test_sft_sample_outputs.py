import ast
import base64
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import yaml

from common.english_prompts import render_prompt
from common.io import read_jsonl, write_json, write_jsonl
from eval import run_batched_eval
from eval import sample_sft_outputs as sample


def settings():
    return yaml.safe_load(Path("configs/sft_sample15.yaml").read_text())


def datasets():
    return {name: [{"id": f"{name}-{i}", "problem": f"Problem {i}",
                    "question": f"Problem {i}", "answer": "42"} for i in range(count)]
            for name, count in [("aime_2024", 30), ("aime_2025", 30), ("aime_2026", 30),
                                ("amc23", 40), ("math_500", 500)]}


def test_pool_counts_seed_and_order_independence():
    cfg, source = settings(), datasets()
    selected = sample.select_questions(source, cfg["groups"], cfg["seed"])
    assert sum(len(rows) for name, rows in selected.items() if name.startswith("aime")) == 5
    assert len(selected["amc23"]) == len(selected["math_500"]) == 5
    assert len({(name, row["id"]) for name, rows in selected.items() for row in rows}) == 15
    assert sample.select_questions({n: list(reversed(r)) for n, r in source.items()}, cfg["groups"], cfg["seed"]) == selected
    assert sample.select_questions(source, cfg["groups"], cfg["seed"] + 1) != selected


@pytest.mark.parametrize("text,status,reasoning,answer", [
    ("<think>work\n</think>\n\\boxed{42}", "closed_think", "work\n", "\n\\boxed{42}"),
    ("<think>unfinished", "unclosed_think", "unfinished", None),
    ("Direct answer", "missing_think_tag", None, None),
])
def test_split_does_not_invent_missing_answer(text, status, reasoning, answer):
    assert sample.split_response(text) == {"reasoning": reasoning, "answer_text": answer, "parse_status": status}


def test_qwen_plain_response_retains_full_solution():
    text = "Compute 6 * 7 = 42. Therefore \\boxed{42}."
    assert sample.split_response(text, plain_response=True) == {
        "reasoning": None, "answer_text": text, "parse_status": "plain_response"}


def test_qwen_selection_needs_no_sft_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "load_eval_sets", lambda *args: datasets())
    make_checkpoint(tmp_path / "sft")
    cfg = settings()
    original = sample.run(cfg, tmp_path / "sft", tmp_path / "outputs", prepare_only=True)
    cfg["model_kind"] = "qwen"
    qwen = sample.run(cfg, tmp_path / "nonexistent", tmp_path / "outputs", prepare_only=True)
    assert original.parent.name == "sft" and qwen.parent.name == "qwen"
    assert json.loads((original / "selection.json").read_text()) == json.loads((qwen / "selection.json").read_text())
    assert sample.model_source(None, cfg) == (cfg["qwen_model"]["repo"], cfg["qwen_model"]["revision"], None)


def make_checkpoint(root):
    cfg = yaml.safe_load(Path("configs/stage1_sft.yaml").read_text())
    checkpoint = root / "checkpoints/stage1/sft_s1k/epoch_5"
    write_json(checkpoint / "stage1_checkpoint.json", {"epoch": 5, "run_identity": "original"})
    write_json(checkpoint / "config.json", {})
    (checkpoint / "model.safetensors").write_bytes(b"fake weights")
    write_json(root / "logs/stage1/sft_s1k/run_manifest.json", {"identity": "original", "config": cfg})
    return checkpoint


def test_rejects_pro_column_checkpoint(tmp_path):
    make_checkpoint(tmp_path)
    path = tmp_path / "logs/stage1/sft_s1k/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["data"]["columns"] = {"reasoning": "deepseek-v4-pro_reasoning"}
    write_json(path, manifest)
    with pytest.raises(ValueError, match="original DeepSeek"):
        sample.checkpoint_source(tmp_path, settings())


@pytest.mark.parametrize("kind", ["sft", "qwen"])
def test_generate_export_and_resume(tmp_path, monkeypatch, kind):
    from eval import engines, english_engines

    root = tmp_path / "train"
    cfg = settings()
    cfg["model_kind"] = kind
    if kind == "sft":
        make_checkpoint(root)
    tokenizer = SimpleNamespace(chat_template="native", apply_chat_template=lambda messages, **kwargs: messages[0]["content"] + "<assistant>")
    engine = SimpleNamespace(name="vllm", tokenizer=tokenizer)
    monkeypatch.setattr(sample, "load_eval_sets", lambda *args: datasets())
    monkeypatch.setattr(sample, "resolve_model", lambda model, revision, token: (revision, "fakehash"))
    monkeypatch.setattr(sample, "package_versions", lambda: {})
    monkeypatch.setattr(engines, "check_hardware", lambda config: {"name": "fake GPU"})
    def create_engine(model, config, revision):
        assert config["temperature"] == 0 and config["max_new_tokens"] == 20480
        assert "budget_forcing" not in config
        if kind == "qwen":
            assert model == cfg["qwen_model"]["repo"] and revision == cfg["qwen_model"]["revision"]
        else:
            assert model.endswith("sft_s1k/epoch_5") and revision is None
        return engine, None

    monkeypatch.setattr(english_engines, "create_engine", create_engine)
    monkeypatch.setattr(run_batched_eval, "score_response", lambda *args: ("42", True))
    generated = []

    def generate(engine, jobs, execution):
        for job in reversed(jobs):
            if len(generated) == 3 and fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("Interrupted fake generation")
            generated.append(job)
            yield job, render_prompt(tokenizer, job["problem"]), [{
                "text": "<think>Check arithmetic.\n</think>\n\\boxed{42}",
                "token_count": 15, "finish_reason": "stop"}]

    monkeypatch.setattr(run_batched_eval, "generate_problems", generate)
    output = sample.run(cfg, root, tmp_path / "outputs", prepare_only=True)
    assert generated == []
    assert len(json.loads((output / "selection.json").read_text())) == 15
    fail_once = [True]
    with pytest.raises(RuntimeError, match="Interrupted fake generation"):
        sample.run(cfg, root, tmp_path / "outputs")
    assert len(read_jsonl(output / "generations.jsonl")) == 3
    assert sample.run(cfg, root, tmp_path / "outputs") == output
    exported = read_jsonl(output / "generations.jsonl")
    assert len(exported) == len(generated) == 15
    assert all(r["gold_answer"] == "42" and r["answer_text"] == "\n\\boxed{42}" for r in exported)
    assert all(r["reasoning"] == "Check arithmetic.\n" and not r["truncated"] for r in exported)
    assert all("answer" not in r and r["source"]["budget_forcing"] is False for r in exported)
    assert all(r["source"]["model_kind"] == kind for r in exported)
    sample.run(cfg, root, tmp_path / "outputs")
    assert len(generated) == 15
    assert read_jsonl(output / "generations.jsonl") == exported


def test_notebook_syntax_and_defaults():
    notebook = nbformat.read("notebooks/sample_original_sft_15.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert "@param" not in cell.source
    variables = {}
    exec(notebook.cells[1].source, variables)
    assert variables["TRAIN_ROOT"] == "/content/drive/MyDrive/LG-AIME-Stage1"
    assert "MODEL_KIND" not in variables
    assert variables["TEMPERATURE"] == 0 and variables["MAX_NEW_TOKENS"] == 20480


def test_notebook_runs_both_models_sequentially_and_combines(tmp_path):
    notebook = nbformat.read("notebooks/sample_original_sft_15.ipynb", as_version=4)
    calls = []
    selected = [{"dataset": name, "id": row["id"], "question": row["problem"], "gold_answer": row["answer"]}
                for name, rows in sample.select_questions(datasets(), settings()["groups"], 42).items()
                for row in rows]

    def run_logged(command, **kwargs):
        config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
        kind = config["model_kind"]
        preparing = "--prepare-only" in command
        calls.append((kind, preparing))
        root = Path(command[command.index("--output-root") + 1])
        run_dir = root / kind / "test-run"
        if preparing:
            write_json(root / "latest_run.json", {"directory": str(run_dir)})
            write_json(run_dir / "selection.json", selected)
            write_json(run_dir / "settings.json", {"model": kind})
        else:
            # The first model's blocking subprocess has returned before Qwen starts.
            if kind == "qwen":
                assert (root / "sft/test-run/generations.jsonl").is_file()
            write_jsonl(run_dir / "generations.jsonl", [
                {**row, "source": {"model_kind": kind}, "correct": True,
                 "token_count": 10, "finish_reason": "stop", "response": f"{kind} solution"}
                for row in selected])

    variables = {"Path": Path, "json": json, "run_logged": run_logged}
    exec(notebook.cells[1].source, variables)
    variables.update(CODE_ROOT=str(Path.cwd()), OUTPUT_ROOT=str(tmp_path), EVAL_ENV="/fake/env")
    exec(notebook.cells[5].source, variables)
    exec(notebook.cells[7].source, variables)
    exec(notebook.cells[9].source, variables)
    assert calls == [("sft", True), ("qwen", True), ("sft", False), ("qwen", False)]
    combined = read_jsonl(variables["combined_path"])
    assert len(combined) == 30
    assert {r["source"]["model_kind"] for r in combined} == {"sft", "qwen"}
    assert [r["id"] for r in combined[:15]] == [r["id"] for r in combined[15:]]


def test_bundled_notebook_prepares_both_models_without_repository(tmp_path):
    """Exercise real profile/config loading using only the delivered notebook payload."""
    notebook = nbformat.read("notebooks/sample_original_sft_15.ipynb", as_version=4)
    setup = ast.parse(notebook.cells[3].source)
    encoded = next(node.args[0].value for node in ast.walk(setup)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "b64decode")
    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        archive.extractall(bundle_root)
    train_root = tmp_path / "train"
    make_checkpoint(train_root)
    write_json(tmp_path / "fixture_datasets.json", datasets())
    script = '''
import json, sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "bundle"))
from eval import sample_sft_outputs as sample
assert sample.ROOT == root / "bundle"
# Stub only remote benchmark retrieval: all bundled imports, profiles, checkpoint
# validation, sampling, code fingerprinting and output writes execute unchanged.
sample.load_eval_sets = lambda *args: json.loads((root / "fixture_datasets.json").read_text())
settings = yaml.safe_load((sample.ROOT / "configs/sft_sample15.yaml").read_text())
selections = []
for kind in ["sft", "qwen"]:
    settings["model_kind"] = kind
    output = sample.run(settings, root / "train", root / "outputs", prepare_only=True)
    metadata = json.loads((output / "settings.json").read_text())
    assert metadata["config"]["temperature"] == 0
    assert all(spec["n"] == 1 for spec in metadata["config"]["datasets"].values())
    selections.append(json.loads((output / "selection.json").read_text()))
assert len(selections[0]) == 15 and selections[0] == selections[1]
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
