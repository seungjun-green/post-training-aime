import ast
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import yaml

from common.english_prompts import INSTRUCTION, messages, render_prompt
from common.io import digest
from eval.english_engines import EnglishVLLMEngine
from eval.run_english_eval import bind_protocol, load_eval_sets


def fixture():
    rows = [{"problem_idx": 7, "problem": "What is 6 times 7?", "answer": 42}]
    spec = {
        "repo": "owner/dp_removed_test",
        "config": "default",
        "split": "train",
        "problem_column": "problem",
        "answer_column": "answer",
        "source_rows": 1,
        "n": 32,
    }
    config = {"language": "en", "datasets": {"test": spec}}
    suite = {
        "language": "en",
        "datasets": {
            "test": {
                **spec,
                "revision": "a" * 40,
                "role": "eval",
                "removed_rows": 0,
                "rows": 1,
                "ids": [7],
                "id_column": "problem_idx",
                "content_digest": digest(rows),
            }
        },
    }
    return rows, config, suite


def test_loader_pins_revision_config_split_and_preserves_native_id():
    rows, config, suite = fixture()
    original = deepcopy(rows)

    def loader(repo, **kwargs):
        assert repo == "owner/dp_removed_test"
        assert kwargs == dict(name="default", split="train", revision="a" * 40, token="test")
        return rows

    loaded = load_eval_sets(config, suite, "test", loader)
    assert loaded["test"][0]["id"] == 7 and loaded["test"][0]["problem_idx"] == 7
    assert rows == original


@pytest.mark.parametrize("change", ["content", "ids", "count", "revision", "role", "language"])
def test_loader_rejects_wrong_data(change):
    rows, config, suite = fixture()
    locked = suite["datasets"]["test"]
    if change == "content":
        rows[0]["problem"] = "Altered problem"
    elif change == "ids":
        locked["ids"] = [8]
    elif change == "count":
        locked["rows"] = 2
    elif change == "revision":
        locked["revision"] = "main"
    elif change == "role":
        locked["role"] = "train"
    else:
        suite["language"] = "ko"
    with pytest.raises(ValueError):
        load_eval_sets(config, suite, loader=lambda *a, **k: rows)


def test_english_prompt_and_engine_sampling(monkeypatch):
    assert "English" in INSTRUCTION and r"\boxed{}" in INSTRUCTION
    assert messages("Question") == [{"role": "user", "content": "Question\n\n" + INSTRUCTION}]

    class Tokenizer:
        chat_template = "native"

        def apply_chat_template(self, msgs, **kwargs):
            assert len(msgs) == 1 and msgs[0]["role"] == "user"
            assert msgs[0]["content"].endswith(INSTRUCTION)
            assert kwargs == dict(tokenize=False, add_generation_prompt=True)
            return "native English prompt"

        def encode(self, prompt, **kwargs):
            assert prompt == "native English prompt" and kwargs == dict(add_special_tokens=False)
            return [1, 2]

    assert render_prompt(Tokenizer(), "Question") == "native English prompt"
    captured = {}

    def params(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=params))

    class LLM:
        def generate(self, prompts, params, use_tqdm):
            assert prompts == [{"prompt_token_ids": [1, 2]}]
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text=r"\boxed{42}", token_ids=[3], finish_reason="stop")
                    ]
                )
            ]

    engine = EnglishVLLMEngine.__new__(EnglishVLLMEngine)
    engine.tokenizer = Tokenizer()
    engine.llm = LLM()
    engine.config = dict(temperature=1.0, top_p=0.7, max_new_tokens=20480, max_model_len=32768)
    prompt, result = engine.generate("Question", 32, 123)
    assert captured == dict(n=32, temperature=1.0, top_p=0.7, max_tokens=20480, seed=123)
    assert result[0]["text"] == r"\boxed{42}"


def test_pinned_suite_matches_uploaded_revisions_and_protocol():
    config = yaml.safe_load(Path("configs/eval_english.yaml").read_text())
    suite = json.loads(Path("configs/english_eval_suite.json").read_text())
    assert set(config["datasets"]) == set(suite["datasets"])
    assert sum(s["source_rows"] for s in config["datasets"].values()) == 630
    assert sum(s["source_rows"] * s["n"] for s in config["datasets"].values()) == 6160
    assert suite["datasets"]["aime_2026"]["id_column"] == "problem_idx"
    assert suite["datasets"]["math_500"]["id_column"] == "unique_id"
    assert config["datasets"]["amc23"]["problem_column"] == "question"
    assert all("ko_" not in s["problem_column"] for s in config["datasets"].values())
    assert all(len(s["ids"]) == s["rows"] for s in suite["datasets"].values())


def test_english_protocol_is_frozen_and_smoke_separate(tmp_path):
    _, config, suite = fixture()
    ids = {"test": [7]}
    first = bind_protocol(tmp_path / "full", config, suite, "stage0", ids, False)
    assert bind_protocol(tmp_path / "full", config, suite, "stage1", ids, False) == first
    smoke = bind_protocol(tmp_path / "smoke", config, suite, "stage0", ids, True)
    assert first != smoke
    with pytest.raises(ValueError, match="changed"):
        bind_protocol(tmp_path / "full", config, suite, "stage0", ids, True)


def test_notebook_is_thin_git_launcher_without_embedded_project():
    nb = nbformat.read("notebooks/evaluate_baseline_english.ipynb", as_version=4)
    nbformat.validate(nb)
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "BUNDLE" not in text and "base64" not in text
    assert '"git", "clone"' in text and '"--detach"' in text
    assert "eval.run_english_eval" in text and "RUN_FULL_EVAL = False" in text
    assert len(text.splitlines()) < 70
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<notebook>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            assert not cell.outputs and cell.execution_count is None
