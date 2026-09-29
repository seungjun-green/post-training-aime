import math

import pytest

from common.io import read_jsonl
from common.prompts import INSTRUCTION, messages, render_prompt
from eval.engines import check_context
from eval.run_eval import bind_protocol, evaluate
from eval.scoring import korean_ratio, metrics, pass_at_k, score_response


def test_shared_prompt_has_no_system():
    assert messages("문제") == [{"role": "user", "content": "문제\n\n" + INSTRUCTION}]

    class Tokenizer:
        chat_template = "native"

        def apply_chat_template(self, msgs, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            assert len(msgs) == 1 and msgs[0]["role"] == "user"
            return "native-render"

    assert render_prompt(Tokenizer(), "문제") == "native-render"


@pytest.mark.parametrize(
    "response,gold,correct",
    [
        (r"처음 \boxed{3}, 최종 \boxed{\frac{1}{2}}", "0.5", True),
        (r"\boxed{\frac{1}{2}}", "2", False),
        ("답은 42", "42", False),
        (r"\boxed{42} \boxed{", "42", False),
        (r"\boxed{}", "0", False),
        (r"\boxed{042}", 42, True),
    ],
)
def test_real_math_verify(response, gold, correct):
    assert score_response(response, gold)[1] is correct


def test_pass_at_k_unbiased_formula():
    for n in [4, 32]:
        for c in range(n + 1):
            for k in [1, 4, 8, 16, 32]:
                if k <= n:
                    expected = 1 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0)
                    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_korean_ratio_excludes_math():
    assert korean_ratio(r"가나다 abc $xyz$ \[\text{latin}\]") == 0.5
    assert korean_ratio("123 + 456") == 0
    assert korean_ratio(r"한글 \boxed{English}") == 1


def test_metrics_null_empty_correct_group():
    records = [
        {
            "id": "x",
            "sample_index": i,
            "correct": False,
            "token_count": i + 1,
            "korean_response_ratio": 0.5,
        }
        for i in range(4)
    ]
    result = metrics(records, 4, [1, 4, 8, 16, 32])
    assert result["avg@4"] == 0
    assert result["pass@k"] == {"1": 0, "4": 0}
    assert result["response_length_tokens"] == {"all": 2.5, "correct": None, "incorrect": 2.5}
    with pytest.raises(ValueError, match="Incomplete"):
        metrics(records[:-1], 4, [1])


def test_protocol_frozen_between_stages(tmp_path):
    fingerprint = bind_protocol(tmp_path, {"seed": 42}, {"revision": "a"}, "stage0")
    assert bind_protocol(tmp_path, {"seed": 42}, {"revision": "a"}, "stage1") == fingerprint
    with pytest.raises(ValueError, match="changed"):
        bind_protocol(tmp_path, {"seed": 43}, {"revision": "a"}, "stage1")


def test_no_silent_context_truncation():
    with pytest.raises(ValueError, match="refusing to truncate"):
        check_context([1] * 9, {"max_new_tokens": 5, "max_model_len": 10})


def test_evaluation_records_metrics_and_resumes(tmp_path):
    class Engine:
        calls = 0

        def generate(self, problem, n, seed):
            self.calls += 1
            return "prompt", [
                {"text": r"답은 \boxed{42}", "token_count": 8, "finish_reason": "stop"},
                {"text": "답을 모르겠습니다", "token_count": 5, "finish_reason": "stop"},
            ]

    config = {
        "seed": 42,
        "verify_timeout_seconds": 5,
        "pass_k": [1, 4],
        "datasets": {"test": {"n": 2, "problem_column": "ko_problem", "answer_column": "answer"}},
    }
    rows = {"test": [{"id": 1, "ko_problem": "한국어 문제", "answer": "42"}]}
    path = tmp_path / "generations.jsonl"
    engine = Engine()
    results = evaluate(engine, rows, config, path)
    assert results["test"]["avg@2"] == 0.5
    assert results["test"]["korean_response_ratio"] == 1.0
    assert len(read_jsonl(path)) == 2
    assert evaluate(engine, rows, config, path) == results
    assert engine.calls == 1
