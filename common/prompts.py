"""The sole prompt definition for baseline evaluation and every training stage."""

INSTRUCTION = r"문제를 단계별로 풀고, 최종 답을 \boxed{} 안에 쓰세요."


def messages(problem: str) -> list[dict[str, str]]:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("A nonempty Korean problem is required")
    return [{"role": "user", "content": problem + "\n\n" + INSTRUCTION}]


def render_prompt(tokenizer, problem: str) -> str:
    if not tokenizer.chat_template:
        raise ValueError("Checkpoint has no chat template; refusing to invent one")
    return tokenizer.apply_chat_template(
        messages(problem), tokenize=False, add_generation_prompt=True
    )
