"""Fixed English evaluation prompt, shared by baseline and later English stages."""

INSTRUCTION = (
    r"Solve the problem step by step in English and put your final answer inside \boxed{}."
)


def messages(problem):
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("A nonempty English problem is required")
    return [{"role": "user", "content": problem + "\n\n" + INSTRUCTION}]


def render_prompt(tokenizer, problem):
    if not tokenizer.chat_template:
        raise ValueError("Checkpoint has no chat template")
    return tokenizer.apply_chat_template(
        messages(problem), tokenize=False, add_generation_prompt=True
    )
