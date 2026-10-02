"""Native Llama chat formatting, reusing the existing assistant-only label checks."""

from train.sft_data import prepare_examples


def configure_tokenizer(tokenizer, config):
    """Use existing vocabulary tokens, without resizing or training embeddings."""
    vocabulary = tokenizer.get_vocab()
    for name, token in config["tokenizer"].items():
        if token not in vocabulary:
            raise ValueError(f"Missing native Llama {name}: {token}")
        setattr(tokenizer, name, token)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Llama padding must be distinct from the supervised end-of-turn token")
    if not tokenizer.chat_template:
        raise ValueError("The pinned Llama tokenizer must include its native chat template")
    return tokenizer


def prepare_llama_examples(tokenizer, rows, ids, config):
    # Llama's native template trims the outside of each message. The assistant
    # starts with <think>; only trailing whitespace from the final answer is affected.
    # Normalize that boundary before using the shared exact token/mask assertions.
    answer_column = config["data"]["columns"]["answer"]
    normalized = []
    changed = 0
    for row in rows:
        answer = row.get(answer_column)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Missing training text: {answer_column}")
        changed += answer != answer.rstrip()
        normalized.append({**row, answer_column: answer.rstrip()})
    examples, report, preview = prepare_examples(tokenizer, normalized, ids, config)
    report["native_template_trailing_whitespace_normalized_rows"] = changed
    return examples, report, preview


def select_smoke_examples(examples, report, config):
    smoke = config["smoke"]
    if smoke["selection"] != "longest" or not 0 < smoke["num_examples"] <= len(examples):
        raise ValueError("Smoke must select a positive number of longest retained examples")
    indices = sorted(range(len(examples)), key=lambda i: (-len(examples[i]["input_ids"]), i))
    indices = indices[:smoke["num_examples"]]
    report = dict(report, smoke={
        "selection": "longest",
        "ids": [report["kept_ids"][i] for i in indices],
        "lengths": [len(examples[i]["input_ids"]) for i in indices],
    })
    return [examples[i] for i in indices], report
