"""Preserve original s1K text and construct explicit assistant-only labels."""

import json
import math
from collections import Counter
from pathlib import Path

from common.english_prompts import messages, render_prompt
from common.io import digest


def load_rows(config, token=None, local_rows=None):
    """A local preparation export is supported for offline, CPU-only data inspection."""
    spec = config["data"]
    if local_rows is not None:
        envelopes = [json.loads(line) for line in Path(local_rows).read_text().splitlines()]
        rows = [row["original"] for row in envelopes]
        ids = [row["id"] for row in envelopes]
    else:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download

        rows = list(load_dataset(
            spec["repo"], name=spec["config"], split=spec["split"],
            revision=spec["revision"], token=token,
        ))
        provenance = json.loads(Path(hf_hub_download(
            spec["repo"], "provenance.json", repo_type="dataset",
            revision=spec["revision"], token=token,
        )).read_text())
        if provenance["retained_originals_digest"] != spec.get("provenance_content_digest", spec["content_digest"]):
            raise ValueError("Published provenance does not match the configured s1K content")
        ids = provenance["retained_preparation_ids"]
    if len(rows) != spec["expected_rows"] or digest(rows) != spec["content_digest"]:
        raise ValueError("Training rows differ from the pinned, decontaminated s1K dataset")
    if len(ids) != len(rows) or len(set(ids)) != len(ids) or digest(ids) != spec["ids_digest"]:
        raise ValueError("Training IDs differ from the decontamination provenance")
    return rows, ids


DEFAULT_COLUMNS = {"question": "question", "reasoning": "deepseek_thinking_trajectory",
                   "answer": "deepseek_attempt"}


def format_example(tokenizer, row, columns=None):
    columns = columns or DEFAULT_COLUMNS
    for column in columns.values():
        if not isinstance(row[column], str) or not row[column].strip():
            raise ValueError(f"Missing training text: {column}")
    assistant = (
        "<think>\n" + row[columns["reasoning"]]
        + "\n</think>\n\n" + row[columns["answer"]]
    )
    conversation = messages(row[columns["question"]]) + [{"role": "assistant", "content": assistant}]
    prompt = render_prompt(tokenizer, row[columns["question"]])
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    supervised_end = prompt + assistant + tokenizer.eos_token
    # The native EXAONE template appends a newline after EOT. It stays in the input,
    # but is template text, so it must not receive a loss label.
    if not text.startswith(supervised_end) or text[len(supervised_end):].strip():
        raise ValueError("Native template does not place the assistant and EOT after the eval prompt")
    def encode(value):
        return tokenizer.encode(value, add_special_tokens=False, truncation=False)
    input_ids = encode(text)
    prefix_ids, end_ids = encode(prompt), encode(supervised_end)
    start, end = len(prefix_ids), len(end_ids)
    if input_ids[:start] != prefix_ids or input_ids[:end] != end_ids:
        raise ValueError("Tokenizer merges across an assistant/loss boundary")
    if end <= start or input_ids[end - 1] != tokenizer.eos_token_id:
        raise ValueError("Assistant loss must end at the native EOT token")
    # Compare IDs: the native tokenizer applies Unicode NFKC normalization, so a
    # decode/text comparison can reject valid unmodified mathematical source text.
    if input_ids[start:end - 1] != encode(assistant):
        raise ValueError("Loss does not start exactly at the assistant content")
    labels = [-100] * start + input_ids[start:end] + [-100] * (len(input_ids) - end)
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}, text


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def prepare_examples(tokenizer, rows, ids, config):
    examples, lengths, kept_ids, dropped = [], [], [], []
    columns = config["data"].get("columns", DEFAULT_COLUMNS)
    preview = None
    for row, identifier in zip(rows, ids, strict=True):
        if config["data"].get("drop_missing_outputs", False):
            missing = [columns[key] for key in ["reasoning", "answer"]
                       if not isinstance(row.get(columns[key]), str) or not row[columns[key]].strip()]
            if missing:
                dropped.append({"id": identifier, "reason": "missing_output", "columns": missing})
                continue
        example, text = format_example(tokenizer, row, columns)
        length = len(example["input_ids"])
        if length > config["max_seq_length"]:
            dropped.append({"id": identifier, "tokens": length})
            continue
        if len(examples) == config["sanity_example_index"]:
            preview = (example, text)
        examples.append(example)
        lengths.append(length)
        kept_ids.append(identifier)
    if not examples or preview is None:
        raise ValueError("No kept example available at sanity_example_index")
    grades = Counter(str(row.get("deepseek_grade")) for row in rows)
    report = {
        "model": config["model"],
        "dataset": config["data"], "input_rows": len(rows), "kept_rows": len(examples),
        "dropped_rows": len(dropped), "dropped": dropped, "kept_ids": kept_ids,
        "grade_counts": dict(grades),
        "incorrect_grade_rows": sum(grades[v] for v in config["data"]["incorrect_grade_values"]),
        "grade_filter_applied": False, "truncation_applied": False,
        "training_columns": columns,
        "dropped_missing_outputs": sum(row.get("reason") == "missing_output" for row in dropped),
        "dropped_overlength": sum("tokens" in row for row in dropped),
        "max_seq_length": config["max_seq_length"],
        "token_lengths": {
            "min": min(lengths), "median": percentile(lengths, 0.5),
            "p90": percentile(lengths, 0.9), "p99": percentile(lengths, 0.99),
            "max": max(lengths), "total": sum(lengths),
        },
        "examples_digest": digest(examples),
    }
    return examples, report, preview


def sanity_text(tokenizer, preview, placeholder):
    example, text = preview
    labels, ids = example["labels"], example["input_ids"]
    indices = [i for i, label in enumerate(labels) if label != -100]
    start, end = indices[0], indices[-1] + 1
    assert indices == list(range(start, end))
    assert labels[start:end] == ids[start:end]
    assert ids[end - 1] == tokenizer.eos_token_id
    visible = (
        placeholder * start
        + tokenizer.decode(ids[start:end], clean_up_tokenization_spaces=False)
        + placeholder * (len(ids) - end)
    )
    return f"FULL FORMATTED TEXT\n{text}\nLOSS REGION (one placeholder per masked token)\n{visible}\n"


class AssistantCollator:
    """Pad labels by position, preserving supervised EOT even if pad == eos."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch

        width = max(len(row["input_ids"]) for row in examples)
        return {
            key: torch.tensor([
                row[key] + [fill] * (width - len(row[key])) for row in examples
            ], dtype=torch.long)
            for key, fill in [("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)]
        }
