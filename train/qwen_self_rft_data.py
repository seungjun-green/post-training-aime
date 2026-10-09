"""Pinned self-RFT rows and native-EOS completion targets for Qwen base SFT."""

import argparse
import os
from pathlib import Path

from common.english_prompts import render_prompt
from common.io import digest
from train.qwen_sft_data import BUNDLE_FILES as QWEN_FILES, prepare_hf_run
from train.sft_data import percentile
from eval.answer_only import BUNDLE_FILES as EVALUATOR_FILES

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = list(dict.fromkeys(QWEN_FILES + EVALUATOR_FILES + [
    "train/qwen_self_rft_data.py", "train/stage1_qwen_self_rft.py",
    "configs/sft_qwen25_3b_self_rft.yaml", "configs/qwen_self_rft_eval.yaml",
    "train/self_rft_callbacks.py", "train/self_rft_display.py", "train/self_rft_epochs.py",
]))


def training_code_digest():
    return digest({name: (ROOT / name).read_text() for name in BUNDLE_FILES})


def configure_tokenizer(tokenizer, config):
    if config.get("tokenizer") or config.get("target_format") != "native_eos_completion":
        raise ValueError("Preserve native tokenizer settings; use native_eos_completion")
    if (tokenizer.eos_token != "<|endoftext|>" or tokenizer.pad_token != "<|endoftext|>"
            or not tokenizer.chat_template):
        raise ValueError("Expected Qwen base's original endoftext EOS/PAD and chat template")
    tokenizer.padding_side = "right"
    return tokenizer


def format_example(tokenizer, row):
    problem, response = row["problem"], row["response"]
    if any(not isinstance(v, str) or not v.strip() for v in (problem, response)):
        raise ValueError("Each self-RFT row requires nonempty problem and response text")
    # Use the shipped chat template for the inference prompt. Supervise a completion
    # with the original model EOS, rather than the template's assistant im_end suffix.
    if any(token in response for token in ("<|endoftext|>", "<|im_start|>", "<|im_end|>")):
        raise ValueError("Response contains an embedded chat/EOS token")
    prompt = render_prompt(tokenizer, problem)
    text = prompt + response + tokenizer.eos_token
    def encode(value):
        return tokenizer.encode(value, add_special_tokens=False, truncation=False)
    prefix, target, ids = encode(prompt), encode(response), encode(text)
    if ids != prefix + target + [tokenizer.eos_token_id]:
        raise ValueError("Tokenizer merges across prompt/response/EOS boundaries")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + target + [tokenizer.eos_token_id]}, text


def prepare_examples(tokenizer, rows, ids, config):
    examples, lengths, kept, dropped = [], [], [], []
    preview = None
    for row, identifier in zip(rows, ids, strict=True):
        example, text = format_example(tokenizer, row)
        length = len(example["input_ids"])
        if length > config["max_seq_length"]:
            dropped.append({"id": identifier, "tokens": length})
            continue
        if len(examples) == config["sanity_example_index"]:
            preview = example, text
        examples.append(example)
        lengths.append(length)
        kept.append(identifier)
    if not examples or preview is None:
        raise ValueError("No retained example for sanity preview")
    return examples, {
        "model": config["model"], "dataset": config["data"],
        "input_rows": len(rows), "kept_rows": len(examples),
        "unique_problems": len({row["problem"] for row in rows}),
        "dropped_rows": len(dropped), "dropped": dropped, "kept_ids": kept,
        "grade_filter_applied": False, "truncation_applied": False,
        "deduplication_applied": False, "training_columns": config["data"]["columns"],
        "target_format": config["target_format"], "dropped_missing_outputs": 0,
        "dropped_overlength": len(dropped), "max_seq_length": config["max_seq_length"],
        "token_lengths": {"min": min(lengths), "median": percentile(lengths, .5),
                          "p90": percentile(lengths, .9), "p99": percentile(lengths, .99),
                          "max": max(lengths), "total": sum(lengths)},
        "examples_digest": digest(examples),
    }, preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    print("Pinned training config:", prepare_hf_run(
        args.config, args.output_root, args.run_name, token=os.getenv("HF_TOKEN")))


if __name__ == "__main__":
    main()
