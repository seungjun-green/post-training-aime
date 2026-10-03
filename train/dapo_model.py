"""Model-specific token setup for the shared DAPO trainer."""

from common.english_prompts import messages, render_prompt


def configure_tokenizer(tokenizer, config):
    vocabulary = tokenizer.get_vocab()
    for name, token in config.get("tokenizer", {}).items():
        if name not in {"eos_token", "pad_token"} or token not in vocabulary:
            raise ValueError(f"Invalid or missing native tokenizer token: {name}={token}")
        setattr(tokenizer, name, token)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("The DAPO tokenizer must define EOS and padding")
    if config.get("tokenizer") and tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Configured padding must differ from EOS")
    if not tokenizer.chat_template:
        raise ValueError("The pinned model must include a chat template")
    tokenizer.padding_side = "left"
    return tokenizer


def training_prompt(tokenizer, problem, config):
    kwargs = config.get("prompt_template_kwargs", {})
    if not kwargs:
        return render_prompt(tokenizer, problem)
    # Same English instruction; use the native template with a fixed date.
    return tokenizer.apply_chat_template(messages(problem), tokenize=False,
                                        add_generation_prompt=True, **kwargs)


def stop_token_ids(tokenizer, config):
    tokens = config.get("generation", {}).get("stop_tokens")
    if tokens is None:
        return None  # Preserve original EXAONE generation behavior.
    if tokenizer is None:
        raise ValueError("A configured tokenizer is required to resolve generation stop tokens")
    vocabulary = tokenizer.get_vocab()
    if not tokens or any(token not in vocabulary for token in tokens):
        raise ValueError("Every configured stop token must exist in the native vocabulary")
    ids = list(dict.fromkeys(vocabulary[token] for token in tokens))
    if tokenizer.eos_token_id not in ids or tokenizer.pad_token_id in ids:
        raise ValueError("Stop tokens must include EOS and exclude padding")
    return ids
