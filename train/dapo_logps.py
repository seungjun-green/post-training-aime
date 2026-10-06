"""Checkpoint the vocabulary projection as well as the token log-probabilities."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def projected_logps(hidden, weight, bias, targets, temperature, compute_entropy):
    # Match the ordinary autocast LM head followed by FP32 log-prob arithmetic.
    # Crucially, this projection is INSIDE the checkpoint, so neither logits nor
    # their full-sequence gradient ever has shape [completion_length, vocabulary].
    scaled = F.linear(hidden, weight, bias).float() / temperature
    normalizer = scaled.logsumexp(-1)
    logps = scaled.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - normalizer
    entropy = None
    if compute_entropy:
        with torch.no_grad():
            entropy = normalizer.detach() - (scaled.softmax(-1) * scaled).sum(-1)
    return logps, entropy


def chunked_lm_head_logps(hidden, head, targets, temperature, chunk_tokens, compute_entropy=False):
    if not isinstance(head, torch.nn.Linear):
        raise TypeError("Chunked Qwen projection requires its native linear LM head")
    if hidden.ndim != 2 or targets.shape != hidden.shape[:1] or chunk_tokens < 1:
        raise ValueError("Expected aligned hidden states/targets and a positive chunk size")
    logps, entropies = [], []
    for start in range(0, len(targets), chunk_tokens):
        args = (hidden[start:start + chunk_tokens], head.weight, head.bias,
                targets[start:start + chunk_tokens], temperature, compute_entropy)
        if torch.is_grad_enabled():
            lp, entropy = checkpoint(projected_logps, *args, use_reentrant=False)
        else:
            lp, entropy = projected_logps(*args)
        logps.append(lp)
        if compute_entropy:
            entropies.append(entropy)
    return torch.cat(logps), torch.cat(entropies) if compute_entropy else None
