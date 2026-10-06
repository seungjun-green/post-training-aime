"""Check the memory-bounded head against the ordinary full-vocabulary objective."""

from contextlib import nullcontext
from copy import deepcopy

import pytest


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_projection_logps_entropy_and_gradients_match(autocast, with_bias):
    torch = pytest.importorskip("torch")
    from train.dapo_logps import chunked_lm_head_logps

    torch.manual_seed(7)
    torch.set_num_threads(1)
    head = torch.nn.Linear(16, 131, bias=with_bias)
    reference = deepcopy(head)
    hidden = torch.randn(39, 16, requires_grad=True)
    original = hidden.detach().clone().requires_grad_()
    targets = torch.randint(0, 131, (39,))
    weights = torch.randn(39)
    def context():
        return torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
    with context():
        logits = reference(original).float() / 0.73
        expected = logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)
        entropy = logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)
        actual, actual_entropy = chunked_lm_head_logps(hidden, head, targets, 0.73, 8, True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_entropy, entropy, rtol=1e-6, atol=1e-6)
    assert not actual_entropy.requires_grad
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    # BF16 rounds each chunk's weight gradient before FP32 accumulation, whereas
    # the monolithic oracle rounds once. Compare within BF16 arithmetic precision.
    tolerance = {"rtol": 0.02, "atol": 0.02} if autocast else {"rtol": 2e-5, "atol": 2e-6}
    torch.testing.assert_close(hidden.grad, original.grad, **tolerance)
    for a, b in zip(head.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, **tolerance)
    with torch.no_grad(), context():
        lp, ent = chunked_lm_head_logps(hidden, head, targets, 0.73, 8)
    assert ent is None
    torch.testing.assert_close(lp, expected.detach(), rtol=1e-6, atol=1e-6)


def test_full_20480_token_completion_bounds_projection_and_saved_tensors(monkeypatch):
    torch = pytest.importorskip("torch")
    from train import dapo_logps

    torch.manual_seed(7)
    torch.set_num_threads(1)
    # Real completion length, small CPU-only vocabulary. Observe every forward
    # and recomputed backward projection rather than trying to allocate 12 GiB.
    head = torch.nn.Linear(8, 257, bias=False)
    hidden = torch.randn(20480, 8, requires_grad=True)
    targets = torch.randint(0, 257, (20480,))
    linear = dapo_logps.F.linear
    calls, saved = [], []
    def audit_linear(x, weight, bias=None):
        calls.append(x.shape[0])
        assert x.shape[0] <= 128
        return linear(x, weight, bias)
    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor
    monkeypatch.setattr(dapo_logps.F, "linear", audit_linear)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        logps, entropy = dapo_logps.chunked_lm_head_logps(hidden, head, targets, 1.0, 128, True)
    assert len(calls) == 160
    (-logps.mean()).backward()
    assert len(calls) == 320  # Backward recomputes each projection.
    assert (20480, 257) not in saved
    assert not any(len(shape) == 2 and shape[-1] == 257 for shape in saved)
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(head.weight.grad).all()
    assert entropy.shape == (20480,)
