"""
Unit tests for constrained NLL: renormalization over allowed tokens and helpful error paths.

We hand-check a tiny example to verify probabilities renormalize strictly over allowed tokens, and confirm that
disallowed gold tokens and mask rows with no allowed tokens produce clear ValueErrors.
"""

from __future__ import annotations

import math

import pytest
import torch

from smc_clustering.jsonlm.models.criterion import apply_mask_and_logprobs, constrained_nll, invalid_mass


def test_apply_mask_and_logprobs_manual_check() -> None:
    """Single-step renormalization matches a manual softmax over allowed tokens."""
    # Shapes: B=1, T=1, V=4
    logits = torch.tensor([[[0.0, 2.0, -1.0, 0.0]]], dtype=torch.float32)  # [1,1,4]
    mask = torch.tensor([[[True, True, False, False]]], dtype=torch.bool)  # allow ids 0 and 1 only

    logp = apply_mask_and_logprobs(logits, mask)  # [1,1,4]

    # Manual: softmax over {0,1} with logits {0,2}.
    Z = math.exp(0.0) + math.exp(2.0)
    p0 = math.exp(0.0) / Z
    p1 = math.exp(2.0) / Z

    # Check probabilities of allowed tokens; disallowed should be -inf in log-space.
    assert torch.isclose(logp[0, 0, 0].exp(), torch.tensor(p0, dtype=torch.float32), rtol=1e-6, atol=1e-6)
    assert torch.isclose(logp[0, 0, 1].exp(), torch.tensor(p1, dtype=torch.float32), rtol=1e-6, atol=1e-6)
    assert torch.isneginf(logp[0, 0, 2])
    assert torch.isneginf(logp[0, 0, 3])


def test_constrained_nll_matches_manual() -> None:
    """Gathered NLL equals -log of the renormalized probability for the gold token."""
    logits = torch.tensor([[[0.0, 2.0, -1.0, 0.0]]], dtype=torch.float32)  # [1,1,4]
    mask = torch.tensor([[[True, True, False, False]]], dtype=torch.bool)
    target = torch.tensor([[1]], dtype=torch.long)  # gold id 1 is allowed

    loss, nll = constrained_nll(logits, target, mask, reduction="mean")
    # Manual expected NLL for id 1: -log p1
    Z = math.exp(0.0) + math.exp(2.0)
    p1 = math.exp(2.0) / Z
    expected = -math.log(p1)
    assert torch.isclose(loss, torch.tensor(expected, dtype=torch.float32), rtol=1e-6, atol=1e-6)
    assert torch.isclose(nll[0, 0], torch.tensor(expected, dtype=torch.float32), rtol=1e-6, atol=1e-6)


def test_constrained_nll_disallowed_gold_raises() -> None:
    """If the gold token is masked out at any position, raise a clear ValueError."""
    logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    mask = torch.tensor([[[True, False, False], [True, False, False]]], dtype=torch.bool)
    target = torch.tensor([[0, 2]], dtype=torch.long)  # position 1 has gold=2 disallowed
    with pytest.raises(ValueError) as exc:
        _ = constrained_nll(logits, target, mask)
    assert "disallowed by mask" in str(exc.value).lower()


def test_apply_mask_and_logprobs_no_allowed_row_raises() -> None:
    """If a timestep has no allowed tokens, the API raises before log_softmax."""
    logits = torch.zeros((1, 1, 4), dtype=torch.float32)
    mask = torch.tensor([[[False, False, False, False]]], dtype=torch.bool)
    with pytest.raises(ValueError) as exc:
        _ = apply_mask_and_logprobs(logits, mask)
    assert "no allowed tokens" in str(exc.value).lower()


def test_invalid_mass_in_range() -> None:
    """invalid_mass returns a value in [0,1] and matches the sum of softmax over disallowed tokens."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    # Allow first three tokens, disallow last two.
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[..., :3] = True
    mass = invalid_mass(logits, mask)  # [2,3]
    assert mass.min().item() >= 0.0 and mass.max().item() <= 1.0
    # Spot check a position against explicit computation.
    probs = torch.softmax(logits, dim=-1)
    expected = probs[0, 0, 3:].sum()
    assert torch.isclose(mass[0, 0], expected, rtol=1e-6, atol=1e-6)
