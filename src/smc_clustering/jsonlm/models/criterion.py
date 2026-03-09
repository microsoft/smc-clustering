# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Constrained negative log-likelihood under grammar masks for teacher forcing.

This module masks out disallowed tokens at each step (per the grammar), renormalizes with log_softmax over the allowed
set, and gathers the gold token's log-probabilities. It raises errors if a position has no allowed tokens or if
a gold token is disallowed. The primary entry point is `constrained_nll`.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def apply_mask_and_logprobs(logits: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    """Apply a Boolean allowed-token mask to logits and return log-probabilities.

    The mask is True for allowed tokens and False for disallowed ones. Disallowed logits are set to -inf prior to
    log_softmax so that probability mass renormalizes strictly over the allowed subset.

    Shapes:
        logits: [B, T, V] float
        mask:   [B, T, V] bool

    Returns:
        log_probs: [B, T, V] float (invalid positions remain -inf)

    Raises:
        ValueError: If any timestep has no allowed tokens (mask all False along V).
    """
    assert logits.dim() == 3, f"Expected logits [B, T, V], got {tuple(logits.shape)}"
    assert mask.shape == logits.shape, (
        f"Mask shape {tuple(mask.shape)} must match logits {tuple(logits.shape)}"
    )

    # Ensure at least one allowed token per position to avoid NaNs from log_softmax(all -inf).
    allowed_any = mask.any(dim=-1)  # [B, T]
    if not torch.all(allowed_any):
        bad = (~allowed_any).nonzero(as_tuple=False)  # [K, 2] of (b, t) items
        bt_pairs = [(int(b.item()), int(t.item())) for b, t in bad]
        raise ValueError(f"No allowed tokens at positions (batch, time): {bt_pairs}")

    # Mask out invalid tokens by setting -inf prior to log_softmax.
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=-1)
    return log_probs


def constrained_nll(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    masks: torch.BoolTensor,
    reduction: Literal["mean", "sum"] = "mean",
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute constrained NLL under grammar masks with teacher forcing.

    At each step, disallowed tokens are set to -inf, logits are renormalized via log_softmax over the allowed subset,
    and the negative log-probability of the gold token is taken.

    Args:
        logits: Unnormalized model outputs of shape [B, T, V] (float32/float64).
        target_ids: Gold token IDs of shape [B, T] (long); gathered against `logits`.
        masks: Boolean allowed-token masks of shape [B, T, V]; True means "allowed".
        reduction: 'mean' (default) averages NLL over BxT; 'sum' sums over BxT.
        weights: Optional per-position weights of shape [B, T]; if provided, NLL is weighted.

    Returns:
        loss: Scalar tensor (mean or sum over BxT).
        nll_per_token: Tensor of shape [B, T] with per-position NLLs.

    Raises:
        ValueError: If shapes mismatch, a timestep has no allowed tokens, or a gold token is disallowed.
    """
    assert logits.dim() == 3, f"logits must be [B, T, V], got {tuple(logits.shape)}"
    assert target_ids.dim() == 2, f"target_ids must be [B, T], got {tuple(target_ids.shape)}"
    assert masks.shape == logits.shape, (
        f"masks must match logits; got {tuple(masks.shape)} vs {tuple(logits.shape)}"
    )
    B, T, V = logits.shape
    assert target_ids.shape == (B, T), "target_ids shape must be [B, T]"
    if weights is not None:
        assert weights.shape == (B, T), f"weights must be [B, T], got {tuple(weights.shape)}"

    # Compute masked log-probabilities.
    log_probs = apply_mask_and_logprobs(logits, masks)  # [B, T, V]

    # Gather gold token log-probs. Shape: [B, T, 1] -> [B, T]
    gold_lp = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # If any gold token is disallowed, its log-prob will be -inf; raise a helpful error.
    if torch.isinf(gold_lp).any():
        bad = torch.nonzero(torch.isinf(gold_lp), as_tuple=False)  # [K, 2]
        bt_pairs = [(int(b.item()), int(t.item())) for b, t in bad]
        raise ValueError(f"Gold token is disallowed by mask at positions (batch, time): {bt_pairs}")

    nll = -gold_lp  # [B, T]

    if weights is not None:
        nll = nll * weights
        if reduction == "mean":
            denom = weights.sum().clamp_min(1.0)
            loss = nll.sum() / denom
        elif reduction == "sum":
            loss = nll.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction!r}")
    elif reduction == "mean":
        loss = nll.mean()
    elif reduction == "sum":
        loss = nll.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")
    return loss, nll


def invalid_mass(logits: torch.Tensor, masks: torch.BoolTensor) -> torch.Tensor:
    """Return the probability mass assigned to disallowed tokens at each position.

    This is computed from the unmasked softmax over logits; it is a diagnostic metric, not used in the loss.

    Shapes:
        logits: [B, T, V] float
        masks:  [B, T, V] bool

    Returns:
        disallowed_mass: [B, T] float in [0, 1], equal to sum_{v: ~mask} softmax(logits)[v].
    """
    assert logits.shape == masks.shape and logits.dim() == 3, "Shapes must match and be [B, T, V]"
    probs = torch.softmax(logits, dim=-1)  # [B, T, V]
    disallowed = (~masks).to(dtype=probs.dtype)
    mass = (probs * disallowed).sum(dim=-1)  # [B, T]
    # Numerical safety: clamp to [0, 1].
    return mass.clamp(min=0.0, max=1.0)
