"""
Batched scoring utilities for grammar-constrained log-likelihoods and Δ-scores.

This module provides:
  * score_entities_batched(...): compute constrained log-likelihoods for many entities in batches.
  * compute_deltas_batched(...): compute Δ = logP(AuB) - logP(A) - logP(B) for many pairs efficiently.

Internally we:
  * Canonicalize + serialize entities to ensure stable/identifiable scoring.
  * Encode to BOS…EOS IDs.
  * Build EOS-padded batches and per-step allowed-token masks (aligned to targets).
  * Apply the constrained log-softmax and gather gold log-probabilities.

These utilities mirror the single-entity logic in `jsonlm.api.logprob_entity`, but operate on batches for speed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from time import perf_counter as pc
from typing import Literal

import torch
from torch import nn

from jsonlm.models.criterion import apply_mask_and_logprobs
from jsonlm.serialization.encoder import (
    canonicalize_entity,
    entities_to_string_as_set,
    entity_to_string,
    parse_entity,
    parse_sequence,
)
from jsonlm.tokenization.tokenizer import JsonLMTokenizer


logger = logging.getLogger(__name__)


def _ids_from_entity_sequence_or_text_batch(
    items: Sequence[dict[str, list[str]] | list[dict[str, list[str]]] | str],
    tokenizer: JsonLMTokenizer,
) -> list[list[int]]:
    """Convert dict/list/string inputs to BOS…EOS ID sequences (canonicalized & serialized).

    Supports:
    - dict[str, list[str]]: Single entity
    - list[dict[str, list[str]]]: Entity sequence
    - str: Serialized entity or sequence
    """
    out: list[list[int]] = []
    for x in items:
        if isinstance(x, dict):
            # Single entity
            s = entity_to_string(canonicalize_entity(x))
        elif isinstance(x, list):
            # Entity sequence
            s = entities_to_string_as_set(x)  # canonicalization happens inside
        else:
            # String - need to determine if it's a single entity or sequence
            # Try parsing as sequence first, fall back to single entity
            try:
                can_list = parse_sequence(x)
                s = entities_to_string_as_set(can_list)
            except ValueError:
                # Fall back to single entity parsing
                can = parse_entity(x)
                s = entity_to_string(can)
        out.append(tokenizer.encode(s, add_bos_eos=True))
    return out


@torch.no_grad()
def _pad_to_tensor(ids_list: list[list[int]], eos_id: int, pad_id: int) -> torch.Tensor:
    """Return [B, L_max] long tensor from a list of ID lists; tail filled with EOS for clean masking."""
    B = len(ids_list)
    Lm = max(len(x) for x in ids_list)
    out = torch.full((B, Lm), pad_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        L = len(ids)
        out[i, :L] = torch.tensor(ids, dtype=torch.long)
        if Lm > L:
            out[i, L:] = eos_id
    return out


@torch.no_grad()
def score_entities_batched(
    entities_sequences_or_texts: Sequence[dict[str, list[str]] | list[dict[str, list[str]]] | str],
    model: nn.Module,
    tokenizer: JsonLMTokenizer,
    normalize: Literal["sum", "mean", "bpt"] = "sum",
    include_eos: bool = False,
    offset: float = 0,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> list[float]:
    """Compute constrained log-likelihoods for many entities or entity sequences using batches.

    For each input (entity, entity sequence, or serialized string), we:
      1) canonicalize & serialize deterministically,
      2) encode to BOS…EOS IDs,
      3) build EOS-padded batches and per-step allowed-token masks,
      4) apply masked log_softmax and gather gold token log-probabilities.

    Normalization and EOS handling:
      * When include_eos=True: includes EOS token in scoring
      * When include_eos=False: excludes EOS token from scoring (useful for sequence content scoring)
      * 'sum'  → total log-probability (nats) over selected targets
      * 'mean' → average per-token (nats) using the number of active targets per item
      * 'bpt'  → bits per token (mean / ln 2)

    Args:
        entities_sequences_or_texts: Each element is either:
            - dict[str, list[str]]: Single entity
            - list[dict[str, list[str]]]: Entity sequence
            - str: Serialized entity or sequence (with <K>/<V>)
        model: Next-token LM mapping [B, T] → [B, T, V].
        tokenizer: Tokenizer over specials + BPE.
        normalize: 'sum' | 'mean' | 'bpt'.
        include_eos: Whether to include EOS token in scoring. If False, stops at first EOS.
        offset: Add this constant to each final score (for calibration).
        batch_size: Number of items per forward pass.
        device: Torch device; defaults to the model's first parameter device.

    Returns:
        A list of floats with one score per input item (same order as inputs).

    Raises:
        ValueError: If a gold token is disallowed by the grammar masks.
    """
    if device is None:
        try:
            device = next(model.parameters()).device  # type: ignore[assignment]
        except StopIteration:
            device = torch.device("cpu")

    scores: list[float] = []
    V = len(tokenizer)
    eos_id = tokenizer.vocabulary.eos_id
    pad_id = tokenizer.vocabulary.pad_id

    # Mini-batch loop
    for i in range(0, len(entities_sequences_or_texts), batch_size):
        start = pc()
        chunk = entities_sequences_or_texts[i : i + batch_size]

        # Encode each item to BOS…EOS IDs (canonicalized & serialized).
        ids_list = _ids_from_entity_sequence_or_text_batch(chunk, tokenizer)  # list[[L_i]]

        # Rectangularize with EOS-filled tails for clean masking post-EOS.
        batch = _pad_to_tensor(ids_list, eos_id=eos_id, pad_id=pad_id).to(device)  # [B, L]
        input_ids = batch[:, :-1]  # [B, T]
        target_ids = batch[:, 1:]  # [B, T]
        logger.debug(f"Prepared batch of size {input_ids.shape[0]} in {pc() - start:.3f}s")

        # Model forward and basic shape checks.
        start = pc()
        logits = model(input_ids)  # [B, T, V]
        logger.debug(f"Scored batch of size {input_ids.shape[0]} in {pc() - start:.3f}s")

        assert (
            logits.dim() == 3
            and logits.shape[0] == input_ids.shape[0]
            and logits.shape[1] == input_ids.shape[1]
            and logits.shape[2] == V
        ), f"Bad logits shape {tuple(logits.shape)} for input {tuple(input_ids.shape)} and V={V}"

        # Grammar masks aligned to targets (handles post-EOS by allowing EOS-only tails).
        start = pc()

        from jsonlm.grammar.runtime import get_runtime

        rt = get_runtime(tokenizer, device)
        masks = rt.build_masks(batch)  # [B, T, V]

        logger.debug(f"Built masks for batch of size {input_ids.shape[0]} in {pc() - start:.3f}s")

        # Constrained log-probs and gather the gold token at each step.
        start = pc()
        log_probs = apply_mask_and_logprobs(logits, masks)  # [B, T, V]
        gold_lp = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
        logger.debug(f"Computed log-probs for batch of size {input_ids.shape[0]} in {pc() - start:.3f}s")

        start = pc()
        # Disallowed golds appear as -inf; surface a helpful error.
        if torch.isinf(gold_lp).any():
            raise ValueError("Encountered disallowed gold token under grammar masks during batched scoring.")

        # Handle EOS inclusion/exclusion
        if include_eos:
            # Include all target steps (incl. EOS); rectangular tail contributes EOS terms only.
            lp_sum = gold_lp.sum(dim=1)  # [B]
            active_T = torch.tensor([len(ids) - 1 for ids in ids_list], dtype=gold_lp.dtype, device=gold_lp.device)
        else:
            # Compute per-row first_eos_index from targets
            B, T = target_ids.shape
            eos_mask = target_ids == eos_id  # [B, T]
            # Find first EOS position per row
            eos_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
            first_eos_index = torch.where(eos_mask, eos_indices, torch.full_like(eos_indices, T)).amin(dim=1)  # [B]

            # Build weight mask: w[b,t] = 1 if t < first_eos_index[b], else 0
            time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
            weight_mask = (time_indices < first_eos_index.unsqueeze(1)).float()  # [B, T]

            # Apply weight mask and sum
            lp_sum = (gold_lp * weight_mask).sum(dim=1)  # [B]
            active_T = weight_mask.sum(dim=1).clamp(min=1)  # [B], prevent division by zero

        if normalize == "sum":
            scores.extend([float(x) for x in lp_sum])
        else:
            lp_mean = lp_sum / active_T

            if normalize == "mean":
                scores.extend([float(x) for x in lp_mean])
            elif normalize == "bpt":
                ln2 = torch.log(torch.tensor(2.0, dtype=lp_mean.dtype, device=lp_mean.device))
                scores.extend([float(x / ln2) for x in lp_mean])
            else:
                raise ValueError(f"Unknown normalize mode: {normalize!r}")

        logger.debug(f"Computed final scores for batch of size {input_ids.shape[0]} in {pc() - start:.3f}s")

    if offset:
        scores = [s + offset for s in scores]

    return scores


@torch.no_grad()
def compute_deltas_batched(
    pairs: Sequence[
        tuple[
            dict[str, list[str]] | list[dict[str, list[str]]],
            dict[str, list[str]] | list[dict[str, list[str]]],
        ]
    ],
    model: nn.Module,
    tokenizer: JsonLMTokenizer,
    offset: float = 0,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> list[float]:
    """Compute Δ = logP(AuB) - logP(A) - logP(B) for many pairs in batches.

    Args:
        pairs: Iterable of (A, B), where each side is either a canonicalizable
            entity dict or a list of such dicts (entity sequence).
        model: Next-token LM.
        tokenizer: Tokenizer.
        offset: Add this constant to each score (for calibration).
        batch_size: Batch size for scoring.
        device: Torch device.

    Returns:
        List of Δ values in nats, one per input pair (same order).
    """

    # Build canonical serialized strings for A, B, and AuB; keep mapping to input order.
    def _serialize_side(x: dict[str, list[str]] | list[dict[str, list[str]]]) -> str:
        if isinstance(x, list):
            # entities_to_string handles canonicalization internally
            return entities_to_string_as_set(x)
        return entity_to_string(canonicalize_entity(x))

    start = pc()

    serialized_a: list[str] = []
    serialized_b: list[str] = []
    serialized_union: list[str] = []

    for a, b in pairs:
        sa = _serialize_side(a)
        sb = _serialize_side(b)

        # Union: if either side is a list, concatenate lists; else merge dicts
        if isinstance(a, list) or isinstance(b, list):
            a_list = a if isinstance(a, list) else [a]
            b_list = b if isinstance(b, list) else [b]
            su = entities_to_string_as_set(a_list + b_list)
        else:
            merged: dict[str, list[str]] = {}
            for k, vs in a.items():
                merged.setdefault(k, []).extend(vs)
            for k, vs in b.items():
                merged.setdefault(k, []).extend(vs)
            su = entity_to_string(canonicalize_entity(merged))

        serialized_a.append(sa)
        serialized_b.append(sb)
        serialized_union.append(su)

    logger.debug(
        f"Prepared {len(pairs)} pairs (3x{len(pairs)} entities) in {pc() - start:.3f}s",
    )

    # Deduplicate all unique strings to score.
    start = pc()
    all_strings: list[str] = []
    uniq_index: dict[str, int] = {}
    for s in serialized_a + serialized_b + serialized_union:
        if s not in uniq_index:
            uniq_index[s] = len(all_strings)
            all_strings.append(s)

    logger.debug(
        f"Found {len(all_strings)} unique entities to score in {pc() - start:.3f}s",
    )

    # Score all uniques in batches.
    uniq_scores = score_entities_batched(
        all_strings,
        model=model,
        tokenizer=tokenizer,
        normalize="sum",
        include_eos=True,
        offset=offset,
        batch_size=batch_size,
        device=device,
    )

    # Map string -> score
    score_of = {s: uniq_scores[idx] for s, idx in uniq_index.items()}

    # Assemble Δ for each pair with cached scores.
    deltas: list[float] = []
    for sa, sb, su in zip(serialized_a, serialized_b, serialized_union, strict=False):
        deltas.append(score_of[su] - score_of[sa] - score_of[sb])

    return deltas
