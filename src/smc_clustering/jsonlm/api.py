# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public API surface for encoding entities and computing constrained log-likelihoods.

This module exposes core entry points for both single entities and entity sequences: `encode_entity`/`encode_sequence`
(raw dict/list → BOS…EOS IDs) and `logprob_entity`/`logprob_sequence` (teacher-forced log-likelihood under grammar
constraints). Canonicalization happens on the fly so callers can pass non-canonical dicts safely. Strings are parsed
and re-serialized to ensure deterministic scoring. Sequence functions support Kleene-plus grammar with optional EOS exclusion.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from smc_clustering.jsonlm.grammar.automaton import GrammarAutomaton, GrammarState
from smc_clustering.jsonlm.grammar.mask import allowed_token_mask
from smc_clustering.jsonlm.models.criterion import apply_mask_and_logprobs
from smc_clustering.jsonlm.models.decode import decode_greedy
from smc_clustering.jsonlm.serialization.encoder import (
    canonicalize_entity,
    entities_to_string_as_set,
    entity_to_string,
    parse_entity,
    parse_sequence,
)
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer


def encode_entity(
    entity: dict[str, list[str]], tokenizer: JsonLMTokenizer, add_bos_eos: bool = True
) -> list[int]:
    """Convert a raw entity dict into BOS…EOS token IDs using the given tokenizer.

    Canonicalization is performed on the fly, then the entity is serialized with <K>/<V> sentinels and tokenized.

    Args:
        entity: Mapping from keys to list-of-string values.
        tokenizer: Tokenizer that understands the sentinel serialization.
        add_bos_eos: Whether to add BOS/EOS boundary tokens.

    Returns:
        A list of token IDs representing the serialized entity.
    """
    can = canonicalize_entity(entity)
    s = entity_to_string(can)
    return tokenizer.encode(s, add_bos_eos=add_bos_eos)


def encode_sequence(
    entities: list[dict[str, list[str]]],
    tokenizer: JsonLMTokenizer,
    add_bos_eos: bool = True,
) -> list[int]:
    """Convert a list of entity dicts into BOS…EOS token IDs using the given tokenizer.

    Each entity is canonicalized on the fly, then the sequence is serialized with <K>/<V> sentinels using
    entities_to_string and tokenized with a single BOS at the start and EOS at the end.

    Args:
        entities: List of entity mappings from keys to list-of-string values.
        tokenizer: Tokenizer that understands the sentinel serialization.
        add_bos_eos: Whether to add BOS/EOS boundary tokens.

    Returns:
        A list of token IDs representing the serialized entity sequence.
    """
    # Canonicalization happens inside entities_to_string via entity_to_string calls
    s = entities_to_string_as_set(entities)
    return tokenizer.encode(s, add_bos_eos=add_bos_eos)


def _build_masks_for_ids(ids_with_eos: list[int], tokenizer: JsonLMTokenizer) -> torch.BoolTensor:
    """Build [1, T, V] masks aligned to targets for a single BOS…EOS sequence of IDs."""
    assert len(ids_with_eos) >= 2, "Need at least BOS and EOS"
    V = len(tokenizer)
    T = len(ids_with_eos) - 1  # number of targets (includes EOS position)
    automaton = GrammarAutomaton(tokenizer)

    masks = torch.zeros((1, T, V), dtype=torch.bool)
    gs: GrammarState = automaton.start()
    # For each target token y_t, compute allowed mask at step t then step on y_t if not EOS
    for t in range(T):
        y_t = ids_with_eos[t + 1]
        masks[0, t] = allowed_token_mask(gs, automaton, tokenizer)
        if y_t == tokenizer.vocabulary.eos_id:
            break
        gs = automaton.step(gs, y_t)
    return masks


def _ids_from_entity_or_text(
    entity_or_text: dict[str, list[str]] | str,
    tokenizer: JsonLMTokenizer,
    add_bos_eos: bool,
) -> list[int]:
    """Normalize input (dict or serialized text) to a BOS…EOS ID sequence."""
    if isinstance(entity_or_text, dict):
        return encode_entity(entity_or_text, tokenizer=tokenizer, add_bos_eos=add_bos_eos)
    if isinstance(entity_or_text, str):
        # Parse and re-serialize to ensure canonicalized, deterministic form.
        can = parse_entity(entity_or_text)
        s = entity_to_string(can)
        return tokenizer.encode(s, add_bos_eos=add_bos_eos)
    raise TypeError(f"Expected dict or str, got {type(entity_or_text).__name__}")


def _ids_from_sequence_or_text(
    entities_or_text: list[dict[str, list[str]]] | str,
    tokenizer: JsonLMTokenizer,
    add_bos_eos: bool,
) -> list[int]:
    """Normalize sequence input (list of dicts or serialized text) to a BOS…EOS ID sequence."""
    if isinstance(entities_or_text, list):
        return encode_sequence(entities_or_text, tokenizer=tokenizer, add_bos_eos=add_bos_eos)
    if isinstance(entities_or_text, str):
        # Parse and re-serialize to ensure canonicalized, deterministic form.
        can_list = parse_sequence(entities_or_text)
        s = entities_to_string_as_set(can_list)
        return tokenizer.encode(s, add_bos_eos=add_bos_eos)
    raise TypeError(f"Expected list or str, got {type(entities_or_text).__name__}")


def logprob_entity(
    entity_or_text: dict[str, list[str]] | str,
    model: nn.Module,
    tokenizer: JsonLMTokenizer,
    normalize: Literal["sum", "mean", "bpt"] = "sum",
    device: torch.device | None = None,
) -> float:
    """Return the constrained log-likelihood of a given entity under the model.

    The method uses teacher forcing with grammar masks: at each step it masks disallowed tokens, renormalizes the logits
    over the allowed subset, and gathers the gold token probability. By default it returns the **sum** of per-token
    log-probabilities (nats). You can request normalization by token or in bits per token.

    Args:
        entity_or_text: Either a raw dict[str, list[str]] or a sentinel-serialized string for the entity.
        model: Next-token LM that maps input IDs [1, T] → logits [1, T, V].
        tokenizer: The tokenizer used for encoding.
        normalize: 'sum' (default) → total log-prob in nats; 'mean' → average per token (nats);
                   'bpt' → bits per token (negative, i.e., -NLL / ln 2).
        device: Optional torch.device to run scoring on; defaults to model's first parameter device.

    Returns:
        The requested log-probability scalar (float).

    Raises:
        ValueError: If the sequence is malformed (e.g., missing BOS/EOS) or the model returns wrong shapes.
    """
    # Prepare IDs and move tensors to the appropriate device.
    ids: list[int] = _ids_from_entity_or_text(entity_or_text, tokenizer, add_bos_eos=True)
    if len(ids) < 2:
        raise ValueError("Encoded sequence must include BOS and EOS.")
    T = len(ids) - 1  # number of predictions
    if device is None:
        try:
            device = next(model.parameters()).device  # type: ignore[assignment]
        except StopIteration:
            device = torch.device("cpu")

    input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    target_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    # Forward pass.
    logits = model(input_ids)  # [1, T, V]
    V = len(tokenizer)
    assert logits.shape == (1, T, V), f"Model must return logits [1, T, V], got {tuple(logits.shape)}"

    # Masks aligned with targets (CPU for construction, then move).
    masks = _build_masks_for_ids(ids, tokenizer).to(device=device)  # [1, T, V]

    # Renormalize and gather gold log-probs.
    log_probs = apply_mask_and_logprobs(logits, masks)  # [1, T, V]
    gold_lp = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [1, T]
    if torch.isinf(gold_lp).any():
        raise ValueError("Encountered disallowed gold token under grammar masks during scoring.")
    lp_sum = float(gold_lp.sum().item())  # total log-prob in nats
    if normalize == "sum":
        return lp_sum
    lp_mean = float(gold_lp.mean().item())
    if normalize == "mean":
        return lp_mean
    if normalize == "bpt":
        return lp_mean / math.log(2.0)  # nats → bits (note: typically negative)
    raise ValueError(f"Unknown normalization mode: {normalize!r}")


def logprob_sequence(
    entities_or_text: list[dict[str, list[str]]] | str,
    model: nn.Module,
    tokenizer: JsonLMTokenizer,
    include_eos: bool = False,
    normalize: Literal["sum", "mean", "bpt"] = "sum",
    device: torch.device | None = None,
) -> float:
    """Return the constrained log-likelihood of a given entity sequence under the model.

    The method uses teacher forcing with grammar masks: at each step it masks disallowed tokens, renormalizes the logits
    over the allowed subset, and gathers the gold token probability. By default it returns the **sum** of per-token
    log-probabilities (nats) and excludes the EOS token from scoring for sequences.

    Args:
        entities_or_text: Either a list of entity dicts or a sentinel-serialized string for the entity sequence.
        model: Next-token LM that maps input IDs [1, T] → logits [1, T, V].
        tokenizer: The tokenizer used for encoding.
        include_eos: Whether to include the EOS token in scoring. If False (default), stops scoring at the
                    first EOS position and only counts true content tokens before EOS.
        normalize: 'sum' (default) → total log-prob in nats; 'mean' → average per token (nats);
                   'bpt' → bits per token (negative, i.e., -NLL / ln 2).
        device: Optional torch.device to run scoring on; defaults to model's first parameter device.

    Returns:
        The requested log-probability scalar (float).

    Raises:
        ValueError: If the sequence is malformed (e.g., missing BOS/EOS) or the model returns wrong shapes.
    """
    # Prepare IDs and move tensors to the appropriate device.
    ids: list[int] = _ids_from_sequence_or_text(entities_or_text, tokenizer, add_bos_eos=True)
    if len(ids) < 2:
        raise ValueError("Encoded sequence must include BOS and EOS.")
    T = len(ids) - 1  # number of predictions
    if device is None:
        try:
            device = next(model.parameters()).device  # type: ignore[assignment]
        except StopIteration:
            device = torch.device("cpu")

    input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    target_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    # Forward pass.
    logits = model(input_ids)  # [1, T, V]
    V = len(tokenizer)
    assert logits.shape == (1, T, V), f"Model must return logits [1, T, V], got {tuple(logits.shape)}"

    # Masks aligned with targets (CPU for construction, then move).
    masks = _build_masks_for_ids(ids, tokenizer).to(device=device)  # [1, T, V]

    # Renormalize and gather gold log-probs.
    log_probs = apply_mask_and_logprobs(logits, masks)  # [1, T, V]
    gold_lp = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [1, T]
    if torch.isinf(gold_lp).any():
        raise ValueError("Encountered disallowed gold token under grammar masks during scoring.")

    # Handle EOS inclusion/exclusion
    if include_eos:
        # Include all positions (including EOS)
        active_positions = T
        lp_sum = float(gold_lp.sum().item())  # total log-prob in nats
    else:
        # Find first EOS position and exclude it from scoring
        target_ids_1d = target_ids.squeeze(0)  # [T]
        eos_positions = (target_ids_1d == tokenizer.vocabulary.eos_id).nonzero(as_tuple=False)
        if len(eos_positions) > 0:
            t_eos = int(eos_positions[0].item())  # first EOS position
            # Only sum over positions before EOS (t < t_eos)
            if t_eos > 0:
                lp_sum = float(gold_lp[0, :t_eos].sum().item())
                active_positions = t_eos
            else:
                # EOS is at position 0, no content tokens to score
                lp_sum = 0.0
                active_positions = 0
        else:
            # No EOS found, use all positions
            lp_sum = float(gold_lp.sum().item())
            active_positions = T

    if normalize == "sum":
        return lp_sum
    if active_positions == 0:
        # Avoid division by zero for mean/bpt when no active positions
        return 0.0
    lp_mean = lp_sum / active_positions
    if normalize == "mean":
        return lp_mean
    if normalize == "bpt":
        return lp_mean / math.log(2.0)  # nats → bits (note: typically negative)
    raise ValueError(f"Unknown normalization mode: {normalize!r}")


def decode_entity(
    model: nn.Module,
    tokenizer: JsonLMTokenizer,
    max_steps: int = 512,
    device: torch.device | None = None,
) -> dict[str, list[str]]:
    """Decode a valid entity via constrained greedy and return it as a canonical dict."""
    text = decode_greedy(
        model=model, tokenizer=tokenizer, max_steps=max_steps, device=device, stop_at_end=True
    )
    return parse_entity(text)


def delta(
    a: dict[str, list[str]],
    b: dict[str, list[str]],
    model: nn.Module,
    tokenizer: JsonLMTokenizer,
) -> float:
    """Return Δ = logP(AuB) - logP(A) - logP(B) with canonicalized set semantics.

    Keys present in both are merged and values concatenated; canonicalization then sorts keys and de-dups/sorts values.
    All three scores are computed with normalize="sum" (total log-probability in nats).

    Args:
        a: First entity (dict[str, list[str]]).
        b: Second entity.
        model: Next-token LM.
        tokenizer: Tokenizer.

    Returns:
        The Δ score (float, in nats).
    """
    merged: dict[str, list[str]] = {k: list(vs) for k, vs in a.items()}
    for k, vs in b.items():
        merged.setdefault(k, []).extend(vs)
    merged = canonicalize_entity(merged)

    return (
        logprob_entity(merged, model=model, tokenizer=tokenizer, normalize="sum")
        - logprob_entity(a, model=model, tokenizer=tokenizer, normalize="sum")
        - logprob_entity(b, model=model, tokenizer=tokenizer, normalize="sum")
    )
