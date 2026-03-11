# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Constrained greedy decoding for valid JSON entities with <K>/<V> sentinels.

This decoder uses the grammar automaton to mask disallowed tokens at each step, then greedily selects the
argmax among the allowed set. It returns a sentinel-serialized string (no BOS/EOS) that always parses into a valid
entity when the model is at least non-degenerate. With a uniform model, it emits the smallest valid object: `{ }`.
"""

from __future__ import annotations

import torch
from torch import nn

from smc_clustering.jsonlm.grammar.automaton import GrammarAutomaton
from smc_clustering.jsonlm.grammar.mask import allowed_token_mask
from smc_clustering.jsonlm.grammar.spec import State
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer


def decode_greedy(
    model: nn.Module,
    tokenizer: JsonLMTokenizer,
    max_steps: int = 512,
    device: torch.device | None = None,
    stop_at_end: bool = False,
) -> str:
    """Generate a valid entity string using grammar-constrained greedy decoding.

    The procedure starts from BOS, then at each step:
      1) runs the model on the current prefix to get logits for the next token,
      2) builds an allowed-token mask from the grammar state,
      3) sets disallowed logits to -inf and takes argmax,
      4) appends the chosen token, updates grammar, and repeats until EOS.

    Args:
        model: Next-token LM mapping input IDs [1, T] → logits [1, T, V].
        tokenizer: Tokenizer providing vocabulary layout and decode().
        max_steps: Maximum number of tokens to generate (including EOS); guards against infinite loops.
        device: Optional torch.device; defaults to the model's parameter device if not provided.
        stop_at_end: If True, force EOS when gs.state==END by masking to EOS-only (single entity termination).

    Returns:
        A space-separated serialized entity string (without BOS/EOS).

    Raises:
        RuntimeError: If decoding hits max_steps without producing EOS, or if no token is allowed at a step.
        AssertionError: If the model fails to return logits of the correct shape.
    """
    if device is None:
        try:
            device = next(model.parameters()).device  # type: ignore[assignment]
        except StopIteration:
            device = torch.device("cpu")

    bos_id = tokenizer.vocabulary.bos_id
    eos_id = tokenizer.vocabulary.eos_id

    # Running generated sequence of IDs (will include BOS and finally EOS).
    seq: list[int] = [bos_id]

    automaton = GrammarAutomaton(tokenizer)
    gs = automaton.start()

    for _ in range(max_steps):
        # Model forward on current prefix (without EOS).
        inp = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        logits = model(inp)  # [1, T, V]
        V = len(tokenizer)
        assert logits.dim() == 3 and logits.shape[0] == 1 and logits.shape[2] == V, (
            f"Bad logits shape {tuple(logits.shape)}"
        )
        # Logits for the next token are at the last timestep.
        next_logits = logits[0, -1, :]  # [V]

        # Grammar mask for current state (allowed next tokens).
        mask = allowed_token_mask(gs, automaton, tokenizer).to(device=device)  # [V]

        # If stop_at_end=True and we're in END state, force EOS-only to guarantee single entity.
        if stop_at_end and gs.state == State.END:
            mask.fill_(False)
            mask[eos_id] = True

        if not mask.any():
            raise RuntimeError("No allowed next tokens from grammar; decoding cannot proceed.")

        # Greedy choice among allowed tokens.
        masked_logits = next_logits.masked_fill(~mask, float("-inf"))
        next_id = int(torch.argmax(masked_logits).item())

        # Append and update grammar. If EOS, we're done.
        seq.append(next_id)
        if next_id == eos_id:
            break
        gs = automaton.step(gs, next_id)
    else:
        # Loop exhausted without EOS.
        raise RuntimeError(f"Decoding exceeded max_steps={max_steps} without emitting EOS.")

    # Convert BOS…EOS ID sequence back to serialized text (strip boundaries).
    text = tokenizer.decode(seq, strip_bos_eos=True)
    return text


def decode_sample(
    model: nn.Module,
    tokenizer: JsonLMTokenizer,
    max_steps: int = 512,
    device: torch.device | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> str:
    """Grammar-constrained *stochastic* decoding (top-k / nucleus) that always yields valid JSON.

    At each step we:
      1) compute logits for the next token,
      2) mask disallowed ids via the grammar,
      3) apply temperature + top-k and/or top-p filtering on the allowed subset,
      4) sample the next id.

    Args:
        model: Autoregressive language model returning logits of shape ``[B, T, V]``.
        tokenizer: Tokenizer whose vocabulary and grammar define valid continuations.
        max_steps: Maximum number of decoding iterations before failing.
        device: Device on which to run decoding. Defaults to the model device when omitted.
        temperature: >0; 1.0 = no scale, <1 = sharper, >1 = flatter.
        top_k: keep only the largest-k allowed logits (after masking). None disables.
        top_p: nucleus threshold in (0, 1]; keep smallest set whose cumulative prob ≥ top_p. None disables.
        seed: if set, makes sampling deterministic for reproducibility.

    Returns:
        A space-separated serialized entity string (without BOS/EOS).
    """
    if device is None:
        try:
            device = next(model.parameters()).device  # type: ignore[assignment]
        except StopIteration:
            device = torch.device("cpu")

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    bos = tokenizer.vocabulary.bos_id
    eos = tokenizer.vocabulary.eos_id
    seq: list[int] = [bos]

    automaton = GrammarAutomaton(tokenizer)
    gs = automaton.start()

    for _ in range(max_steps):
        inp = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        logits = model(inp)  # [1, T, V]
        V = len(tokenizer)
        assert logits.dim() == 3 and logits.shape[0] == 1 and logits.shape[2] == V, (
            f"Bad logits shape {tuple(logits.shape)}"
        )
        next_logits = logits[0, -1, :].clone()  # [V]

        # Mask disallowed → -inf
        mask = allowed_token_mask(gs, automaton, tokenizer).to(device=device)
        if not mask.any():
            raise RuntimeError("No allowed next tokens; sampling cannot proceed.")
        next_logits[~mask] = float("-inf")

        # Temperature
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        next_logits = next_logits / temperature

        # Top-k filter (on allowed subset)
        if top_k is not None and top_k > 0 and top_k < mask.sum().item():
            # find kth threshold among allowed only
            allowed_logits = next_logits[mask]
            kth = torch.topk(allowed_logits, k=top_k).values.min()
            # drop anything below kth
            next_logits[next_logits < kth] = float("-inf")

        # Top-p (nucleus) filter (on allowed subset)
        if top_p is not None:
            if not (0.0 < top_p <= 1.0):
                raise ValueError("top_p must be in (0, 1]")
            probs = torch.softmax(next_logits, dim=-1)
            # sort by prob desc
            probs_sorted, idx_sorted = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(probs_sorted, dim=-1)
            keep = cumsum <= top_p
            # ensure at least one token kept
            if not bool(keep.any()):
                keep[0] = True
            # mask out dropped ids
            drop_idx = idx_sorted[~keep]
            next_logits[drop_idx] = float("-inf")

        # sample from re-normalized distribution
        probs = torch.softmax(next_logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1, generator=gen).item())

        seq.append(next_id)
        if next_id == eos:
            break
        gs = automaton.step(gs, next_id)
    else:
        raise RuntimeError(f"Decoding exceeded max_steps={max_steps} without emitting EOS.")

    return tokenizer.decode(seq, strip_bos_eos=True)
