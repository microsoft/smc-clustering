"""Allowed-token masking for the grammar to enforce 100% valid sequences.

This module exposes `allowed_token_mask`, which returns a Bool mask over the joint vocabulary for the next token given
the current grammar state. It disallows BOS/PAD in all positions and only allows EOS after the END state. Inside string
states it allows any BPE piece or the closing quote; outside strings it allows only structural/sentinel tokens per state.
"""

from __future__ import annotations

import torch

from smc_clustering.jsonlm.grammar.automaton import GrammarAutomaton, GrammarState
from smc_clustering.jsonlm.grammar.spec import State
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer


def allowed_token_mask(
    gs: GrammarState, automaton: GrammarAutomaton, tokenizer: JsonLMTokenizer
) -> torch.BoolTensor:
    """Return a Bool mask over the joint vocabulary for valid next tokens.

    Uses precomputed tables if automaton has runtime, otherwise falls back to explicit logic.

    Args:
        gs: Current grammar state.
        automaton: The automaton carrying cached token IDs and optional runtime.
        tokenizer: The tokenizer describing vocab layout (specials + BPE).

    Returns:
        torch.BoolTensor of shape [|V_joint|] with True for allowed tokens.

    Notes:
        BOS and PAD are always disallowed; EOS is allowed if and only if state == END.
    """
    V = len(tokenizer)
    s = gs.state

    # Use precomputed tables if available
    if automaton._runtime is not None:
        try:
            state_idx = automaton._get_state_index(s)
            # Return a copy to avoid modifying the cached tensor
            mask = automaton._runtime.allowed[state_idx, :].clone()
            return mask
        except (IndexError, RuntimeError):
            # Fall back to explicit logic if table lookup fails
            pass

    # Explicit mask construction (fallback)
    mask = torch.zeros(V, dtype=torch.bool)

    # Convenience IDs and ranges.
    id_lbrace = automaton._id_lbrace
    id_rbrace = automaton._id_rbrace
    id_lbracket = automaton._id_lbracket
    id_rbracket = automaton._id_rbracket
    id_colon = automaton._id_colon
    id_comma = automaton._id_comma
    id_quote = automaton._id_quote
    id_k = automaton._id_k
    id_v = automaton._id_v
    id_eos = automaton._id_eos

    bpe_start = tokenizer.specials_size
    bpe_end = tokenizer.specials_size + tokenizer.bpe_size  # exclusive

    # State-specific allowances.
    if s == State.START:
        mask[id_lbrace] = True

    elif s == State.AFTER_LBRACE:
        mask[id_k] = True
        mask[id_rbrace] = True

    elif s == State.AFTER_K:
        mask[id_quote] = True

    elif s == State.IN_KEY_STRING:
        mask[id_quote] = True  # allow closing quote
        if bpe_end > bpe_start:
            mask[bpe_start:bpe_end] = True  # allow any BPE piece

    elif s == State.AFTER_KEY_STRING:
        mask[id_colon] = True

    elif s == State.AFTER_COLON:
        mask[id_lbracket] = True

    elif s == State.AFTER_LBRACKET:
        mask[id_v] = True
        mask[id_rbracket] = True  # empty array

    elif s == State.AFTER_V:
        mask[id_quote] = True

    elif s == State.IN_VALUE_STRING:
        mask[id_quote] = True
        if bpe_end > bpe_start:
            mask[bpe_start:bpe_end] = True

    elif s == State.AFTER_VALUE_STRING:
        mask[id_comma] = True
        mask[id_rbracket] = True

    elif s == State.AFTER_RBRACKET:
        mask[id_comma] = True
        mask[id_rbrace] = True

    elif s == State.END:
        # Allow '{' to start another entity; EOS handled below
        mask[id_lbrace] = True

    else:
        # Defensive default: no tokens allowed.
        pass

    # BOS and PAD are globally disallowed; EOS only in END.
    # (BOS/PAD remain False; set EOS if in END.)
    if s == State.END:
        mask[id_eos] = True

    return mask
