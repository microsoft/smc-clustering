# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Vectorized grammar mask construction runtime for training and scoring.

This module provides a cached, device-local runtime that precomputes allowed token masks and state transitions
for all grammar states.

Key components:
  - GrammarRuntime: Device-local tables with vectorized mask building
  - get_runtime(): Cached factory keyed by tokenizer signature and device
  - _build_tables(): Precomputed allowed[S,V] and next_state[S,V] tables

The state count S corresponds to len(State) grammar states, and vocabulary layout V includes
specials + BPE tokens. Caching ensures tables are built once per (tokenizer, device) combination.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import torch

from smc_clustering.jsonlm.grammar.spec import State


if TYPE_CHECKING:
    from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer


class GrammarRuntime:
    """Vectorized grammar runtime with precomputed allowed-token masks and transitions.

    This class holds device-local lookup tables that enable batch mask construction.

    Attributes:
        allowed: BoolTensor [S, V] indicating which tokens are allowed from each state
        next_state: LongTensor [S, V] giving next state after consuming each token (-1 for invalid)
        idx_START: Index of the START state in the state enumeration
        idx_END: Index of the END state in the state enumeration
        eos_id: Token ID for end-of-sequence
    """

    def __init__(
        self,
        allowed: torch.Tensor,
        next_state: torch.Tensor,
        idx_start: int,
        idx_end: int,
        eos_id: int,
    ) -> None:
        """Initialize a GrammarRuntime with precomputed tables."""
        self.allowed = allowed  # [S, V], bool
        self.next_state = next_state  # [S, V], long
        self.idx_START = idx_start
        self.idx_END = idx_end
        self.eos_id = eos_id

        self._state_list = list(State)
        assert 0 <= self.idx_START < len(self._state_list)

    @property
    def state_list(self) -> list[State]:
        """List of grammar states in enumeration order."""
        return self._state_list

    @torch.no_grad()
    def build_masks(self, ids_with_eos: torch.Tensor) -> torch.BoolTensor:
        """Construct [B, T, V] Boolean masks of allowed next tokens.

        Given sequences with BOS…EOS, builds masks per timestep aligned with targets (for predicting
        token y_t given prefix y_<t). After the first EOS in each row, masks allow EOS-only.

        Args:
            ids_with_eos: LongTensor [B, L] containing BOS…EOS sequences

        Returns:
            BoolTensor [B, T, V] where T = L-1, indicating allowed tokens at each position
        """
        assert ids_with_eos.dim() == 2 and ids_with_eos.dtype == torch.long
        B, L = ids_with_eos.shape
        assert L >= 2, "Need at least BOS and EOS"
        T = L - 1
        device = ids_with_eos.device
        V = self.allowed.shape[1]

        # Targets y_t (the "next" tokens)
        Y = ids_with_eos[:, 1:]  # [B, T]

        # First EOS position per row (T if none)
        eos_bt = self.eos_id == Y  # [B, T], bool
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        first_eos = idxs.masked_fill(~eos_bt, T).amin(dim=1)  # pyright: ignore[reportCallIssue]

        # Prepare output: initialize everything to EOS-only for padding positions
        eos_only = torch.zeros(V, dtype=torch.bool, device=device)  # [V]
        eos_only[self.eos_id] = True  # Only EOS allowed in padding
        masks = eos_only.view(1, 1, V).expand(B, T, V).clone()  # [B,T,V]

        # Current automaton state per row
        state = torch.full((B,), self.idx_START, dtype=torch.long, device=device)

        # Time-major loop, batch vectorized
        rows = torch.arange(B, device=device)
        for t in range(T):
            # rows still producing (t <= first_eos) -> we must write allowed mask for this step
            active = rows[first_eos >= t]  # [N] indices of active rows
            if active.numel() == 0:
                break

            s_active = state.index_select(0, active)  # [N]
            masks[active, t, :] = self.allowed.index_select(0, s_active)  # write [N,V]

            # advance automaton where we have not yet hit EOS at y_t (t < first_eos)
            will_step = rows[first_eos > t]
            if will_step.numel() > 0:
                s = state.index_select(0, will_step)  # [M]
                y = Y.index_select(0, will_step)[:, t]  # [M]

                ns = self.next_state[s, y]
                if (ns < 0).any():
                    raise ValueError("Disallowed gold token under grammar at some positions.")
                state.scatter_(0, will_step, ns)  # update

        return masks


def _enumerate_states() -> tuple[dict[State, int], list[State]]:
    """Return (state2index, index2state) for all grammar states."""
    states = list(State)
    s2i = {s: i for i, s in enumerate(states)}
    i2s = states
    return s2i, i2s


def _build_tables(
    tokenizer: JsonLMTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Build allowed-token masks and next-state transitions for all grammar states.

    This function mirrors the logic from scoring._build_allowed_masks_and_transitions
    and creates precomputed lookup tables for vectorized mask construction.

    Args:
        tokenizer: JsonLMTokenizer with vocabulary and special tokens
        device: Target device for tensors

    Returns:
        allowed_masks: BoolTensor [S, V] indicating allowed tokens per state
        next_state: LongTensor [S, V] giving next state per token (-1 for invalid)
        idx_START: Index of START state
        idx_END: Index of END state
        eos_id: Token ID for end-of-sequence
    """
    s2i, i2s = _enumerate_states()
    S = len(i2s)
    V = len(tokenizer)

    # Cache token IDs
    voc = tokenizer.vocabulary
    id_lbrace = voc.token_id("{")
    id_rbrace = voc.token_id("}")
    id_lbracket = voc.token_id("[")
    id_rbracket = voc.token_id("]")
    id_colon = voc.token_id(":")
    id_comma = voc.token_id(",")
    id_quote = voc.token_id('"')
    id_k = voc.token_id("<K>")
    id_v = voc.token_id("<V>")
    id_eos = voc.eos_id

    bpe_start = tokenizer.specials_size
    bpe_end = tokenizer.specials_size + tokenizer.bpe_size  # exclusive

    allowed = torch.zeros((S, V), dtype=torch.bool, device=device)
    nxt = torch.full((S, V), -1, dtype=torch.long, device=device)

    def allow(s: int, toks: torch.Tensor | int | slice) -> None:
        """Mark tokens as allowed from state s."""
        allowed[s, toks] = True

    def trans(s: int, tok: int, s2: int) -> None:
        """Define a state transition from state s on token tok to state s2."""
        nxt[s, tok] = s2

    # Fill per-state rules (mirrors allowed_token_mask/step logic)
    START = s2i[State.START]
    AFTER_LBRACE = s2i[State.AFTER_LBRACE]
    AFTER_K = s2i[State.AFTER_K]
    IN_KEY_STRING = s2i[State.IN_KEY_STRING]
    AFTER_KEY_STRING = s2i[State.AFTER_KEY_STRING]
    AFTER_COLON = s2i[State.AFTER_COLON]
    AFTER_LBRACKET = s2i[State.AFTER_LBRACKET]
    AFTER_V = s2i[State.AFTER_V]
    IN_VALUE_STRING = s2i[State.IN_VALUE_STRING]
    AFTER_VALUE_STR = s2i[State.AFTER_VALUE_STRING]
    AFTER_RBRACKET = s2i[State.AFTER_RBRACKET]
    END = s2i[State.END]

    # START
    allow(START, id_lbrace)
    trans(START, id_lbrace, AFTER_LBRACE)

    # AFTER_LBRACE
    allow(AFTER_LBRACE, torch.tensor([id_k, id_rbrace], device=device))
    trans(AFTER_LBRACE, id_k, AFTER_K)
    trans(AFTER_LBRACE, id_rbrace, END)

    # AFTER_K
    allow(AFTER_K, id_quote)
    trans(AFTER_K, id_quote, IN_KEY_STRING)

    # IN_KEY_STRING
    allow(IN_KEY_STRING, id_quote)
    trans(IN_KEY_STRING, id_quote, AFTER_KEY_STRING)
    if bpe_end > bpe_start:
        allow(IN_KEY_STRING, slice(bpe_start, bpe_end))
        nxt[IN_KEY_STRING, bpe_start:bpe_end] = IN_KEY_STRING  # stay in string

    # AFTER_KEY_STRING
    allow(AFTER_KEY_STRING, id_colon)
    trans(AFTER_KEY_STRING, id_colon, AFTER_COLON)

    # AFTER_COLON
    allow(AFTER_COLON, id_lbracket)
    trans(AFTER_COLON, id_lbracket, AFTER_LBRACKET)

    # AFTER_LBRACKET
    allow(AFTER_LBRACKET, torch.tensor([id_v, id_rbracket], device=device))
    trans(AFTER_LBRACKET, id_v, AFTER_V)
    trans(AFTER_LBRACKET, id_rbracket, AFTER_RBRACKET)

    # AFTER_V
    allow(AFTER_V, id_quote)
    trans(AFTER_V, id_quote, IN_VALUE_STRING)

    # IN_VALUE_STRING
    allow(IN_VALUE_STRING, id_quote)
    trans(IN_VALUE_STRING, id_quote, AFTER_VALUE_STR)
    if bpe_end > bpe_start:
        allow(IN_VALUE_STRING, slice(bpe_start, bpe_end))
        nxt[IN_VALUE_STRING, bpe_start:bpe_end] = IN_VALUE_STRING  # transition

    # AFTER_VALUE_STRING
    allow(AFTER_VALUE_STR, torch.tensor([id_comma, id_rbracket], device=device))
    trans(AFTER_VALUE_STR, id_comma, AFTER_LBRACKET)
    trans(AFTER_VALUE_STR, id_rbracket, AFTER_RBRACKET)

    # AFTER_RBRACKET
    allow(AFTER_RBRACKET, torch.tensor([id_comma, id_rbrace], device=device))
    trans(AFTER_RBRACKET, id_comma, AFTER_LBRACE)
    trans(AFTER_RBRACKET, id_rbrace, END)

    # END -> EOS and '{' allowed (Kleene-plus: one or more entities)
    allow(END, torch.tensor([id_eos, id_lbrace], device=device))
    trans(END, id_lbrace, AFTER_LBRACE)
    # (EOS doesn't transition - handled by caller)

    return allowed, nxt, START, END, id_eos


def _tok_signature(tokenizer: JsonLMTokenizer) -> str:
    """Generate a stable signature for tokenizer vocabulary and configuration."""
    tokens = tokenizer.vocabulary.as_list()
    meta = {
        "tokens": tokens,
        "specials_size": tokenizer.specials_size,
        "bpe_size": tokenizer.bpe_size,
    }
    payload = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


# Global cache for runtime instances
_RUNTIMES: dict[tuple[str, torch.device, torch.dtype], GrammarRuntime] = {}


def get_runtime(tokenizer: JsonLMTokenizer, device: torch.device) -> GrammarRuntime:
    """Get or create a cached GrammarRuntime for the given tokenizer and device.

    Returns a cached runtime instance keyed by tokenizer signature, device, and default dtype.
    Multiple calls with the same parameters return the same object.

    Args:
        tokenizer: JsonLMTokenizer defining vocabulary and special tokens
        device: Target device for tensor operations

    Returns:
        GrammarRuntime instance with precomputed tables on the specified device
    """
    key = (_tok_signature(tokenizer), device, torch.get_default_dtype())
    rt = _RUNTIMES.get(key)
    if rt is None:
        allowed, nxt, idx_s, idx_e, eos = _build_tables(tokenizer, device)
        rt = GrammarRuntime(allowed, nxt, idx_s, idx_e, eos)
        _RUNTIMES[key] = rt
    return rt
