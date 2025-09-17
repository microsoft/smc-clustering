"""
Prefix automaton and transitions for the constrained JSON-with-<K>/<V> grammar.

This module advances a compact State enum given observed token IDs from the tokenizer. It uses only the tokenizer's
special token IDs plus knowledge that BPE pieces occupy a contiguous range after specials. The automaton validates
transitions; callers can use `allowed_token_mask` to mask logits before sampling or loss computation.
"""

from __future__ import annotations

from dataclasses import dataclass

from jsonlm.grammar.spec import State
from jsonlm.tokenization.tokenizer import JsonLMTokenizer


@dataclass(slots=True)
class GrammarState:
    """Holds the current grammar state for constrained scoring/decoding."""

    state: State


class GrammarAutomaton:
    """Deterministic state machine for the project grammar."""

    def __init__(self, tokenizer: JsonLMTokenizer) -> None:
        """Initialize with tokenizer to access special IDs and BPE ranges."""
        self.tok = tokenizer
        # Cache IDs for speed and readability.
        self._id_lbrace = tokenizer.vocabulary.token_id("{")
        self._id_rbrace = tokenizer.vocabulary.token_id("}")
        self._id_lbracket = tokenizer.vocabulary.token_id("[")
        self._id_rbracket = tokenizer.vocabulary.token_id("]")
        self._id_colon = tokenizer.vocabulary.token_id(":")
        self._id_comma = tokenizer.vocabulary.token_id(",")
        self._id_quote = tokenizer.vocabulary.token_id('"')
        self._id_k = tokenizer.vocabulary.token_id("<K>")
        self._id_v = tokenizer.vocabulary.token_id("<V>")
        self._id_bos = tokenizer.vocabulary.bos_id
        self._id_eos = tokenizer.vocabulary.eos_id
        self._id_pad = tokenizer.vocabulary.pad_id

        # Derived ranges for BPE.
        self._specials = set(range(self.tok.specials_size))
        self._bpe_start = self.tok.specials_size
        self._bpe_end = self.tok.specials_size + self.tok.bpe_size  # exclusive

    def start(self) -> GrammarState:
        """Return the initial grammar state (after BOS, before '{')."""
        return GrammarState(state=State.START)

    def step(self, gs: GrammarState, token_id: int) -> GrammarState:
        """Advance the grammar state by consuming `token_id`.

        Args:
            gs: Current grammar state.
            token_id: The observed token ID (BOS/EOS/PAD are not expected here).

        Returns:
            The next GrammarState.

        Raises:
            ValueError: If the token is not valid in the current state.
        """
        s = gs.state
        t = token_id

        # Disallow BOS/PAD anywhere; EOS handled by the caller based on END state.
        if t in (self._id_bos, self._id_pad):
            raise ValueError("BOS/PAD are not valid tokens in constrained sequence positions.")

        # Helper predicates.
        is_bpe = self._bpe_start <= t < self._bpe_end

        if s == State.START:
            if t == self._id_lbrace:
                return GrammarState(State.AFTER_LBRACE)
            raise ValueError("Expected '{' at start.")

        if s == State.AFTER_LBRACE:
            if t == self._id_k:
                return GrammarState(State.AFTER_K)
            if t == self._id_rbrace:
                return GrammarState(State.END)
            raise ValueError("Expected '<K>' or '}' after '{'.")

        if s == State.AFTER_K:
            if t == self._id_quote:
                return GrammarState(State.IN_KEY_STRING)
            raise ValueError('Expected opening quote after "<K>".')

        if s == State.IN_KEY_STRING:
            if t == self._id_quote:
                return GrammarState(State.AFTER_KEY_STRING)
            if is_bpe:
                return gs  # remain in string
            raise ValueError("Expected BPE piece or closing quote inside key string.")

        if s == State.AFTER_KEY_STRING:
            if t == self._id_colon:
                return GrammarState(State.AFTER_COLON)
            raise ValueError('Expected ":" after key string.')

        if s == State.AFTER_COLON:
            if t == self._id_lbracket:
                return GrammarState(State.AFTER_LBRACKET)
            raise ValueError('Expected "[" after ":".')

        if s == State.AFTER_LBRACKET:
            if t == self._id_v:
                return GrammarState(State.AFTER_V)
            if t == self._id_rbracket:
                return GrammarState(State.AFTER_RBRACKET)
            raise ValueError('Expected "<V>" or "]" after "[".')

        if s == State.AFTER_V:
            if t == self._id_quote:
                return GrammarState(State.IN_VALUE_STRING)
            raise ValueError('Expected opening quote after "<V>".')

        if s == State.IN_VALUE_STRING:
            if t == self._id_quote:
                return GrammarState(State.AFTER_VALUE_STRING)
            if is_bpe:
                return gs  # remain in string
            raise ValueError("Expected BPE piece or closing quote inside value string.")

        if s == State.AFTER_VALUE_STRING:
            if t == self._id_comma:
                return GrammarState(State.AFTER_LBRACKET)  # Expect another <V> or ']'
            if t == self._id_rbracket:
                return GrammarState(State.AFTER_RBRACKET)
            raise ValueError('Expected "," or "]" after value string.')

        if s == State.AFTER_RBRACKET:
            if t == self._id_comma:
                return GrammarState(State.AFTER_LBRACE)  # Next key or end
            if t == self._id_rbrace:
                return GrammarState(State.END)
            raise ValueError('Expected "," or "}" after value array.')

        if s == State.END:
            # Accept '{' to start another entity, or EOS (handled by caller).
            if t == self._id_lbrace:
                return GrammarState(State.AFTER_LBRACE)
            # Any other token is invalid (EOS is handled by caller).
            raise ValueError("Only '{' or EOS are valid after top-level '}' (to start another entity or terminate).")

        # Should be unreachable.
        raise ValueError(f"Unhandled state {s} with token id {t}.")
