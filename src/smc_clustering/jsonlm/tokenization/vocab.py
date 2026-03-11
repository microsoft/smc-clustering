# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Provides Vocabulary, a deterministic mapping between token strings and IDs (specials-first).

We use special tokens to represent JSON structure symbols, and to support grammar-constrained decoding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from smc_clustering.jsonlm import constants


@dataclass(slots=True)
class Vocabulary:
    """Holds an ordered token list and fast id↔token lookup tables.

    Attributes:
        tokens: Ordered list of token strings; the token ID is its index in this list.
        token_to_id: Mapping from token string to its integer ID.
        id_to_token: List enabling reverse lookup from ID to token string.
    """

    tokens: list[str]
    token_to_id: dict[str, int] = field(init=False, repr=False)
    id_to_token: list[str] = field(init=False, repr=False)

    # Convenience cached IDs (filled in __post_init__).
    _bos_id: int = field(init=False, repr=False)
    _eos_id: int = field(init=False, repr=False)
    _pad_id: int = field(init=False, repr=False)
    _k_id: int = field(init=False, repr=False)
    _v_id: int = field(init=False, repr=False)
    _quote_id: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate uniqueness, build lookup tables, and cache convenience IDs."""
        # Ensure deterministic ordering & uniqueness. Using a list preserves order for stable IDs.
        assert isinstance(self.tokens, list), "Vocabulary.tokens must be a list to preserve order."
        if len(set(self.tokens)) != len(self.tokens):
            # Identify the first duplicate to aid debugging; stable behavior across runs.
            seen: set[str] = set()
            dup: str | None = None
            for t in self.tokens:
                if t in seen:
                    dup = t
                    break
                seen.add(t)
            raise ValueError(f"Duplicate token detected in Vocabulary: {dup!r}")

        # Build forward and reverse lookups (shape: [|V|]).
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = list(self.tokens)

        # Validate presence of mandatory specials for the project.
        required = (
            constants.BOS,
            constants.EOS,
            constants.PAD,
            constants.K_SENTINEL,
            constants.V_SENTINEL,
            constants.QUOTE,
        )
        missing = [tok for tok in required if tok not in self.token_to_id]
        if missing:
            raise ValueError(f"Vocabulary missing required tokens: {missing}")

        # Cache hot-path IDs for convenience properties.
        self._bos_id = self.token_to_id[constants.BOS]
        self._eos_id = self.token_to_id[constants.EOS]
        self._pad_id = self.token_to_id[constants.PAD]
        self._k_id = self.token_to_id[constants.K_SENTINEL]
        self._v_id = self.token_to_id[constants.V_SENTINEL]
        self._quote_id = self.token_to_id[constants.QUOTE]

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.id_to_token)

    def as_list(self) -> list[str]:
        """Return a shallow copy of the ordered token list."""
        return list(self.id_to_token)

    def contains(self, token: str) -> bool:
        """Return True if the token string exists in the vocabulary."""
        return token in self.token_to_id

    def token_id(self, token: str) -> int:
        """Return the integer ID for a token string."""
        try:
            return self.token_to_id[token]
        except KeyError as exc:
            raise KeyError(f"Unknown token: {token!r}") from exc

    def id_token(self, idx: int) -> str:
        """Return the token string for an integer ID."""
        if idx < 0 or idx >= len(self.id_to_token):
            raise IndexError(f"Token ID out of range: {idx}")
        return self.id_to_token[idx]

    @property
    def bos_token(self) -> str:
        """Return the BOS token string."""
        return constants.BOS

    @property
    def eos_token(self) -> str:
        """Return the EOS token string."""
        return constants.EOS

    @property
    def pad_token(self) -> str:
        """Return the PAD token string."""
        return constants.PAD

    @property
    def k_token(self) -> str:
        """Return the key sentinel token string."""
        return constants.K_SENTINEL

    @property
    def v_token(self) -> str:
        """Return the value sentinel token string."""
        return constants.V_SENTINEL

    @property
    def quote_token(self) -> str:
        """Return the double-quote token string."""
        return constants.QUOTE

    @property
    def bos_id(self) -> int:
        """Return the BOS token ID."""
        return self._bos_id

    @property
    def eos_id(self) -> int:
        """Return the EOS token ID."""
        return self._eos_id

    @property
    def pad_id(self) -> int:
        """Return the PAD token ID."""
        return self._pad_id

    @property
    def k_id(self) -> int:
        """Return the key sentinel token ID."""
        return self._k_id

    @property
    def v_id(self) -> int:
        """Return the value sentinel token ID."""
        return self._v_id

    @property
    def quote_id(self) -> int:
        """Return the double-quote token ID."""
        return self._quote_id

    @classmethod
    def from_default(cls) -> Vocabulary:
        """Construct a Vocabulary from jsonlm.constants.SPECIAL_TOKENS order."""
        return cls(tokens=list(constants.SPECIAL_TOKENS))

    @classmethod
    def from_tokens(cls, tokens: Sequence[str]) -> Vocabulary:
        """Construct a Vocabulary from a custom ordered list of tokens."""
        return cls(tokens=list(tokens))
