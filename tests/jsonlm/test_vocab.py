"""Unit tests for token constants and the Vocabulary dataclass.

We assert uniqueness, ordering stability, and correctness of id↔token mappings. Using a fixed snapshot of
SPECIAL_TOKENS helps catch accidental reorderings that would break trainings/checkpoints and grammar masks.
"""

from __future__ import annotations

import pytest

from smc_clustering.jsonlm import constants
from smc_clustering.jsonlm.tokenization import Vocabulary


def test_special_tokens_uniqueness() -> None:
    """All special token strings must be unique."""
    toks = list(constants.SPECIAL_TOKENS)
    assert len(toks) == len(set(toks)), "SPECIAL_TOKENS contains duplicate strings."


def test_special_tokens_order_snapshot() -> None:
    """SPECIAL_TOKENS should remain in the agreed canonical order (snapshot test)."""
    expected = [
        "{",
        "}",
        "[",
        "]",
        ":",
        ",",
        '"',
        "<K>",
        "<V>",
        "<BOS>",
        "<EOS>",
        "<PAD>",
    ]
    assert expected == constants.SPECIAL_TOKENS


def test_vocabulary_build_and_lookup() -> None:
    """Building Vocabulary creates consistent id↔token mappings and convenience IDs."""
    vocab = Vocabulary.from_default()

    # Ensure the length matches the constants list.
    assert len(vocab) == len(constants.SPECIAL_TOKENS)

    # Check round-trip id↔token for all positions.
    for i, tok in enumerate(constants.SPECIAL_TOKENS):
        assert vocab.id_token(i) == tok
        assert vocab.token_id(tok) == i

    # Check convenience properties align with mapping.
    assert vocab.bos_token == "<BOS>"
    assert vocab.eos_token == "<EOS>"
    assert vocab.pad_token == "<PAD>"
    assert vocab.k_token == "<K>"
    assert vocab.v_token == "<V>"
    assert vocab.quote_token == '"'

    assert vocab.bos_id == vocab.token_id("<BOS>")
    assert vocab.eos_id == vocab.token_id("<EOS>")
    assert vocab.pad_id == vocab.token_id("<PAD>")
    assert vocab.k_id == vocab.token_id("<K>")
    assert vocab.v_id == vocab.token_id("<V>")
    assert vocab.quote_id == vocab.token_id('"')


def test_vocabulary_missing_required_raises() -> None:
    """Missing required mandatory specials should raise a helpful error."""
    # Remove EOS to simulate a bad configuration.
    bad_tokens = [t for t in constants.SPECIAL_TOKENS if t != constants.EOS]
    with pytest.raises(ValueError) as excinfo:
        _ = Vocabulary.from_tokens(bad_tokens)
    assert "missing required tokens" in str(excinfo.value).lower()


def test_vocabulary_duplicate_raises() -> None:
    """Duplicate token strings should raise with the duplicate indicated."""
    dup_tokens = list(constants.SPECIAL_TOKENS)
    dup_tokens.insert(0, constants.BOS)  # duplicate BOS
    with pytest.raises(ValueError) as excinfo:
        _ = Vocabulary.from_tokens(dup_tokens)
    assert "duplicate token" in str(excinfo.value).lower()
    assert "<BOS>" in str(excinfo.value)
