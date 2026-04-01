# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Unit tests for the hybrid tokenizer: specials-first + byte-level BPE inside quotes.

We verify round-trip encode/decode on strings with spaces and unseen content, specials occupying first IDs,
and basic range/structure checks during decode.
"""

from __future__ import annotations

import pytest

from smc_clustering.jsonlm.serialization.encoder import entity_to_string
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _corpus() -> list[str]:
    """Build a small corpus from a few serialized entities, including spaces in strings."""
    entities = [
        {"author": ["Ada Lovelace", "Augusta Ada"], "tags": ["ai", "ml"]},
        {"author": ["Grace Hopper"], "tags": ["compilers", "systems"]},
        {"title": ["Notes on BPE"], "tags": ["ml", "ai"]},
    ]
    return [entity_to_string(e) for e in entities]


def test_train_and_roundtrip_with_spaces_and_unseen() -> None:
    """Tokenizer should round-trip and handle unseen strings (no OOVs due to byte-level)."""
    vocab = Vocabulary.from_default()
    corpus = _corpus()
    tok = train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=128)

    # Round-trip corpus lines.
    for s in corpus:
        ids = tok.encode(s, add_bos_eos=True)
        s2 = tok.decode(ids, strip_bos_eos=True)
        assert s2 == s

    # Unseen content (not present in training) should still encode/decode.
    unseen = entity_to_string({"k": ["𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ✓", "new token never seen"]})
    ids = tok.encode(unseen, add_bos_eos=False)
    back = tok.decode(ids, strip_bos_eos=False)
    assert back == unseen


def test_specials_first_ids() -> None:
    """Special tokens occupy the first IDs in the joint vocabulary."""
    vocab = Vocabulary.from_default()
    tok = train_tokenizer(_corpus(), vocabulary=vocab, bpe_vocab_size=64)
    assert tok.specials_size == len(vocab)
    # Sanity: joint length equals specials + bpe.
    assert len(tok) == tok.specials_size + tok.bpe_size


def test_decode_structure_errors() -> None:
    """Decoding should error if BPE appears outside quotes or quotes are unbalanced."""
    vocab = Vocabulary.from_default()
    tok = train_tokenizer(_corpus(), vocabulary=vocab, bpe_vocab_size=64)

    # Craft an invalid sequence: a BPE id outside quotes.
    bad = [vocab.bos_id, tok._bpe_to_joint_id(0), vocab.eos_id]
    with pytest.raises(ValueError):
        _ = tok.decode(bad, strip_bos_eos=False)

    # Missing closing quote.
    # Start a quote, add a piece, but no closing quote.
    bad2 = [vocab.bos_id, vocab.quote_id, tok._bpe_to_joint_id(0), vocab.eos_id]
    with pytest.raises(ValueError):
        _ = tok.decode(bad2, strip_bos_eos=False)
