# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Unit tests for the grammar automaton and allowed-token masks.

We verify local expectations at critical states and an end-to-end walk over a sample entity: each consumed token is
allowed by the mask, and EOS becomes allowed exactly after the closing '}' token.
"""

from __future__ import annotations

from smc_clustering.jsonlm.grammar.automaton import GrammarAutomaton
from smc_clustering.jsonlm.grammar.mask import allowed_token_mask
from smc_clustering.jsonlm.grammar.spec import State
from smc_clustering.jsonlm.serialization.encoder import entity_to_string
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _tok() -> tuple[GrammarAutomaton, Vocabulary]:
    vocab = Vocabulary.from_default()
    # Minimal corpus to ensure BPE has some pieces.
    s = entity_to_string({"a": ["x", "y"]})

    tok = train_tokenizer([s], vocabulary=vocab, bpe_vocab_size=64)
    automaton = GrammarAutomaton(tok)
    return automaton, vocab


def test_masks_basic_states() -> None:
    """Spot-check allowed tokens at a few key states."""
    automaton, _vocab = _tok()
    tok = automaton.tok

    # START => only '{' is allowed
    gs = automaton.start()
    m = allowed_token_mask(gs, automaton, tok)
    assert m[tok.vocabulary.token_id("{")].item() is True
    assert m[tok.vocabulary.token_id("}")].item() is False
    assert m[tok.vocabulary.token_id("<K>")].item() is False

    # AFTER_LBRACE => '<K>' or '}'
    gs = automaton.step(gs, tok.vocabulary.token_id("{"))
    m = allowed_token_mask(gs, automaton, tok)
    assert m[tok.vocabulary.token_id("<K>")].item() is True
    assert m[tok.vocabulary.token_id("}")].item() is True
    assert m[tok.vocabulary.token_id("[")].item() is False

    # AFTER_K => opening quote only
    gs = automaton.step(gs, tok.vocabulary.token_id("<K>"))
    m = allowed_token_mask(gs, automaton, tok)
    assert m[tok.vocabulary.token_id('"')].item() is True
    assert m.any().item() is True  # at least one allowed


def test_masks_inside_string_allow_bpe_or_quote() -> None:
    """Inside strings, any BPE piece or closing quote must be allowed."""
    automaton, _vocab = _tok()
    tok = automaton.tok
    gs = automaton.start()
    gs = automaton.step(gs, tok.vocabulary.token_id("{"))
    gs = automaton.step(gs, tok.vocabulary.token_id("<K>"))
    gs = automaton.step(gs, tok.vocabulary.token_id('"'))  # enter key string

    m = allowed_token_mask(gs, automaton, tok)
    # Closing quote allowed:
    assert m[tok.vocabulary.token_id('"')].item() is True
    # Some BPE id (e.g., 0) mapped into joint space should be allowed when bpe_size > 0.
    if tok.bpe_size > 0:
        bpe0 = tok._bpe_to_joint_id(0)
        assert m[bpe0].item() is True


def test_end_to_end_sequence_masks_allow_all_steps() -> None:
    """Walk a real sequence and ensure each next token is allowed; EOS only after final '}'."""
    automaton, _vocab = _tok()
    tok = automaton.tok

    s = entity_to_string({"a": ["x", "y"]})
    ids = tok.encode(s, add_bos_eos=True)

    # Skip BOS at position 0; automaton starts at START and expects '{'.
    gs = automaton.start()

    # Consume all tokens except EOS; each must be allowed by the mask.
    for t in ids[1:-1]:
        m = allowed_token_mask(gs, automaton, tok)
        assert m[t].item() is True, f"Token id {t} not allowed in state {gs.state}"
        gs = automaton.step(gs, t)

    # After consuming '}', EOS is allowed and '{' (to start another entity).
    assert gs.state == State.END
    m_end = allowed_token_mask(gs, automaton, tok)
    assert m_end[tok.vocabulary.eos_id].item() is True
    # NEW: '{' is allowed at END to start another entity (Kleene-plus).
    assert m_end[tok.vocabulary.token_id("{")].item() is True
    # Other structural tokens still shouldn't be allowed at END.
    assert m_end[tok.vocabulary.token_id("}")].item() is False
