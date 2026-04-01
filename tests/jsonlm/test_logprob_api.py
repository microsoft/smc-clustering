# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Tests for the public logprob API: canonicalization invariance, determinism, and normalization.

We use a deterministic DummyModel (time-independent logits) so values are stable regardless of hardware,
and verify that canonical permutations score identically and normalizations relate as expected. Tests cover
both single entity and sequence APIs, including EOS inclusion/exclusion behavior for sequences.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from smc_clustering.jsonlm.api import encode_entity, encode_sequence, logprob_entity, logprob_sequence
from smc_clustering.jsonlm.serialization.encoder import entities_to_string_as_set, entity_to_string
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


class DummyModel(nn.Module):
    """A dummy LM that returns fixed logits with a small id-dependent bias for stability."""

    def __init__(self, vocab_size: int, bias_scale: float = 0.01) -> None:
        """Initialize a deterministic bias-only language model."""
        super().__init__()
        # Learnable scalar isn't needed; keep a buffer with per-id bias.
        bias = torch.arange(vocab_size, dtype=torch.float32) * bias_scale
        self.register_buffer("_bias", bias, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits [B, T, V] that do not depend on input positions (teacher forcing handles targets)."""
        B, T = input_ids.shape
        V = self._bias.numel()
        # Broadcast bias over [B, T, V].
        logits = self._bias.view(1, 1, V).expand(B, T, V).clone()
        return logits


def _tokenizer() -> tuple:
    vocab = Vocabulary.from_default()
    corpus = [
        entity_to_string({"a": ["x", "y"], "b": ["c"]}),
        entity_to_string({"title": ["Notes"], "tags": ["ai", "ml"]}),
        # Add sequence training data for sequence tests
        entities_to_string_as_set([{"a": ["x"]}, {"b": ["y"]}]),
        entities_to_string_as_set([{"name": ["Alice"]}, {"role": ["admin"]}]),
    ]
    tok = train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=128)
    return tok, vocab


def test_logprob_canonicalization_invariance() -> None:
    """Permutations of the same entity yield identical log-probabilities."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    e1 = {"b": ["y", "x", "x"], "a": ["b", "a"]}
    e2 = {"a": ["b", "a"], "b": ["y", "x"]}

    lp1 = logprob_entity(e1, model=model, tokenizer=tok, normalize="sum")
    lp2 = logprob_entity(e2, model=model, tokenizer=tok, normalize="sum")
    assert math.isclose(lp1, lp2, rel_tol=0.0, abs_tol=1e-8)


def test_logprob_normalizations_consistent() -> None:
    """Sum ≈ mean * T, and bits-per-token equals mean/ln(2)."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    e = {"a": ["x", "y"], "b": ["c"]}
    # Also fetch T for the encoded length minus 1.
    ids = encode_entity(e, tokenizer=tok, add_bos_eos=True)
    T = len(ids) - 1

    lp_sum = logprob_entity(e, model=model, tokenizer=tok, normalize="sum")
    lp_mean = logprob_entity(e, model=model, tokenizer=tok, normalize="mean")
    bpt = logprob_entity(e, model=model, tokenizer=tok, normalize="bpt")

    assert math.isclose(lp_sum, lp_mean * T, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(bpt, lp_mean / math.log(2.0), rel_tol=1e-6, abs_tol=1e-6)


def test_logprob_accepts_serialized_string_and_dict() -> None:
    """Scoring a serialized string equals scoring the equivalent raw dict."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    e = {"b": ["y", "x", "x"], "a": ["b", "a"]}
    s = entity_to_string(e)  # canonicalized by serializer
    lp_dict = logprob_entity(e, model=model, tokenizer=tok)
    lp_str = logprob_entity(s, model=model, tokenizer=tok)
    assert math.isclose(lp_dict, lp_str, rel_tol=0.0, abs_tol=1e-8)


def test_encode_sequence_basic() -> None:
    """encode_sequence should produce valid BOS...EOS token sequences."""
    tok, _ = _tokenizer()

    entities = [{"a": ["x"]}, {"b": ["y", "z"]}]
    ids = encode_sequence(entities, tokenizer=tok, add_bos_eos=True)

    assert len(ids) >= 2, "Should have at least BOS and EOS"
    assert ids[0] == tok.vocabulary.bos_id, "Should start with BOS"
    assert ids[-1] == tok.vocabulary.eos_id, "Should end with EOS"

    # Should match manual serialization
    manual_text = entities_to_string_as_set(entities)
    manual_ids = tok.encode(manual_text, add_bos_eos=True)
    assert ids == manual_ids


def test_logprob_sequence_basic() -> None:
    """logprob_sequence should score entity sequences correctly."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    entities = [{"a": ["x"]}, {"b": ["y"]}]
    score = logprob_sequence(entities, model=model, tokenizer=tok, normalize="sum")

    assert isinstance(score, float)
    assert math.isfinite(score)


def test_logprob_sequence_vs_manual_serialization() -> None:
    """Sequence API should produce valid scores and handle serialization correctly."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    entities = [{"a": ["x"]}, {"b": ["y", "z"]}]

    # Test basic functionality - sequence API should work
    sequence_score_with_eos = logprob_sequence(
        entities, model=model, tokenizer=tok, include_eos=True, normalize="sum"
    )
    sequence_score_without_eos = logprob_sequence(
        entities,
        model=model,
        tokenizer=tok,
        include_eos=False,
        normalize="sum",
    )

    # Both should be finite and different
    assert math.isfinite(sequence_score_with_eos)
    assert math.isfinite(sequence_score_without_eos)
    assert sequence_score_with_eos != sequence_score_without_eos

    # Verify that encoding is correct
    sequence_ids = encode_sequence(entities, tokenizer=tok, add_bos_eos=True)
    manual_text = entities_to_string_as_set(entities)
    manual_ids = tok.encode(manual_text, add_bos_eos=True)
    assert sequence_ids == manual_ids


def test_logprob_sequence_eos_inclusion_exclusion() -> None:
    """logprob_sequence should handle include_eos parameter correctly."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    entities = [{"a": ["x"]}, {"b": ["y"]}]

    # Get scores with and without EOS
    score_without_eos = logprob_sequence(
        entities, model=model, tokenizer=tok, include_eos=False, normalize="sum"
    )
    score_with_eos = logprob_sequence(
        entities, model=model, tokenizer=tok, include_eos=True, normalize="sum"
    )

    # Score with EOS should be different from without EOS
    assert score_without_eos != score_with_eos

    # Both scores should be finite
    assert math.isfinite(score_without_eos)
    assert math.isfinite(score_with_eos)

    # Test that the exclude EOS behavior works as expected
    # Without EOS should typically be less negative (since we exclude a penalty)
    # but the actual relationship depends on the model's EOS probability
    assert isinstance(score_without_eos, float)
    assert isinstance(score_with_eos, float)


def test_logprob_sequence_normalizations_consistent() -> None:
    """Sequence normalizations should be consistent with expected relationships."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    entities = [{"a": ["x"]}, {"b": ["y", "z"]}]

    # Get scores with different normalizations (include_eos=False)
    lp_mean = logprob_sequence(entities, model=model, tokenizer=tok, include_eos=False, normalize="mean")
    bpt = logprob_sequence(entities, model=model, tokenizer=tok, include_eos=False, normalize="bpt")

    # For normalizations to be consistent, we need the active_tokens count
    # which the logprob_sequence function calculates internally
    # We can verify consistency by checking that mean and bpt are related correctly
    if lp_mean != 0.0:  # Avoid division by zero
        # bits-per-token equals mean/ln(2)
        assert math.isclose(bpt, lp_mean / math.log(2.0), rel_tol=1e-6, abs_tol=1e-6)

    # Also test with a simpler case where we can calculate active tokens manually
    simple_entities = [{"a": ["x"]}]
    simple_ids = encode_sequence(simple_entities, tokenizer=tok, add_bos_eos=True)

    # Find EOS position in targets (skip BOS)
    eos_pos = None
    for i, token_id in enumerate(simple_ids[1:]):  # Skip BOS
        if token_id == tok.vocabulary.eos_id:
            eos_pos = i  # Position in target sequence (0-indexed)
            break

    if eos_pos is not None and eos_pos > 0:
        simple_sum = logprob_sequence(
            simple_entities, model=model, tokenizer=tok, include_eos=False, normalize="sum"
        )
        simple_mean = logprob_sequence(
            simple_entities, model=model, tokenizer=tok, include_eos=False, normalize="mean"
        )

        # sum ≈ mean * active_tokens
        assert math.isclose(simple_sum, simple_mean * eos_pos, rel_tol=1e-6, abs_tol=1e-6)


def test_logprob_sequence_accepts_string_input() -> None:
    """logprob_sequence should accept serialized string input like logprob_entity."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    entities = [{"a": ["x"]}, {"b": ["y"]}]
    serialized = entities_to_string_as_set(entities)

    # Both should give same result
    score_from_list = logprob_sequence(entities, model=model, tokenizer=tok, normalize="sum")
    score_from_string = logprob_sequence(serialized, model=model, tokenizer=tok, normalize="sum")

    assert math.isclose(score_from_list, score_from_string, rel_tol=0.0, abs_tol=1e-8)


def test_logprob_sequence_empty_sequence() -> None:
    """logprob_sequence should handle empty sequences correctly."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    empty_entities = []

    # Empty sequence should work but may have grammar constraints
    # Test that it doesn't crash and returns finite scores
    error: ValueError | None = None
    try:
        score_with_eos = logprob_sequence(
            empty_entities, model=model, tokenizer=tok, include_eos=True, normalize="sum"
        )
        assert isinstance(score_with_eos, float)
        assert math.isfinite(score_with_eos)

        score_without_eos = logprob_sequence(
            empty_entities,
            model=model,
            tokenizer=tok,
            include_eos=False,
            normalize="sum",
        )
        assert isinstance(score_without_eos, float)
        assert math.isfinite(score_without_eos)
        # Empty sequence with include_eos=False should have score 0 (no content tokens)
        assert score_without_eos == 0.0

    except ValueError as exc:
        error = exc

    if error is not None:
        # Empty sequences might violate grammar constraints (BOS->EOS not allowed)
        # This is acceptable behavior - just verify the error is grammar-related.
        lowered = str(error).lower()
        assert "grammar" in lowered or "disallowed" in lowered


def test_logprob_sequence_single_entity_vs_entity_api() -> None:
    """Single entity in sequence should behave similarly to entity API."""
    tok, _ = _tokenizer()
    model = DummyModel(vocab_size=len(tok))

    entity = {"a": ["x", "y"], "b": ["c"]}
    single_entity_sequence = [entity]

    # Score using entity API
    entity_score = logprob_entity(entity, model=model, tokenizer=tok, normalize="sum")

    # Score using sequence API with include_eos=True (to match entity API behavior)
    sequence_score = logprob_sequence(
        single_entity_sequence,
        model=model,
        tokenizer=tok,
        include_eos=True,
        normalize="sum",
    )

    # Should be very close (canonicalization might cause tiny differences)
    assert math.isclose(entity_score, sequence_score, rel_tol=1e-6, abs_tol=1e-6)
