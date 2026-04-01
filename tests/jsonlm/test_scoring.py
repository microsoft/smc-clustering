# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Unit tests for batched scoring utilities: score_entities_batched and compute_deltas_batched.

We verify:
  * Batched log-likelihoods (sum/mean/bpt) match single-item API results.
  * Mixed dict/string inputs are handled identically.
  * Variable-length entities batch without errors (EOS-tail padding is respected).
  * Batched Δ matches api.delta for a list of pairs, preserving order.
"""

from __future__ import annotations

import math
import random

import torch
from torch import nn

from smc_clustering.jsonlm.api import delta, logprob_entity
from smc_clustering.jsonlm.models.scoring import compute_deltas_batched, score_entities_batched
from smc_clustering.jsonlm.serialization.encoder import entity_to_string
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


class DummyModel(nn.Module):
    """A deterministic LM that returns per-id biases as logits (independent of input/time).

    Stable across hardware/backends; perfect for equality-style tests of scoring logic.
    """

    def __init__(self, vocab_size: int, bias_scale: float = 0.01) -> None:
        """Initialize a deterministic bias-only scorer."""
        super().__init__()
        bias = torch.arange(vocab_size, dtype=torch.float32) * bias_scale
        self.register_buffer("_bias", bias, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return broadcasted per-token logits with no dependence on the input sequence."""
        B, T = input_ids.shape
        V = self._bias.numel()
        # Broadcast [V] -> [B, T, V]
        return self._bias.view(1, 1, V).expand(B, T, V).clone()


def _make_tokenizer() -> tuple:
    """Train a tiny tokenizer on a small serialized corpus."""
    vocab = Vocabulary.from_default()
    corpus = [
        entity_to_string({"a": ["x", "y"], "b": ["c"]}),
        entity_to_string({"title": ["Notes"], "tags": ["ai", "ml"]}),
        entity_to_string({"name": ["Ada Lovelace"], "field": ["computing"]}),
    ]
    tok = train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=64)
    return tok, vocab


def test_batched_scores_match_single_api() -> None:
    """score_entities_batched should equal logprob_entity for each item (sum/mean/bpt)."""
    tok, _ = _make_tokenizer()
    model = DummyModel(vocab_size=len(tok))

    # Mix dicts and serialized strings + variable lengths.
    ents = [
        {"a": ["x", "y"], "b": ["c"]},
        {"a": ["x"]},
        {"title": ["Notes"], "tags": ["ai", "ml"]},
    ]
    texts = [entity_to_string(e) for e in ents]
    mixed = [ents[0], texts[1], ents[2]]  # dict, str, dict

    # Compare for sum
    batched_sum = score_entities_batched(
        mixed,
        model=model,
        tokenizer=tok,
        normalize="sum",
        batch_size=2,
        include_eos=True,
    )
    single_sum = [logprob_entity(x, model=model, tokenizer=tok, normalize="sum") for x in mixed]
    assert len(batched_sum) == len(single_sum)
    for a, b in zip(batched_sum, single_sum, strict=False):
        assert math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8)

    # Compare for mean
    batched_mean = score_entities_batched(
        mixed,
        model=model,
        tokenizer=tok,
        normalize="mean",
        batch_size=2,
        include_eos=True,
    )
    single_mean = [logprob_entity(x, model=model, tokenizer=tok, normalize="mean") for x in mixed]
    for a, b in zip(batched_mean, single_mean, strict=False):
        assert math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)

    # Compare for bits-per-token
    batched_bpt = score_entities_batched(
        mixed,
        model=model,
        tokenizer=tok,
        normalize="bpt",
        batch_size=2,
        include_eos=True,
    )
    single_bpt = [logprob_entity(x, model=model, tokenizer=tok, normalize="bpt") for x in mixed]
    for a, b in zip(batched_bpt, single_bpt, strict=False):
        assert math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)


def test_batched_scoring_handles_variable_lengths() -> None:
    """Ensure different sequence lengths (due to different entity sizes) don't break batching."""
    tok, _ = _make_tokenizer()
    model = DummyModel(vocab_size=len(tok))

    # Create a set of entities with varying key/value counts to force different lengths.
    entities = [
        {"k": []},
        {"k": ["v"]},
        {"a": ["x", "y"], "b": ["c"]},
        {"title": ["A"], "tags": ["ai", "ml", "nlp"]},
    ]
    # Shuffle to avoid any accidental ordering artifacts.
    random.seed(0)
    random.shuffle(entities)

    scores = score_entities_batched(
        entities,
        model=model,
        tokenizer=tok,
        normalize="sum",
        batch_size=3,
    )
    assert len(scores) == len(entities)
    # Finite sanity
    for s in scores:
        assert math.isfinite(s)


def test_compute_deltas_batched_matches_api_delta() -> None:
    """compute_deltas_batched should equal api.delta for every pair (order preserved)."""
    tok, _ = _make_tokenizer()
    model = DummyModel(vocab_size=len(tok))

    A1 = {"a": ["x"]}
    B1 = {"a": ["x", "y"]}
    A2 = {"b": ["c"]}
    B2 = {"title": ["Notes"], "tags": ["ai"]}

    pairs = [(A1, B1), (A2, B2)]
    # Batched
    deltas_batched = compute_deltas_batched(pairs, model=model, tokenizer=tok, batch_size=2)
    # Single via API
    deltas_single = [delta(a, b, model=model, tokenizer=tok) for (a, b) in pairs]

    assert len(deltas_batched) == len(deltas_single)
    for a, b in zip(deltas_batched, deltas_single, strict=False):
        assert math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8)


def test_batched_accepts_dicts_and_serialized_strings() -> None:
    """Mixed input types should score identically after canonicalization and serialization."""
    tok, _ = _make_tokenizer()
    model = DummyModel(vocab_size=len(tok))

    d = {"name": ["Ada", "Lovelace"], "field": ["computing"]}
    s = entity_to_string(d)  # serialized (will be parsed & canonicalized inside scoring)
    scores = score_entities_batched(
        [d, s],
        model=model,
        tokenizer=tok,
        normalize="sum",
        batch_size=2,
    )

    assert len(scores) == 2
    assert math.isclose(scores[0], scores[1], rel_tol=0.0, abs_tol=1e-8)
