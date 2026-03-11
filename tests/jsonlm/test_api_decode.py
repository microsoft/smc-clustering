# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the decode wrapper: it returns a canonical dict and matches parse(decode_greedy).

We use a uniform-logit dummy model so constrained greedy emits the minimal valid entity '{}'.
"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer
from torch import nn

from smc_clustering.jsonlm.api import decode_entity
from smc_clustering.jsonlm.models.decode import decode_greedy
from smc_clustering.jsonlm.serialization.encoder import entity_to_string, parse_entity
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


class UniformDummyModel(nn.Module):
    """A dummy model that returns zeros (equal logits) with EOS bias."""

    def __init__(self, vocab_size: int, eos_id: int) -> None:
        """Initialize the model."""
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return zeros logits with slight EOS bias to prefer termination over continuation."""
        B, T = input_ids.shape
        V = self.vocab_size
        logits = torch.zeros((B, T, V), dtype=torch.float32, device=input_ids.device)
        # Slight bias toward EOS to prefer termination in Kleene-plus grammar
        logits[:, :, self.eos_id] = 0.1
        return logits


def _tokenizer() -> Tokenizer:
    """Train and return a tokenizer for testing."""
    vocab = Vocabulary.from_default()
    corpus = [
        entity_to_string({"a": ["x", "y"], "b": ["c"]}),
        entity_to_string({"title": ["Notes"], "tags": ["ai", "ml"]}),
    ]
    tok = train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=128)
    return tok


def test_decode_entity_returns_canonical_dict() -> None:
    """decode_entity should return {} for a model with EOS bias and match parse(decode_greedy)."""
    tok = _tokenizer()
    model = UniformDummyModel(vocab_size=len(tok), eos_id=tok.vocabulary.eos_id)

    # Public wrapper
    d = decode_entity(model=model, tokenizer=tok, max_steps=64)
    assert d == {}

    # Compare with direct decode_greedy + parse
    s = decode_greedy(model=model, tokenizer=tok, max_steps=64)
    assert d == parse_entity(s)
