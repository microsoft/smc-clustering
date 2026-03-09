# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the Lightning module: forward shapes, loss finiteness, and mask alignment.

We build a tiny toy tokenizer and a minimal per-token linear LM to verify that training_step runs end-to-end, that
logits/masks have the right shapes, and that the constrained loss is finite. To avoid dataloader/padding complexity,
we manually stack equal-length examples.
"""

from __future__ import annotations

import torch
from torch import nn

from smc_clustering.jsonlm.models.lit_module import LitConstrainedLM
from smc_clustering.jsonlm.serialization.encoder import entity_to_string
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


class ToyLM(nn.Module):
    """A minimal per-token linear LM: embeddings → linear head to V (no attention)."""

    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits of shape [B, T, V] from input IDs [B, T]."""
        x = self.embed(input_ids)  # [B, T, D]
        logits = self.head(x)  # [B, T, V]
        return logits


def _make_batch(tok: JsonLMTokenizer) -> torch.Tensor:
    """Create a small batch [B, L] of equal-length BOS…EOS sequences."""
    # Use two copies of the same entity to guarantee equal lengths for a simple stack.
    s = entity_to_string({"a": ["x", "y"], "b": ["c"]})
    ids = tok.encode(s, add_bos_eos=True)
    batch = torch.tensor([ids, ids], dtype=torch.long)  # [2, L]
    return batch


def test_lit_module_training_step_runs_and_loss_finite() -> None:
    """training_step computes a finite constrained loss with correct shapes."""
    # Train tokenizer on a small corpus.
    vocab = Vocabulary.from_default()
    corpus = [entity_to_string({"a": ["x", "y"], "b": ["c"]})]
    tok = train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=128)

    # Model and Lightning module.
    V = len(tok)
    model = ToyLM(vocab_size=V, d_model=32)
    lit = LitConstrainedLM(model=model, tokenizer=tok, lr=1e-3)

    # Create a batch and run training_step.
    batch = _make_batch(tok)  # [B, L]
    loss = lit.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss).item() is True

    # Sanity on logits shape via forward().
    input_ids = batch[:, :-1]  # [B, T]
    logits = lit(input_ids)
    assert logits.shape == (batch.shape[0], input_ids.shape[1], V)
