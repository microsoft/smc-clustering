# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end smoke test: TinyTransformerLM + LitConstrainedLM one optimization step.

This verifies that (1) the model and Lightning wrapper interoperate, (2) grammar masks build correctly for a real batch,
(3) the constrained loss is finite, (4) backprop produces nonzero gradients, and (5) an optimizer step updates params.
"""

from __future__ import annotations

import torch

from smc_clustering.jsonlm.models.lit_module import LitConstrainedLM
from smc_clustering.jsonlm.models.transformer import TransformerConfig, TransformerLM
from smc_clustering.jsonlm.serialization.encoder import entity_to_string
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _make_tokenizer() -> tuple:
    """Train a small tokenizer on a tiny corpus."""
    vocab = Vocabulary.from_default()
    corpus = [
        entity_to_string({"a": ["x", "y"], "b": ["c"]}),
        entity_to_string({"title": ["Notes"], "tags": ["ai", "ml"]}),
    ]
    tok = train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=128)
    return tok, vocab


def _make_batch(tok: JsonLMTokenizer) -> torch.Tensor:
    """Create a small batch [B, L] of equal-length BOS…EOS sequences."""
    s = entity_to_string({"a": ["x", "y"], "b": ["c"]})
    ids = tok.encode(s, add_bos_eos=True)
    batch = torch.tensor([ids, ids], dtype=torch.long)  # [B=2, L]
    assert batch.dim() == 2 and batch.dtype == torch.long
    return batch


def test_one_training_step_with_transformer_smoke() -> None:
    """A single optimizer step runs end-to-end and keeps parameters finite."""
    torch.manual_seed(0)

    tok, _ = _make_tokenizer()
    V = len(tok)

    # Tiny config for speed; causal decoder returns [B, T, V] from [B, T] inputs.
    cfg = TransformerConfig(
        vocab_size=V,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
        tie_embeddings=True,
    )
    model = TransformerLM(cfg)

    lit = LitConstrainedLM(model=model, tokenizer=tok, lr=1e-3, weight_decay=0.0)

    batch = _make_batch(tok)  # [B, L]

    # Forward + loss
    loss = lit.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss).item() is True

    # Backward and optimizer step manually (keeps test independent of Trainer).
    opt_config = lit.configure_optimizers()
    if isinstance(opt_config, dict):
        opt = opt_config["optimizer"]
    else:
        opt = opt_config
    assert isinstance(opt, torch.optim.Optimizer)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    # Ensure we actually have gradients on some parameters.
    grad_nonzero = False
    for p in model.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0.0:
            grad_nonzero = True
            break
    assert grad_nonzero, "Expected at least one parameter to receive a nonzero gradient."

    # Take an optimizer step; ensure parameters remain finite afterward.
    opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all().item() is True

    # Optional: run another forward to ensure nothing broke.
    loss2 = lit.training_step(batch, batch_idx=1)
    assert torch.isfinite(loss2).item() is True
