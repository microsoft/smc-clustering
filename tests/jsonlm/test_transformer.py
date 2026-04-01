# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Tests for the tiny GPT-style Transformer: shapes, weight tying, and causality.

We verify logits shapes match input sizes, that lm_head is tied to tok_emb if requested, and that
changing future tokens does not affect earlier timestep logits (causal masking).
"""

from __future__ import annotations

import torch

from smc_clustering.jsonlm.models.transformer import (
    TransformerConfig,
    TransformerLM,
    _build_rope_cache,
)
from smc_clustering.jsonlm.serialization.encoder import entity_to_string
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _build_tokenizer() -> tuple:
    vocab = Vocabulary.from_default()
    corpus = [
        entity_to_string({"a": ["x", "y"], "b": ["c"]}),
        entity_to_string({"a": ["x"], "b": []}),
    ]
    tok = train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=128)
    return tok, vocab


def test_forward_shapes_and_tying() -> None:
    """Model returns [B, T, V] logits and can tie embedding/head weights."""
    tok, _ = _build_tokenizer()
    V = len(tok)
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

    # Dummy input of equal-length sequences.
    s = entity_to_string({"a": ["x", "y"], "b": ["c"]})
    ids = tok.encode(s, add_bos_eos=True)
    batch = torch.tensor([ids, ids], dtype=torch.long)  # [B=2, T]

    logits = model(batch[:, :-1])  # predict next token
    assert logits.shape == (batch.shape[0], batch.shape[1] - 1, V)

    # Weight tying check: weights share storage.
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_causality_no_peek_ahead() -> None:
    """Logits at positions <= t are unchanged if we alter tokens after t."""
    torch.manual_seed(0)
    tok, _ = _build_tokenizer()
    V = len(tok)
    cfg = TransformerConfig(
        vocab_size=V,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
        tie_embeddings=False,
    )
    model = TransformerLM(cfg).eval()  # disable dropout

    s = entity_to_string({"a": ["x", "y"], "b": ["c"]})
    idsA = tok.encode(s, add_bos_eos=True)
    # Create a variant that differs only in the last few tokens (after step t_cut).
    idsB = idsA.copy()
    # Flip one token near the end safely to a known structural token (e.g., ':') where possible.
    # To ensure a change after position t_cut, we modify the last non-EOS token if it's not a special; otherwise, we pick another.
    t_cut = min(len(idsA) - 3, 3)  # compare up to this step
    # Modify a future token deterministically: swap with BOS (harmless change).
    idsB[-3] = (
        tok.vocabulary.token_id(",")
        if idsA[-3] != tok.vocabulary.token_id(",")
        else tok.vocabulary.token_id(":")
    )

    inpA = torch.tensor([idsA[:-1]], dtype=torch.long)  # [1, T]
    inpB = torch.tensor([idsB[:-1]], dtype=torch.long)
    logitsA = model(inpA)  # [1, T, V]
    logitsB = model(inpB)

    # Compare logits up to t_cut (inclusive), which must be identical under causality.
    assert torch.allclose(logitsA[0, : t_cut + 1], logitsB[0, : t_cut + 1], rtol=1e-5, atol=1e-5)


def test_rope_vs_learned_positions() -> None:
    """RoPE and learned positional embeddings produce different but valid outputs."""
    torch.manual_seed(42)
    tok, _ = _build_tokenizer()
    V = len(tok)

    # Create configs for both positional encoding types
    base_cfg = {
        "vocab_size": V,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 128,
        "max_seq_len": 64,
        "dropout": 0.0,
        "tie_embeddings": True,
    }

    cfg_rope = TransformerConfig(**base_cfg, pos_encoding="rope")
    cfg_learned = TransformerConfig(**base_cfg, pos_encoding="learned")

    model_rope = TransformerLM(cfg_rope).eval()
    model_learned = TransformerLM(cfg_learned).eval()

    # Test input
    s = entity_to_string({"a": ["x", "y"], "b": ["c"]})
    ids = tok.encode(s, add_bos_eos=True)
    batch = torch.tensor([ids], dtype=torch.long)  # [1, T]

    with torch.no_grad():
        logits_rope = model_rope(batch)
        logits_learned = model_learned(batch)

    # Both should produce valid logits of the same shape
    assert logits_rope.shape == logits_learned.shape
    assert logits_rope.shape == (1, len(ids), V)

    # They should be different (different positional encodings)
    assert not torch.allclose(logits_rope, logits_learned, atol=1e-3)

    # Both should be finite and well-behaved
    assert torch.isfinite(logits_rope).all()
    assert torch.isfinite(logits_learned).all()


def test_rope_cache_shapes_and_building() -> None:
    """RoPE cache is built with correct shapes and contains valid values."""
    T, H, theta = 32, 16, 10000.0
    device = torch.device("cpu")
    dtype = torch.float32

    cache = _build_rope_cache(T, H, theta, device, dtype)

    # Shape should be [T, H/2, 2]
    assert cache.shape == (T, H // 2, 2)
    assert cache.device == device
    assert cache.dtype == dtype

    # Extract cos and sin
    cos_vals = cache[..., 0]  # [T, H/2]
    sin_vals = cache[..., 1]  # [T, H/2]

    # Both should be finite and in range [-1, 1]
    assert torch.isfinite(cos_vals).all()
    assert torch.isfinite(sin_vals).all()
    assert (cos_vals >= -1.0).all() and (cos_vals <= 1.0).all()
    assert (sin_vals >= -1.0).all() and (sin_vals <= 1.0).all()

    # For position 0, cos should be close to 1, sin close to 0
    assert torch.allclose(cos_vals[0], torch.ones_like(cos_vals[0]), atol=1e-3)
    assert torch.allclose(sin_vals[0], torch.zeros_like(sin_vals[0]), atol=1e-3)


def test_rope_sequence_length_extension() -> None:
    """RoPE models can handle sequences longer than max_seq_len during inference."""
    torch.manual_seed(42)
    tok, _ = _build_tokenizer()
    V = len(tok)

    cfg = TransformerConfig(
        vocab_size=V,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=16,  # Small max_seq_len
        dropout=0.0,
        pos_encoding="rope",
    )
    model = TransformerLM(cfg).eval()

    # Create a sequence longer than max_seq_len
    long_sequence = torch.randint(0, V, (1, 24))  # 24 > 16

    # Should work without errors (RoPE cache dynamically extends)
    with torch.no_grad():
        logits = model(long_sequence)

    assert logits.shape == (1, 24, V)
    assert torch.isfinite(logits).all()

    # The cache should have been extended
    actual_cache_len = model.blocks[0].attn.rope_cache.size(0)
    assert actual_cache_len >= 24


def test_rope_parameter_count() -> None:
    """RoPE models have fewer parameters than learned positional embedding models."""
    tok, _ = _build_tokenizer()
    V = len(tok)

    cfg_base = {
        "vocab_size": V,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 128,
        "max_seq_len": 64,
        "dropout": 0.0,
        "tie_embeddings": True,
    }

    cfg_rope = TransformerConfig(**cfg_base, pos_encoding="rope")
    cfg_learned = TransformerConfig(**cfg_base, pos_encoding="learned")

    model_rope = TransformerLM(cfg_rope)
    model_learned = TransformerLM(cfg_learned)

    # Count parameters
    rope_params = sum(p.numel() for p in model_rope.parameters())
    learned_params = sum(p.numel() for p in model_learned.parameters())

    # RoPE should have fewer parameters (no pos_emb)
    expected_diff = cfg_base["max_seq_len"] * cfg_base["d_model"]  # pos_emb size
    assert learned_params - rope_params == expected_diff

    # Verify RoPE model has no pos_emb
    assert model_rope.pos_emb is None
    assert model_learned.pos_emb is not None


def test_rope_configuration_options() -> None:
    """Different RoPE configuration options work correctly."""
    tok, _ = _build_tokenizer()
    V = len(tok)

    # Test different theta values
    cfg1 = TransformerConfig(
        vocab_size=V,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=128,
        pos_encoding="rope",
        rope_theta=5000.0,
    )
    cfg2 = TransformerConfig(
        vocab_size=V,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=128,
        pos_encoding="rope",
        rope_theta=20000.0,
    )

    model1 = TransformerLM(cfg1).eval()
    model2 = TransformerLM(cfg2).eval()

    # Test input
    test_input = torch.randint(0, V, (1, 10))

    with torch.no_grad():
        logits1 = model1(test_input)
        logits2 = model2(test_input)

    # Should have same shapes but different values (different theta)
    assert logits1.shape == logits2.shape
    assert not torch.allclose(logits1, logits2, atol=1e-3)

    # Test learned positional embeddings still work
    cfg_learned = TransformerConfig(
        vocab_size=V, d_model=64, n_layers=1, n_heads=4, d_ff=128, pos_encoding="learned"
    )
    model_learned = TransformerLM(cfg_learned).eval()

    with torch.no_grad():
        logits_learned = model_learned(test_input)

    assert logits_learned.shape == logits1.shape
    assert model_learned.pos_emb is not None
