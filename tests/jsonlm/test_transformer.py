"""
Tests for the tiny GPT-style Transformer: shapes, weight tying, and causality.

We verify logits shapes match input sizes, that lm_head is tied to tok_emb if requested, and that
changing future tokens does not affect earlier timestep logits (causal masking).
"""

from __future__ import annotations

import torch

from jsonlm.models.transformer import TinyTransformerLM, TransformerConfig
from jsonlm.serialization.encoder import entity_to_string
from jsonlm.tokenization.trainer import train_tokenizer
from jsonlm.tokenization.vocab import Vocabulary


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
    model = TinyTransformerLM(cfg)

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
    model = TinyTransformerLM(cfg).eval()  # disable dropout

    s = entity_to_string({"a": ["x", "y"], "b": ["c"]})
    idsA = tok.encode(s, add_bos_eos=True)
    # Create a variant that differs only in the last few tokens (after step t_cut).
    idsB = idsA.copy()
    # Flip one token near the end safely to a known structural token (e.g., ':') where possible.
    # To ensure a change after position t_cut, we modify the last non-EOS token if it's not a special; otherwise, we pick another.
    t_cut = min(len(idsA) - 3, 3)  # compare up to this step
    # Modify a future token deterministically: swap with BOS (harmless change).
    idsB[-3] = (
        tok.vocabulary.token_id(",") if idsA[-3] != tok.vocabulary.token_id(",") else tok.vocabulary.token_id(":")
    )

    inpA = torch.tensor([idsA[:-1]], dtype=torch.long)  # [1, T]
    inpB = torch.tensor([idsB[:-1]], dtype=torch.long)
    logitsA = model(inpA)  # [1, T, V]
    logitsB = model(inpB)

    # Compare logits up to t_cut (inclusive), which must be identical under causality.
    assert torch.allclose(logitsA[0, : t_cut + 1], logitsB[0, : t_cut + 1], rtol=1e-5, atol=1e-5)
