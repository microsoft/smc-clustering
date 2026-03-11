# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for constrained greedy decoding: validity, determinism, and mask adherence.

We use a uniform dummy model that assigns equal logits to every token. Constrained greedy then always picks the lowest-ID
allowed token at each step, which produces the smallest valid object: '{ }'. We verify that the result parses and that
every generated token was allowed by the mask at generation time.
"""

from __future__ import annotations

import torch
from torch import nn

from smc_clustering.jsonlm.grammar.automaton import GrammarAutomaton
from smc_clustering.jsonlm.grammar.mask import allowed_token_mask
from smc_clustering.jsonlm.models.decode import decode_greedy
from smc_clustering.jsonlm.serialization.encoder import entity_to_string, parse_entity
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


class UniformDummyModel(nn.Module):
    """A dummy model that returns zeros (equal logits) with EOS bias."""

    def __init__(self, vocab_size: int, eos_id: int) -> None:
        """Initialize the dummy model with a fixed vocabulary size and EOS bias target."""
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


def _tokenizer() -> tuple:
    vocab = Vocabulary.from_default()
    corpus = [
        entity_to_string({"a": ["x", "y"], "b": ["c"]}),
        entity_to_string({"title": ["Notes"], "tags": ["ai", "ml"]}),
    ]
    tok = train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=128)
    return tok, vocab


def test_decode_uniform_model_yields_valid_empty_object() -> None:
    """Model with EOS bias + constrained greedy should yield the minimal valid entity: '{ }'."""
    tok, _ = _tokenizer()
    model = UniformDummyModel(vocab_size=len(tok), eos_id=tok.vocabulary.eos_id)
    text = decode_greedy(model, tokenizer=tok, max_steps=64)
    # Should parse and equal {}.
    parsed = parse_entity(text)
    assert parsed == {}, f"Expected empty object, got: {parsed!r}"


def test_decode_deterministic_and_mask_respected() -> None:
    """Run decoding twice and ensure identical output; also verify each token was allowed by the mask."""
    tok, _ = _tokenizer()
    model = UniformDummyModel(vocab_size=len(tok), eos_id=tok.vocabulary.eos_id)

    text1 = decode_greedy(model, tokenizer=tok, max_steps=64)
    text2 = decode_greedy(model, tokenizer=tok, max_steps=64)
    assert text1 == text2

    # Verify every generated token was allowed at generation time.
    ids = tok.encode(text1, add_bos_eos=True)
    automaton = GrammarAutomaton(tok)
    gs = automaton.start()
    # Skip BOS; evaluate each next token's allowance until EOS.
    for t in ids[1:]:
        m = allowed_token_mask(gs, automaton, tok)
        assert m[t].item() is True, f"Token id {t} not allowed in state {gs.state}"
        if t == tok.vocabulary.eos_id:
            break
        gs = automaton.step(gs, t)
