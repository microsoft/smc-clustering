# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests to verify that the new runtime implementation produces correct results.

This module tests the runtime implementation across various entity structures and edge cases,
particularly with the new Kleene-plus grammar that allows multiple entities. Since the runtime
supersedes both old implementations, we test it against a simple reference implementation.
"""

from __future__ import annotations

import torch

from smc_clustering.jsonlm.grammar.automaton import GrammarAutomaton, GrammarState
from smc_clustering.jsonlm.grammar.mask import allowed_token_mask
from smc_clustering.jsonlm.grammar.runtime import get_runtime
from smc_clustering.jsonlm.serialization.encoder import entity_to_string
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _build_masks_reference(ids_with_eos: torch.Tensor, tokenizer) -> torch.BoolTensor:
    """Reference implementation using per-step automaton for comparison."""
    assert ids_with_eos.dim() == 2 and ids_with_eos.dtype == torch.long
    B, L = ids_with_eos.shape
    assert L >= 2
    V = len(tokenizer)
    T = L - 1
    masks = torch.zeros((B, T, V), dtype=torch.bool, device=ids_with_eos.device)
    eos = tokenizer.vocabulary.eos_id

    automaton = GrammarAutomaton(tokenizer)
    for b in range(B):
        seq = ids_with_eos[b]
        gs: GrammarState = automaton.start()
        for t in range(T):
            y_t = int(seq[t + 1].item())
            m = allowed_token_mask(gs, automaton, tokenizer)
            masks[b, t] = m.to(device=ids_with_eos.device)
            if y_t == eos:
                if t + 1 < T:
                    masks[b, t + 1 :, :] = False
                    masks[b, t + 1 :, eos] = True
                break
            gs = automaton.step(gs, y_t)
    return masks


def _make_tokenizer():
    """Create a tokenizer for testing."""
    vocab = Vocabulary.from_default()
    corpus = [
        entity_to_string({"a": ["x", "y"], "b": ["c"]}),
        entity_to_string({"title": ["Notes"], "tags": ["ai", "ml"]}),
        entity_to_string({"key": ["value"]}),
        entity_to_string({}),  # empty entity
    ]
    tok = train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=128)
    return tok


def _encode_entities_to_batch(entities: list[dict[str, list[str]]], tokenizer) -> torch.Tensor:
    """Encode a list of entities to a padded batch tensor."""
    # Encode each entity to BOS...EOS
    ids_list = []
    for entity in entities:
        text = entity_to_string(entity)
        ids = tokenizer.encode(text, add_bos_eos=True)
        ids_list.append(ids)

    # Pad to same length with EOS
    max_len = max(len(ids) for ids in ids_list)
    batch = torch.full((len(entities), max_len), tokenizer.vocabulary.eos_id, dtype=torch.long)

    for i, ids in enumerate(ids_list):
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    return batch


def test_runtime_mask_equivalence_basic():
    """Test that runtime implementation produces identical results to reference for basic entities."""
    tok = _make_tokenizer()
    device = torch.device("cpu")

    # Test various entity structures
    entities = [
        {"a": ["x"]},  # simple entity
        {"a": ["x", "y"]},  # multi-value
        {"a": ["x"], "b": ["y"]},  # multi-key
        {"title": ["Notes"], "tags": ["ai"]},  # different keys
        {},  # empty entity
    ]

    batch = _encode_entities_to_batch(entities, tok)

    # Generate masks with runtime and reference implementations
    rt = get_runtime(tok, device)
    masks_runtime = rt.build_masks(batch)
    masks_reference = _build_masks_reference(batch, tok)

    # Should be identical
    assert torch.equal(masks_runtime, masks_reference), (
        f"Runtime and reference implementations differ!\n"
        f"Shapes: runtime={masks_runtime.shape}, reference={masks_reference.shape}\n"
        f"First difference at: {torch.where(masks_runtime != masks_reference)}"
    )


def test_runtime_mask_equivalence_variable_lengths():
    """Test equivalence with variable-length sequences (different padding)."""
    tok = _make_tokenizer()
    device = torch.device("cpu")

    # Mix of short and long entities to test padding handling
    entities = [
        {"k": ["v"]},  # very short
        {"a": ["x", "y", "z"], "b": ["long", "values"]},  # longer
        {"title": ["A", "B"], "tags": ["ai", "ml", "nlp"]},  # longest
        {},  # shortest (empty)
    ]

    batch = _encode_entities_to_batch(entities, tok)

    rt = get_runtime(tok, device)
    masks_runtime = rt.build_masks(batch)
    masks_reference = _build_masks_reference(batch, tok)

    assert torch.equal(masks_runtime, masks_reference), (
        f"Variable length runtime and reference implementations differ!\nBatch shape: {batch.shape}\nEntities: {entities}"
    )


def test_runtime_mask_equivalence_edge_cases():
    """Test equivalence for edge cases like empty entities and single tokens."""
    tok = _make_tokenizer()
    device = torch.device("cpu")

    # Edge case entities
    entities = [
        {},  # empty entity (just {})
        {"": []},  # empty key with empty values (should be filtered out by canonicalization)
        {"a": []},  # key with no values (should be filtered out)
        {"x": [""]},  # empty string value (might be interesting)
    ]

    # Filter out invalid entities that canonicalization removes
    from jsonlm.serialization.encoder import canonicalize_entity

    valid_entities = []
    for entity in entities:
        try:
            canonical = canonicalize_entity(entity)
            if canonical:  # non-empty after canonicalization
                valid_entities.append(canonical)
            else:
                valid_entities.append({})  # empty entity
        except:
            valid_entities.append({})  # fallback to empty

    if not valid_entities:
        valid_entities = [{}]  # ensure at least one entity

    batch = _encode_entities_to_batch(valid_entities, tok)

    rt = get_runtime(tok, device)
    masks_runtime = rt.build_masks(batch)
    masks_reference = _build_masks_reference(batch, tok)

    assert torch.equal(masks_runtime, masks_reference), (
        f"Edge case runtime and reference implementations differ!\nValid entities: {valid_entities}\nBatch shape: {batch.shape}"
    )


def test_runtime_mask_equivalence_single_entity():
    """Test equivalence for single entity (common case)."""
    tok = _make_tokenizer()
    device = torch.device("cpu")

    entity = {"name": ["Alice", "Bob"], "role": ["admin"]}
    batch = _encode_entities_to_batch([entity], tok)

    rt = get_runtime(tok, device)
    masks_runtime = rt.build_masks(batch)
    masks_reference = _build_masks_reference(batch, tok)

    assert torch.equal(masks_runtime, masks_reference), (
        f"Single entity runtime and reference implementations differ!\n"
        f"Entity: {entity}\n"
        f"Batch shape: {batch.shape}\n"
        f"Mask shapes: runtime={masks_runtime.shape}, reference={masks_reference.shape}"
    )


def test_runtime_mask_equivalence_batch_sizes():
    """Test equivalence across different batch sizes."""
    tok = _make_tokenizer()
    device = torch.device("cpu")

    # Create a larger set of diverse entities
    entities = [
        {"a": ["x"]},
        {"b": ["y", "z"]},
        {"c": ["p"], "d": ["q"]},
        {"title": ["Test"], "tags": ["one", "two", "three"]},
        {},
        {"single": ["val"]},
        {"multi": ["a", "b"], "other": ["c", "d", "e"]},
    ]

    # Test different batch sizes
    for batch_size in [1, 3, 5, len(entities)]:
        test_entities = entities[:batch_size]
        batch = _encode_entities_to_batch(test_entities, tok)

        rt = get_runtime(tok, device)
        masks_runtime = rt.build_masks(batch)
        masks_reference = _build_masks_reference(batch, tok)

        assert torch.equal(masks_runtime, masks_reference), (
            f"Batch size {batch_size} runtime and reference implementations differ!\n"
            f"Entities: {test_entities}\n"
            f"Batch shape: {batch.shape}"
        )


def test_runtime_mask_states_at_each_position():
    """Test that runtime implementation handles grammar state transitions correctly."""
    tok = _make_tokenizer()
    device = torch.device("cpu")

    # Entity that exercises various grammar states
    entity = {"key1": ["val1", "val2"], "key2": ["val3"]}
    batch = _encode_entities_to_batch([entity], tok)

    rt = get_runtime(tok, device)
    masks_runtime = rt.build_masks(batch)
    masks_reference = _build_masks_reference(batch, tok)

    # Should be identical
    assert torch.equal(masks_runtime, masks_reference)

    # Additionally, check that the masks have the expected structure:
    # - At each position, only grammatically valid tokens should be allowed
    # - The number of allowed tokens should be reasonable (not all or none except at EOS)
    B, T, V = masks_runtime.shape

    for b in range(B):
        for t in range(T):
            allowed_count = masks_runtime[b, t].sum().item()
            # Sanity check: should have some allowed tokens (unless at end with EOS-only)
            assert allowed_count > 0, f"No tokens allowed at batch {b}, position {t}"
            assert allowed_count < V, f"All tokens allowed at batch {b}, position {t} (likely wrong)"


def test_runtime_mask_end_state_kleene_plus():
    """Test that runtime implementation correctly handles the Kleene-plus END state behavior."""
    tok = _make_tokenizer()
    device = torch.device("cpu")

    # Test entity that reaches END state
    entity = {"test": ["value"]}
    batch = _encode_entities_to_batch([entity], tok)

    rt = get_runtime(tok, device)
    masks_runtime = rt.build_masks(batch)
    masks_reference = _build_masks_reference(batch, tok)

    assert torch.equal(masks_runtime, masks_reference)

    # Specifically check the mask at the position where we'd be in END state
    # This would be after the closing '}' token, where we should allow both EOS and '{'
    lbrace_id = tok.vocabulary.token_id("{")
    eos_id = tok.vocabulary.eos_id

    # Find the position in the sequence where END state occurs
    # The exact position depends on the tokenization, but we can verify that
    # when EOS is allowed, '{' should also be allowed (Kleene-plus behavior)
    B, T, V = masks_runtime.shape
    for b in range(B):
        for t in range(T):
            if masks_runtime[b, t, eos_id]:  # If EOS is allowed
                # Then '{' should also be allowed (Kleene-plus)
                assert masks_runtime[b, t, lbrace_id], (
                    f"At batch {b}, position {t}: EOS allowed but '{{' not allowed. "
                    f"This violates Kleene-plus grammar where both should be allowed at END state."
                )
