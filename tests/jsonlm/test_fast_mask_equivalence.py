"""
Tests to verify that _build_masks_for_batch_fast and _build_masks_for_batch produce identical results.

This module tests the equivalence between the optimized fast mask implementation and the reference
slow implementation across various entity structures and edge cases, particularly with the new
Kleene-plus grammar that allows multiple entities.
"""

from __future__ import annotations

import torch

from jsonlm.models.scoring import _build_masks_for_batch, _build_masks_for_batch_fast
from jsonlm.serialization.encoder import entity_to_string
from jsonlm.tokenization.trainer import train_tokenizer
from jsonlm.tokenization.vocab import Vocabulary


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


def test_fast_slow_mask_equivalence_basic():
    """Test that fast and slow mask implementations produce identical results for basic entities."""
    tok = _make_tokenizer()

    # Test various entity structures
    entities = [
        {"a": ["x"]},  # simple entity
        {"a": ["x", "y"]},  # multi-value
        {"a": ["x"], "b": ["y"]},  # multi-key
        {"title": ["Notes"], "tags": ["ai"]},  # different keys
        {},  # empty entity
    ]

    batch = _encode_entities_to_batch(entities, tok)

    # Generate masks with both implementations
    masks_slow = _build_masks_for_batch(batch, tok)
    masks_fast = _build_masks_for_batch_fast(batch, tok)

    # Should be identical
    assert torch.equal(masks_slow, masks_fast), (
        f"Mask implementations differ!\n"
        f"Shapes: slow={masks_slow.shape}, fast={masks_fast.shape}\n"
        f"First difference at: {torch.where(masks_slow != masks_fast)}"
    )


def test_fast_slow_mask_equivalence_variable_lengths():
    """Test equivalence with variable-length sequences (different padding)."""
    tok = _make_tokenizer()

    # Mix of short and long entities to test padding handling
    entities = [
        {"k": ["v"]},  # very short
        {"a": ["x", "y", "z"], "b": ["long", "values"]},  # longer
        {"title": ["A", "B"], "tags": ["ai", "ml", "nlp"]},  # longest
        {},  # shortest (empty)
    ]

    batch = _encode_entities_to_batch(entities, tok)

    masks_slow = _build_masks_for_batch(batch, tok)
    masks_fast = _build_masks_for_batch_fast(batch, tok)

    assert torch.equal(masks_slow, masks_fast), (
        f"Variable length mask implementations differ!\nBatch shape: {batch.shape}\nEntities: {entities}"
    )


def test_fast_slow_mask_equivalence_edge_cases():
    """Test equivalence for edge cases like empty entities and single tokens."""
    tok = _make_tokenizer()

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

    masks_slow = _build_masks_for_batch(batch, tok)
    masks_fast = _build_masks_for_batch_fast(batch, tok)

    assert torch.equal(masks_slow, masks_fast), (
        f"Edge case mask implementations differ!\nValid entities: {valid_entities}\nBatch shape: {batch.shape}"
    )


def test_fast_slow_mask_equivalence_single_entity():
    """Test equivalence for single entity (common case)."""
    tok = _make_tokenizer()

    entity = {"name": ["Alice", "Bob"], "role": ["admin"]}
    batch = _encode_entities_to_batch([entity], tok)

    masks_slow = _build_masks_for_batch(batch, tok)
    masks_fast = _build_masks_for_batch_fast(batch, tok)

    assert torch.equal(masks_slow, masks_fast), (
        f"Single entity mask implementations differ!\n"
        f"Entity: {entity}\n"
        f"Batch shape: {batch.shape}\n"
        f"Mask shapes: slow={masks_slow.shape}, fast={masks_fast.shape}"
    )


def test_fast_slow_mask_equivalence_batch_sizes():
    """Test equivalence across different batch sizes."""
    tok = _make_tokenizer()

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

        masks_slow = _build_masks_for_batch(batch, tok)
        masks_fast = _build_masks_for_batch_fast(batch, tok)

        assert torch.equal(masks_slow, masks_fast), (
            f"Batch size {batch_size} mask implementations differ!\n"
            f"Entities: {test_entities}\n"
            f"Batch shape: {batch.shape}"
        )


def test_fast_slow_mask_states_at_each_position():
    """Test that both implementations handle grammar state transitions identically."""
    tok = _make_tokenizer()

    # Entity that exercises various grammar states
    entity = {"key1": ["val1", "val2"], "key2": ["val3"]}
    batch = _encode_entities_to_batch([entity], tok)

    masks_slow = _build_masks_for_batch(batch, tok)
    masks_fast = _build_masks_for_batch_fast(batch, tok)

    # Should be identical
    assert torch.equal(masks_slow, masks_fast)

    # Additionally, check that the masks have the expected structure:
    # - At each position, only grammatically valid tokens should be allowed
    # - The number of allowed tokens should be reasonable (not all or none except at EOS)
    B, T, V = masks_slow.shape

    for b in range(B):
        for t in range(T):
            allowed_count = masks_slow[b, t].sum().item()
            # Sanity check: should have some allowed tokens (unless at end with EOS-only)
            assert allowed_count > 0, f"No tokens allowed at batch {b}, position {t}"
            assert allowed_count < V, f"All tokens allowed at batch {b}, position {t} (likely wrong)"


def test_fast_slow_mask_end_state_kleene_plus():
    """Test that both implementations correctly handle the Kleene-plus END state behavior."""
    tok = _make_tokenizer()

    # Test entity that reaches END state
    entity = {"test": ["value"]}
    batch = _encode_entities_to_batch([entity], tok)

    masks_slow = _build_masks_for_batch(batch, tok)
    masks_fast = _build_masks_for_batch_fast(batch, tok)

    assert torch.equal(masks_slow, masks_fast)

    # Specifically check the mask at the position where we'd be in END state
    # This would be after the closing '}' token, where we should allow both EOS and '{'
    lbrace_id = tok.vocabulary.token_id("{")
    eos_id = tok.vocabulary.eos_id

    # Find the position in the sequence where END state occurs
    # The exact position depends on the tokenization, but we can verify that
    # when EOS is allowed, '{' should also be allowed (Kleene-plus behavior)
    B, T, V = masks_slow.shape
    for b in range(B):
        for t in range(T):
            if masks_slow[b, t, eos_id]:  # If EOS is allowed
                # Then '{' should also be allowed (Kleene-plus)
                assert masks_slow[b, t, lbrace_id], (
                    f"At batch {b}, position {t}: EOS allowed but '{{' not allowed. "
                    f"This violates Kleene-plus grammar where both should be allowed at END state."
                )
