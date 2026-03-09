"""Unit tests for the unified grammar runtime system.

Tests verify equivalence with the old per-step automaton approach, proper EOS handling,
correct shapes, and caching behavior across different devices and tokenizers.
"""

from __future__ import annotations

import torch

from smc_clustering.jsonlm.grammar.automaton import GrammarAutomaton, GrammarState
from smc_clustering.jsonlm.grammar.mask import allowed_token_mask
from smc_clustering.jsonlm.grammar.runtime import get_runtime
from smc_clustering.jsonlm.serialization.encoder import entity_to_string
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _build_masks_for_batch_slow(ids_with_eos: torch.Tensor, tokenizer) -> torch.BoolTensor:
    """Reference implementation: per-step automaton building (ported from old lit_module code)."""
    assert ids_with_eos.dim() == 2 and ids_with_eos.dtype == torch.long
    B, L = ids_with_eos.shape
    assert L >= 2
    V = len(tokenizer)
    T = L - 1
    masks = torch.zeros((B, T, V), dtype=torch.bool, device=ids_with_eos.device)
    eos = tokenizer.vocabulary.eos_id

    automaton = GrammarAutomaton(tokenizer)
    for b in range(B):
        seq = ids_with_eos[b]  # [L]
        assert seq[0].item() == tokenizer.vocabulary.bos_id, "Sequence must start with BOS"

        gs: GrammarState = automaton.start()
        for t in range(T):
            y_t = int(seq[t + 1].item())

            # Allowed next tokens before consuming y_t
            m = allowed_token_mask(gs, automaton, tokenizer)  # [V]
            masks[b, t] = m.to(device=ids_with_eos.device)

            # If y_t is EOS, allow EOS only for the rest of this row and stop stepping
            if y_t == eos:
                if t + 1 < T:
                    masks[b, t + 1 :, :] = False
                    masks[b, t + 1 :, eos] = True
                break

            # Consume non-EOS gold token in the grammar
            try:
                gs = automaton.step(gs, y_t)
            except ValueError as e:
                raise ValueError(f"Automaton reject at b={b}, t={t}, token_id={y_t}: {e}") from e

    return masks


def _make_test_tokenizer():
    """Create a test tokenizer with some vocabulary."""
    vocab = Vocabulary.from_default()
    # Minimal corpus to ensure BPE has some pieces
    entities = [
        {"a": ["x", "y"]},
        {"key1": ["value1", "value2"]},
        {"field": ["data", "more_data"]},
    ]
    corpus = [entity_to_string(e) for e in entities]
    return train_tokenizer(corpus, vocabulary=vocab, bpe_vocab_size=64)


def test_runtime_equivalence_with_slow_path():
    """Test that runtime.build_masks() equals the old per-step automaton approach."""
    tokenizer = _make_test_tokenizer()
    device = torch.device("cpu")

    # Create some test sequences of varying lengths
    entities = [
        {"a": ["x"]},
        {"key": ["val1", "val2"]},
        {"field": ["data"]},
        {"longer_key": ["value1", "value2", "value3"]},
    ]

    # Encode to BOS...EOS sequences
    ids_list = []
    for entity in entities:
        s = entity_to_string(entity)
        ids = tokenizer.encode(s, add_bos_eos=True)
        ids_list.append(ids)

    # Pad to tensor
    B = len(ids_list)
    L_max = max(len(ids) for ids in ids_list)
    batch = torch.full((B, L_max), tokenizer.vocabulary.pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(ids_list):
        L = len(ids)
        batch[i, :L] = torch.tensor(ids, dtype=torch.long)
        if L_max > L:
            batch[i, L:] = tokenizer.vocabulary.eos_id  # EOS-pad for clean masking

    # Compare runtime vs slow path
    rt = get_runtime(tokenizer, device)
    masks_fast = rt.build_masks(batch)
    masks_slow = _build_masks_for_batch_slow(batch, tokenizer)

    assert masks_fast.shape == masks_slow.shape
    assert torch.equal(masks_fast, masks_slow), "Runtime masks don't match slow automaton masks"


def test_post_eos_behavior():
    """Test that after first EOS in each row, mask allows EOS-only."""
    tokenizer = _make_test_tokenizer()
    device = torch.device("cpu")

    # Create a sequence where we manually control EOS placement
    entity = {"a": ["x"]}
    s = entity_to_string(entity)
    ids = tokenizer.encode(s, add_bos_eos=True)
    eos_id = tokenizer.vocabulary.eos_id

    # Create a batch with different EOS positions
    # Row 0: normal sequence
    # Row 1: early EOS (inject EOS in middle and pad with EOS)
    B, L = 2, len(ids) + 2  # extra length for testing
    batch = torch.full((B, L), eos_id, dtype=torch.long, device=device)

    # Row 0: normal sequence followed by EOS padding
    batch[0, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    # Row 1: BOS, then early EOS, then EOS padding
    batch[1, 0] = tokenizer.vocabulary.bos_id
    batch[1, 1] = eos_id  # early EOS

    rt = get_runtime(tokenizer, device)
    masks = rt.build_masks(batch)  # [B, T, V]

    V = len(tokenizer)
    T = L - 1

    # For row 1 (early EOS): after position 0 (which predicts the early EOS),
    # all subsequent positions should be EOS-only
    for t in range(1, T):
        expected_mask = torch.zeros(V, dtype=torch.bool, device=device)
        expected_mask[eos_id] = True
        assert torch.equal(masks[1, t], expected_mask), (
            f"Row 1, timestep {t} should be EOS-only after early EOS"
        )


def test_shapes_match_model_outputs():
    """Test that output shapes are [B, T, V] where T = L-1."""
    tokenizer = _make_test_tokenizer()
    device = torch.device("cpu")

    entities = [{"a": ["x"]}, {"key": ["val1", "val2"]}]
    ids_list = [tokenizer.encode(entity_to_string(e), add_bos_eos=True) for e in entities]

    B = len(ids_list)
    L_max = max(len(ids) for ids in ids_list)
    V = len(tokenizer)
    T = L_max - 1

    batch = torch.full((B, L_max), tokenizer.vocabulary.eos_id, dtype=torch.long, device=device)
    for i, ids in enumerate(ids_list):
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    rt = get_runtime(tokenizer, device)
    masks = rt.build_masks(batch)

    assert masks.shape == (B, T, V), f"Expected shape ({B}, {T}, {V}), got {masks.shape}"
    assert masks.dtype == torch.bool, f"Expected bool dtype, got {masks.dtype}"


def test_cache_behavior():
    """Test that repeated get_runtime calls return the same object, different devices return different objects."""
    tokenizer = _make_test_tokenizer()
    device_cpu = torch.device("cpu")

    # Same tokenizer + device should return cached instance
    rt1 = get_runtime(tokenizer, device_cpu)
    rt2 = get_runtime(tokenizer, device_cpu)
    assert rt1 is rt2, "Same tokenizer + device should return cached runtime"

    # Different device should return different instance
    if torch.cuda.is_available():
        device_cuda = torch.device("cuda:0")
        rt3 = get_runtime(tokenizer, device_cuda)
        assert rt1 is not rt3, "Different device should return different runtime"
        assert rt1.allowed.device != rt3.allowed.device, "Runtimes should be on different devices"


def test_equivalence_with_different_lengths():
    """Test equivalence on sequences of multiple lengths."""
    tokenizer = _make_test_tokenizer()
    device = torch.device("cpu")

    # Test with single sequences of different lengths
    entities = [
        {"a": ["x"]},  # short
        {"key": ["val1", "val2", "val3", "val4"]},  # long
        {"": []},  # edge case: empty values
    ]

    for entity in entities:
        s = entity_to_string(entity)
        ids = tokenizer.encode(s, add_bos_eos=True)
        batch = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]

        rt = get_runtime(tokenizer, device)
        masks_fast = rt.build_masks(batch)
        masks_slow = _build_masks_for_batch_slow(batch, tokenizer)

        assert torch.equal(masks_fast, masks_slow), (
            f"Mismatch for entity {entity}: fast={masks_fast.shape}, slow={masks_slow.shape}"
        )


def test_cuda_if_available():
    """Test CUDA functionality if available."""
    if not torch.cuda.is_available():
        return

    tokenizer = _make_test_tokenizer()
    device = torch.device("cuda:0")

    entity = {"test": ["value"]}
    s = entity_to_string(entity)
    ids = tokenizer.encode(s, add_bos_eos=True)
    batch = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    rt = get_runtime(tokenizer, device)
    masks = rt.build_masks(batch)

    assert masks.device == device, f"Expected masks on {device}, got {masks.device}"
    assert masks.dtype == torch.bool, f"Expected bool dtype, got {masks.dtype}"
