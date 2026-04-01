# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Tests for EntityDataset: offset indexing, on-the-fly canonicalization, and encoding.

We verify that dataset items are 1-D LongTensors containing BOS/EOS by default, and that two permutations of the same
entity yield identical ID sequences thanks to canonicalization inside serialization. Also tests support for entity
sequences (list of dicts) using Kleene-plus grammar with entities_to_string serialization.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from smc_clustering.jsonlm.data.dataset import EntityDataset
from smc_clustering.jsonlm.serialization.encoder import entities_to_string_as_set, entity_to_string
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _make_jsonl(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def test_dataset_item_tensor_and_bos_eos() -> None:
    """EntityDataset returns 1-D LongTensors of IDs including BOS/EOS by default."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    path = Path(path)
    try:
        # Small corpus and tokenizer.
        entities = [
            {"author": ["Ada", "Lovelace"], "tags": ["ai", "ml"]},
            {"k": ["v"]},
        ]
        corpus = [entity_to_string(e) for e in entities]
        tok = train_tokenizer(corpus, vocabulary=Vocabulary.from_default(), bpe_vocab_size=64)

        _make_jsonl(path, ['{"author": ["Ada", "Lovelace"], "tags": ["ai", "ml"]}', '{"k": ["v"]}'])
        ds = EntityDataset([str(path)], tokenizer=tok, add_bos_eos=True)

        x0 = ds[0]
        assert isinstance(x0, torch.Tensor)
        assert x0.dtype == torch.long and x0.dim() == 1
        # BOS/EOS present.
        assert x0[0].item() == tok.vocabulary.bos_id
        assert x0[-1].item() == tok.vocabulary.eos_id
    finally:
        path.unlink()


def test_dataset_canonicalization_invariance() -> None:
    """Two differently ordered input dicts should yield identical encoded sequences."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    path = Path(path)
    try:
        # Two permutations of the same logical entity.
        l1 = '{"b": ["y", "x", "x"], "a": ["b", "a"]}'
        l2 = '{"a": ["b", "a"], "b": ["y", "x"]}'
        # Tokenizer trained on their canonical serializations.
        corpus = [entity_to_string({"b": ["y", "x", "x"], "a": ["b", "a"]})]
        tok = train_tokenizer(corpus, vocabulary=Vocabulary.from_default(), bpe_vocab_size=64)

        _make_jsonl(path, [l1, l2])
        ds = EntityDataset([str(path)], tokenizer=tok, add_bos_eos=True)

        x1 = ds[0].tolist()
        x2 = ds[1].tolist()
        assert x1 == x2
    finally:
        path.unlink()


def test_dataset_handles_entity_sequences() -> None:
    """EntityDataset should handle lists of entities and serialize them as sequences."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    path = Path(path)
    try:
        # Prepare entities and sequences
        entity1 = {"a": ["x"]}
        entity2 = {"b": ["y", "z"]}
        sequence = [entity1, entity2]

        # Create corpus for tokenizer training
        corpus = [
            entity_to_string(entity1),
            entity_to_string(entity2),
            entities_to_string_as_set(sequence),
        ]
        tok = train_tokenizer(corpus, vocabulary=Vocabulary.from_default(), bpe_vocab_size=64)

        # Create JSONL with both single entities and sequences
        _make_jsonl(
            path,
            [
                '{"a": ["x"]}',  # single entity
                '[{"a": ["x"]}, {"b": ["y", "z"]}]',  # entity sequence
            ],
        )
        ds = EntityDataset([str(path)], tokenizer=tok, add_bos_eos=True)

        # Test single entity
        x0 = ds[0]
        assert isinstance(x0, torch.Tensor)
        assert x0.dtype == torch.long and x0.dim() == 1
        assert x0[0].item() == tok.vocabulary.bos_id
        assert x0[-1].item() == tok.vocabulary.eos_id

        # Test entity sequence
        x1 = ds[1]
        assert isinstance(x1, torch.Tensor)
        assert x1.dtype == torch.long and x1.dim() == 1
        assert x1[0].item() == tok.vocabulary.bos_id
        assert x1[-1].item() == tok.vocabulary.eos_id

        # Sequence should be longer than single entity
        assert len(x1) > len(x0)

    finally:
        path.unlink()


def test_dataset_sequence_vs_manual_serialization() -> None:
    """Sequence from dataset should match manual entities_to_string serialization."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    path = Path(path)
    try:
        # Prepare a sequence
        entities = [{"author": ["Ada"]}, {"field": ["computing"]}, {"tags": ["ai", "ml"]}]

        # Create corpus for tokenizer
        corpus = [entities_to_string_as_set(entities)] + [entity_to_string(e) for e in entities]
        tok = train_tokenizer(corpus, vocabulary=Vocabulary.from_default(), bpe_vocab_size=64)

        # Create JSONL with sequence
        _make_jsonl(path, [json.dumps(entities)])
        ds = EntityDataset([str(path)], tokenizer=tok, add_bos_eos=True)

        # Get sequence from dataset
        dataset_ids = ds[0]

        # Manually serialize and encode the same sequence
        manual_text = entities_to_string_as_set(entities)
        manual_ids = torch.tensor(tok.encode(manual_text, add_bos_eos=True), dtype=torch.long)

        # Should be identical
        assert torch.equal(dataset_ids, manual_ids)

    finally:
        path.unlink()


def test_dataset_empty_sequence_handling() -> None:
    """Dataset should handle empty sequences correctly."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    path = Path(path)
    try:
        # Create tokenizer
        corpus = [entity_to_string({"a": ["x"]})]
        tok = train_tokenizer(corpus, vocabulary=Vocabulary.from_default(), bpe_vocab_size=64)

        # Create JSONL with empty sequence
        _make_jsonl(path, ["[]"])  # empty list
        ds = EntityDataset([str(path)], tokenizer=tok, add_bos_eos=True)

        # Should handle empty sequence
        x0 = ds[0]
        assert isinstance(x0, torch.Tensor)
        assert x0.dtype == torch.long and x0.dim() == 1
        # Empty sequence should just be BOS + EOS
        assert len(x0) == 2  # Just BOS and EOS
        assert x0[0].item() == tok.vocabulary.bos_id
        assert x0[-1].item() == tok.vocabulary.eos_id

    finally:
        path.unlink()


def test_dataset_mixed_single_and_sequence_lines() -> None:
    """Dataset should handle mixed single entities and sequences in same file."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    path = Path(path)
    try:
        # Prepare mixed content
        single_entity = {"name": ["Alice"]}
        sequence = [{"role": ["admin"]}, {"team": ["engineering"]}]

        corpus = [entity_to_string(single_entity), entities_to_string_as_set(sequence)]
        tok = train_tokenizer(corpus, vocabulary=Vocabulary.from_default(), bpe_vocab_size=64)

        # Create JSONL with mixed content
        _make_jsonl(
            path,
            [
                json.dumps(single_entity),
                json.dumps(sequence),
                json.dumps(single_entity),  # another single
            ],
        )
        ds = EntityDataset([str(path)], tokenizer=tok, add_bos_eos=True)

        assert len(ds) == 3

        # All should be valid tensors
        for i in range(len(ds)):
            x = ds[i]
            assert isinstance(x, torch.Tensor)
            assert x.dtype == torch.long and x.dim() == 1
            assert x[0].item() == tok.vocabulary.bos_id
            assert x[-1].item() == tok.vocabulary.eos_id

        # Single entities should have same encoding
        x0_ids = ds[0].tolist()
        x2_ids = ds[2].tolist()
        assert x0_ids == x2_ids

        # Sequence should be different
        x1_ids = ds[1].tolist()
        assert x1_ids != x0_ids

    finally:
        path.unlink()


def test_dataset_sequence_error_handling() -> None:
    """Dataset should properly handle malformed sequences."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    path = Path(path)
    try:
        corpus = [entity_to_string({"a": ["x"]})]
        tok = train_tokenizer(corpus, vocabulary=Vocabulary.from_default(), bpe_vocab_size=64)

        # Test list with non-dict elements
        _make_jsonl(path, ['[{"a": ["x"]}, "not_a_dict"]'])
        ds = EntityDataset([str(path)], tokenizer=tok, add_bos_eos=True)

        with pytest.raises(TypeError, match="List items must all be dicts"):
            _ = ds[0]

    finally:
        path.unlink()
