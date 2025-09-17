"""
Unit tests for canonicalization and sentinel-based serialization/parsing.

The tests assert idempotent canonicalization, deterministic ordering, round-trip stability, and helpful error handling
for invalid input shapes. The serialized form is whitespace-stable and parses back to a canonical dict.
"""

from __future__ import annotations

import pytest

from jsonlm.serialization.encoder import canonicalize_entity, entity_to_string, parse_entity


def test_canonicalization_idempotent() -> None:
    """Applying canonicalization twice yields the same mapping."""
    e = {"b": ["y", "x", "x"], "a": ["b", "a"]}
    c1 = canonicalize_entity(e)
    c2 = canonicalize_entity(c1)
    assert c1 == c2
    # Keys sorted, values deduped and sorted.
    assert list(c1.keys()) == ["a", "b"]
    assert c1["a"] == ["a", "b"]
    assert c1["b"] == ["x", "y"]


def test_serialization_deterministic_and_sorted() -> None:
    """Serialization should present keys/values sorted and with deterministic punctuation and sentinels."""
    e = {"b": ["y", "x", "x"], "a": ["b", "a"]}
    s = entity_to_string(e)
    # Expected exact token order (single spaces between tokens).
    expect = '{ <K> "a" : [ <V> "a" , <V> "b" ] , <K> "b" : [ <V> "x" , <V> "y" ] }'
    assert s == expect


def test_round_trip_parse_equals_canonical() -> None:
    """parse(entity_to_string(e)) equals canonicalize_entity(e)."""
    e = {"b": ["y", "x", "x"], "a": ["b", "a"]}
    s = entity_to_string(e)
    parsed = parse_entity(s)
    assert parsed == canonicalize_entity(e)


def test_empty_object_round_trip() -> None:
    """Empty entities serialize as '{}' with spaces and parse back to empty dict."""
    e = {}
    s = entity_to_string(e)
    assert s == "{ }"
    parsed = parse_entity(s)
    assert parsed == {}


def test_values_can_be_empty_list() -> None:
    """Keys with empty value lists serialize with empty brackets and parse back."""
    e = {"k": []}
    s = entity_to_string(e)
    assert s == '{ <K> "k" : [ ] }'
    parsed = parse_entity(s)
    assert parsed == {"k": []}


def test_bad_input_types_raise() -> None:
    """Non-list values or non-string items should raise ValueError."""
    with pytest.raises(ValueError):
        _ = canonicalize_entity({"a": "not-a-list"})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _ = canonicalize_entity({"a": [1, 2]})  # type: ignore[list-item]
    with pytest.raises(ValueError):
        _ = canonicalize_entity({1: ["x"]})  # type: ignore[dict-item]


def test_parser_rejects_misaligned_sentinels() -> None:
    """Parser should reject inputs that violate <K>/<V> placements."""
    # Missing <K>
    with pytest.raises(ValueError):
        _ = parse_entity('{ "a" : [ <V> "x" ] }')
    # Missing <V>
    with pytest.raises(ValueError):
        _ = parse_entity('{ <K> "a" : [ "x" ] }')
    # Unterminated string
    with pytest.raises(ValueError):
        _ = parse_entity('{ <K> "a : [ <V> "x" ] }')


def test_parser_merges_duplicate_keys_then_canonicalizes() -> None:
    """Duplicate keys merge values and canonicalize via set-semantics."""
    s = '{ <K> "a" : [ <V> "x" ] , <K> "a" : [ <V> "y" , <V> "x" ] }'
    parsed = parse_entity(s)
    assert parsed == {"a": ["x", "y"]}
