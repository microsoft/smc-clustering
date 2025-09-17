"""
Tests for sequence serialization functions in encoder module.

This module tests the new entities_to_string and parse_sequence functions that handle
multiple entities, ensuring they work correctly with the Kleene-plus grammar and maintain
backward compatibility with existing single-entity functions.
"""

from __future__ import annotations

import pytest

from jsonlm.serialization.encoder import entities_to_string, entity_to_string, parse_entity, parse_sequence


def test_entities_to_string_empty():
    """Test that empty entity list serializes to empty string."""
    result = entities_to_string([])
    assert result == ""


def test_entities_to_string_single_entity():
    """Test that single entity serializes identically to entity_to_string."""
    entity = {"a": ["x"]}

    single_result = entity_to_string(entity)
    sequence_result = entities_to_string([entity])

    assert sequence_result == single_result


def test_entities_to_string_multiple_entities():
    """Test serialization of multiple entities with space separation."""
    entities = [{"a": ["x"]}, {"b": ["y", "z"]}, {"c": ["p"], "d": ["q"]}]

    result = entities_to_string(entities)

    # Should be individual entity strings joined with spaces
    expected_parts = [entity_to_string(entity) for entity in entities]
    expected = " ".join(expected_parts)

    assert result == expected


def test_entities_to_string_with_empty_entity():
    """Test serialization including empty entities."""
    entities = [
        {"a": ["x"]},
        {},  # empty entity
        {"b": ["y"]},
    ]

    result = entities_to_string(entities)

    expected_parts = [entity_to_string(entity) for entity in entities]
    expected = " ".join(expected_parts)

    assert result == expected

    # Verify empty entity serializes as just "{ }"
    assert "{ }" in result


def test_parse_sequence_empty():
    """Test parsing empty string returns empty list."""
    assert parse_sequence("") == []
    assert parse_sequence("   ") == []  # whitespace only


def test_parse_sequence_single_entity():
    """Test that single entity parsing matches parse_entity result."""
    text = '{ <K> "test" : [ <V> "value" ] }'

    single_result = parse_entity(text)
    sequence_result = parse_sequence(text)

    assert len(sequence_result) == 1
    assert sequence_result[0] == single_result


def test_parse_sequence_multiple_entities():
    """Test parsing multiple entities from concatenated string."""
    entities = [{"a": ["x"]}, {"b": ["y", "z"]}, {"title": ["Notes"], "tags": ["ai", "ml"]}]

    # Serialize to string
    text = entities_to_string(entities)

    # Parse back
    parsed = parse_sequence(text)

    assert len(parsed) == len(entities)
    for original, parsed_entity in zip(entities, parsed, strict=False):
        assert parsed_entity == original


def test_parse_sequence_round_trip():
    """Test that serialization and parsing are inverse operations."""
    entities = [
        {"author": ["Ada", "Lovelace"]},
        {"field": ["computing"]},
        {},  # empty entity
        {"tags": ["ai", "history", "mathematics"], "year": ["1843"]},
    ]

    # Round trip: entities -> string -> entities
    serialized = entities_to_string(entities)
    parsed = parse_sequence(serialized)

    assert parsed == entities


def test_parse_sequence_with_whitespace_variations():
    """Test parsing handles various whitespace patterns correctly."""
    # Test with different spacing patterns
    texts = [
        '{ <K> "a" : [ <V> "x" ] }{ <K> "b" : [ <V> "y" ] }',  # no space between entities
        '{ <K> "a" : [ <V> "x" ] }  { <K> "b" : [ <V> "y" ] }',  # double space
        ' { <K> "a" : [ <V> "x" ] } { <K> "b" : [ <V> "y" ] } ',  # leading/trailing spaces
    ]

    expected = [{"a": ["x"]}, {"b": ["y"]}]

    for text in texts:
        result = parse_sequence(text)
        assert result == expected


def test_parse_sequence_complex_values():
    """Test parsing sequences with complex string values (quotes, escapes, etc)."""
    entities = [
        {"message": ['Hello "World"']},  # quotes
        {"path": ["C:\\Users\\test"]},  # backslashes
        {"unicode": ["🚀 rocket"]},  # unicode
        {"newlines": ["line1\nline2"]},  # newlines
    ]

    # Round trip test
    serialized = entities_to_string(entities)
    parsed = parse_sequence(serialized)

    assert parsed == entities


def test_parse_sequence_error_handling():
    """Test that parse_sequence properly handles malformed input."""
    malformed_inputs = [
        '{ <K> "incomplete"',  # incomplete entity
        '{ <K> "bad" : [ <V> "missing_close"',  # missing closing brackets
        '{ <K> "test" : [ <V> "value" ] } { invalid }',  # second entity malformed
        "not_json_at_all",  # not JSON-like at all
        '{ <K> "test" : [ <V> "value" ] } extra_tokens',  # trailing non-JSON tokens
    ]

    for bad_input in malformed_inputs:
        with pytest.raises(ValueError):
            parse_sequence(bad_input)


def test_parse_sequence_maintains_canonicalization():
    """Test that parsed entities are properly canonicalized."""
    # Create text with non-canonical ordering
    text = (
        '{ <K> "z" : [ <V> "last" ] , <K> "a" : [ <V> "first" ] } '
        '{ <K> "b" : [ <V> "second" , <V> "duplicate" , <V> "second" ] }'
    )

    parsed = parse_sequence(text)

    # Should have canonical key ordering and deduped values
    expected = [
        {"a": ["first"], "z": ["last"]},  # keys sorted
        {"b": ["duplicate", "second"]},  # values deduped and sorted
    ]

    assert parsed == expected


def test_entities_to_string_maintains_determinism():
    """Test that entities_to_string produces deterministic output."""
    entities = [
        {"z": ["last"], "a": ["first"]},  # non-canonical key order
        {"b": ["second", "first", "second"]},  # duplicate values
    ]

    result1 = entities_to_string(entities)
    result2 = entities_to_string(entities)

    # Should be identical (deterministic)
    assert result1 == result2

    # Should contain canonical ordering
    assert '"a"' in result1
    assert '"z"' in result1
    # 'a' should appear before 'z' due to canonicalization
    assert result1.find('"a"') < result1.find('"z"')


def test_backward_compatibility():
    """Test that new functions don't break existing single-entity functionality."""
    entity = {"test": ["value1", "value2"]}

    # Original functions should still work
    serialized_old = entity_to_string(entity)
    parsed_old = parse_entity(serialized_old)

    # New functions should be compatible
    serialized_new = entities_to_string([entity])
    parsed_new = parse_sequence(serialized_new)

    assert serialized_old == serialized_new
    assert parsed_old == entity
    assert len(parsed_new) == 1
    assert parsed_new[0] == parsed_old


def test_empty_and_mixed_sequences():
    """Test various edge cases with empty sequences and mixed content."""
    test_cases = [
        [],  # completely empty
        [{}],  # single empty entity
        [{}, {}],  # multiple empty entities
        [{"a": ["x"]}, {}, {"b": ["y"]}],  # mixed empty and non-empty
        [{"a": []}, {"b": ["y"]}],  # empty value list (gets canonicalized out)
    ]

    for entities in test_cases:
        # Should not raise errors
        serialized = entities_to_string(entities)
        parsed = parse_sequence(serialized)

        # Should be canonical
        assert parsed == entities
