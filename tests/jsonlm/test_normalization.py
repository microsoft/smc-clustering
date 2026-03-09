"""Unit tests for jsonlm.serialization.normalization module.

Tests comprehensive behavior of properties wrapper removal functions, including
entity/sequence handling, strict/lenient modes, and error conditions as specified
in the task requirements.
"""

import pytest

from smc_clustering.jsonlm.serialization.normalization import (
    normalize_entity_or_sequence,
    unwrap_properties_entity,
    unwrap_properties_sequence,
)


class TestUnwrapPropertiesEntity:
    """Test unwrap_properties_entity function."""

    def test_raw_dict_no_properties(self) -> None:
        """Entity without properties wrapper is returned unchanged."""
        entity = {"name": "Alice", "age": 30}
        result = unwrap_properties_entity(entity)
        assert result == {"name": "Alice", "age": 30}

    def test_wrapped_dict_with_properties(self) -> None:
        """Entity with properties wrapper is unwrapped correctly."""
        wrapped = {"properties": {"name": "Bob", "city": "NYC"}}
        result = unwrap_properties_entity(wrapped)
        assert result == {"name": "Bob", "city": "NYC"}

    def test_empty_dict(self) -> None:
        """Empty dictionary is handled correctly."""
        empty = {}
        result = unwrap_properties_entity(empty)
        assert result == {}

    def test_empty_properties_dict(self) -> None:
        """Dictionary with empty properties is unwrapped to empty dict."""
        wrapped = {"properties": {}}
        result = unwrap_properties_entity(wrapped)
        assert result == {}

    def test_dict_with_other_keys_no_properties(self) -> None:
        """Dictionary with other keys but no properties is returned unchanged."""
        entity = {"metadata": {"source": "test"}, "data": {"value": 42}}
        result = unwrap_properties_entity(entity)
        assert result == entity

    def test_dict_with_properties_and_other_keys(self) -> None:
        """Dictionary with properties and other keys extracts only properties."""
        mixed = {
            "properties": {"name": "Charlie", "age": 25},
            "metadata": {"timestamp": "2023-01-01"},
        }
        result = unwrap_properties_entity(mixed)
        assert result == {"name": "Charlie", "age": 25}

    def test_invalid_type_raises_error(self) -> None:
        """Non-dict input raises TypeError."""
        with pytest.raises(TypeError, match="Expected dict, got"):
            unwrap_properties_entity("not a dict")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Expected dict, got"):
            unwrap_properties_entity([{"name": "Alice"}])  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Expected dict, got"):
            unwrap_properties_entity(42)  # type: ignore[arg-type]


class TestUnwrapPropertiesSequence:
    """Test unwrap_properties_sequence function."""

    def test_empty_list(self) -> None:
        """Empty list is returned unchanged."""
        empty: list[dict[str, object]] = []
        result = unwrap_properties_sequence(empty, mode="strict")
        assert result == []

        result = unwrap_properties_sequence(empty, mode="lenient")
        assert result == []

    def test_strict_mode_all_wrapped(self) -> None:
        """Strict mode: all items have properties, all are unwrapped."""
        wrapped_seq = [
            {"properties": {"name": "Alice", "age": 30}},
            {"properties": {"name": "Bob", "age": 25}},
        ]
        result = unwrap_properties_sequence(wrapped_seq, mode="strict")
        expected = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        assert result == expected

    def test_strict_mode_none_wrapped(self) -> None:
        """Strict mode: no items have properties, all returned unchanged."""
        raw_seq = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        result = unwrap_properties_sequence(raw_seq, mode="strict")
        assert result == raw_seq

    def test_strict_mode_mixed_properties_raises_error(self) -> None:
        """Strict mode: mixed properties presence raises ValueError."""
        mixed_seq = [
            {"properties": {"name": "Alice", "age": 30}},
            {"name": "Bob", "age": 25},  # No properties wrapper
        ]
        with pytest.raises(
            ValueError, match="if first item has 'properties', all items must have 'properties'"
        ):
            unwrap_properties_sequence(mixed_seq, mode="strict")

    def test_strict_mode_mixed_properties_reverse_raises_error(self) -> None:
        """Strict mode: first without properties, but later with properties raises error."""
        mixed_seq = [
            {"name": "Alice", "age": 30},  # No properties wrapper
            {"properties": {"name": "Bob", "age": 25}},
        ]
        with pytest.raises(
            ValueError, match="if first item lacks 'properties', no items should have 'properties'"
        ):
            unwrap_properties_sequence(mixed_seq, mode="strict")

    def test_lenient_mode_all_wrapped(self) -> None:
        """Lenient mode: all items have properties, all are unwrapped."""
        wrapped_seq = [
            {"properties": {"name": "Alice", "age": 30}},
            {"properties": {"name": "Bob", "age": 25}},
        ]
        result = unwrap_properties_sequence(wrapped_seq, mode="lenient")
        expected = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        assert result == expected

    def test_lenient_mode_none_wrapped(self) -> None:
        """Lenient mode: no items have properties, all returned unchanged."""
        raw_seq = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        result = unwrap_properties_sequence(raw_seq, mode="lenient")
        assert result == raw_seq

    def test_lenient_mode_mixed_properties(self) -> None:
        """Lenient mode: mixed properties presence handled gracefully."""
        mixed_seq = [
            {"properties": {"name": "Alice", "age": 30}},
            {"name": "Bob", "age": 25},  # No properties wrapper
            {"properties": {"name": "Charlie", "age": 35}},
        ]
        result = unwrap_properties_sequence(mixed_seq, mode="lenient")
        expected = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        assert result == expected

    def test_single_item_with_properties(self) -> None:
        """Single item with properties is unwrapped correctly."""
        single_wrapped = [{"properties": {"name": "Alice", "age": 30}}]
        result = unwrap_properties_sequence(single_wrapped, mode="strict")
        assert result == [{"name": "Alice", "age": 30}]

        result = unwrap_properties_sequence(single_wrapped, mode="lenient")
        assert result == [{"name": "Alice", "age": 30}]

    def test_single_item_without_properties(self) -> None:
        """Single item without properties is returned unchanged."""
        single_raw = [{"name": "Alice", "age": 30}]
        result = unwrap_properties_sequence(single_raw, mode="strict")
        assert result == single_raw

        result = unwrap_properties_sequence(single_raw, mode="lenient")
        assert result == single_raw

    def test_invalid_list_type_raises_error(self) -> None:
        """Non-list input raises TypeError."""
        with pytest.raises(TypeError, match="Expected list, got"):
            unwrap_properties_sequence({"name": "Alice"}, mode="strict")  # type: ignore[arg-type]

    def test_non_dict_items_raise_error(self) -> None:
        """List containing non-dict items raises TypeError."""
        invalid_seq = [{"name": "Alice"}, "not a dict"]  # type: ignore[list-item]
        with pytest.raises(TypeError, match="All items in sequence must be dictionaries"):
            unwrap_properties_sequence(invalid_seq, mode="strict")

        with pytest.raises(TypeError, match="All items in sequence must be dictionaries"):
            unwrap_properties_sequence(invalid_seq, mode="lenient")

    def test_invalid_mode_raises_error(self) -> None:
        """Invalid mode raises ValueError."""
        seq = [{"name": "Alice"}]
        with pytest.raises(ValueError, match="Invalid mode: 'invalid'"):
            unwrap_properties_sequence(seq, mode="invalid")  # type: ignore[arg-type]

    def test_default_mode_is_strict(self) -> None:
        """Default mode is strict."""
        mixed_seq = [
            {"properties": {"name": "Alice", "age": 30}},
            {"name": "Bob", "age": 25},  # No properties wrapper
        ]
        with pytest.raises(
            ValueError, match="if first item has 'properties', all items must have 'properties'"
        ):
            unwrap_properties_sequence(mixed_seq)  # No mode specified, should default to strict


class TestNormalizeEntityOrSequence:
    """Test normalize_entity_or_sequence function."""

    def test_normalize_single_entity_no_properties(self) -> None:
        """Single entity without properties is returned unchanged."""
        entity = {"name": "Alice", "age": 30}
        result = normalize_entity_or_sequence(entity, seq_mode="strict")
        assert result == entity

    def test_normalize_single_entity_with_properties(self) -> None:
        """Single entity with properties is unwrapped."""
        wrapped = {"properties": {"name": "Bob", "city": "NYC"}}
        result = normalize_entity_or_sequence(wrapped, seq_mode="strict")
        assert result == {"name": "Bob", "city": "NYC"}

    def test_normalize_sequence_strict_all_wrapped(self) -> None:
        """Sequence with all properties wrappers is unwrapped in strict mode."""
        wrapped_seq = [
            {"properties": {"name": "Alice", "age": 30}},
            {"properties": {"name": "Bob", "age": 25}},
        ]
        result = normalize_entity_or_sequence(wrapped_seq, seq_mode="strict")
        expected = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        assert result == expected

    def test_normalize_sequence_strict_none_wrapped(self) -> None:
        """Sequence without properties wrappers is returned unchanged in strict mode."""
        raw_seq = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        result = normalize_entity_or_sequence(raw_seq, seq_mode="strict")
        assert result == raw_seq

    def test_normalize_sequence_lenient_mixed(self) -> None:
        """Sequence with mixed properties presence is handled in lenient mode."""
        mixed_seq = [
            {"properties": {"name": "Alice", "age": 30}},
            {"name": "Bob", "age": 25},  # No properties wrapper
        ]
        result = normalize_entity_or_sequence(mixed_seq, seq_mode="lenient")
        expected = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        assert result == expected

    def test_normalize_sequence_strict_mixed_raises_error(self) -> None:
        """Sequence with mixed properties presence raises error in strict mode."""
        mixed_seq = [
            {"properties": {"name": "Alice", "age": 30}},
            {"name": "Bob", "age": 25},  # No properties wrapper
        ]
        with pytest.raises(
            ValueError, match="if first item has 'properties', all items must have 'properties'"
        ):
            normalize_entity_or_sequence(mixed_seq, seq_mode="strict")

    def test_invalid_input_type_raises_error(self) -> None:
        """Invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="Expected dict or list, got"):
            normalize_entity_or_sequence("not valid", seq_mode="strict")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Expected dict or list, got"):
            normalize_entity_or_sequence(42, seq_mode="strict")  # type: ignore[arg-type]

    def test_default_seq_mode_is_strict(self) -> None:
        """Default seq_mode is strict."""
        mixed_seq = [
            {"properties": {"name": "Alice", "age": 30}},
            {"name": "Bob", "age": 25},  # No properties wrapper
        ]
        with pytest.raises(
            ValueError, match="if first item has 'properties', all items must have 'properties'"
        ):
            normalize_entity_or_sequence(mixed_seq)  # No seq_mode specified, should default to strict

    def test_seq_mode_only_affects_sequences(self) -> None:
        """seq_mode parameter only affects sequences, not single entities."""
        wrapped_entity = {"properties": {"name": "Alice", "age": 30}}

        result_strict = normalize_entity_or_sequence(wrapped_entity, seq_mode="strict")
        result_lenient = normalize_entity_or_sequence(wrapped_entity, seq_mode="lenient")

        expected = {"name": "Alice", "age": 30}
        assert result_strict == expected
        assert result_lenient == expected


class TestReadJsonlEntitiesScenarios:
    """Test scenarios mentioned in task requirements for read_jsonl_entities behavior."""

    def test_dict_line_scenarios(self) -> None:
        """Test single entity scenarios (dict lines)."""
        # Raw dict
        raw_entity = {"name": "Alice", "age": 30}
        result = normalize_entity_or_sequence(raw_entity)
        assert result == raw_entity

        # Wrapped dict
        wrapped_entity = {"properties": {"name": "Bob", "city": "NYC"}}
        result = normalize_entity_or_sequence(wrapped_entity)
        assert result == {"name": "Bob", "city": "NYC"}

    def test_list_line_multiple_yields_scenarios(self) -> None:
        """Test sequence scenarios (list lines with multiple entities)."""
        # All wrapped
        all_wrapped = [
            {"properties": {"name": "Alice", "age": 30}},
            {"properties": {"name": "Bob", "age": 25}},
        ]
        result = normalize_entity_or_sequence(all_wrapped, seq_mode="strict")
        expected = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        assert result == expected

        # None wrapped
        none_wrapped = [
            {"name": "Charlie", "age": 35},
            {"name": "David", "age": 40},
        ]
        result = normalize_entity_or_sequence(none_wrapped, seq_mode="strict")
        assert result == none_wrapped

        # Mixed (lenient mode only)
        mixed = [
            {"properties": {"name": "Eve", "age": 28}},
            {"name": "Frank", "age": 32},
        ]
        result = normalize_entity_or_sequence(mixed, seq_mode="lenient")
        expected = [
            {"name": "Eve", "age": 28},
            {"name": "Frank", "age": 32},
        ]
        assert result == expected

        # Mixed in strict mode should raise error
        with pytest.raises(ValueError):
            normalize_entity_or_sequence(mixed, seq_mode="strict")

    def test_wrapped_forms_handling(self) -> None:
        """Test various wrapped forms are handled correctly."""
        # Nested properties
        nested = {"properties": {"user": {"name": "Alice"}, "metadata": {"id": 123}}}
        result = normalize_entity_or_sequence(nested)
        assert result == {"user": {"name": "Alice"}, "metadata": {"id": 123}}

        # Properties with mixed types
        mixed_types = {
            "properties": {"name": "Bob", "age": 30, "active": True, "tags": ["user", "admin"]}
        }
        result = normalize_entity_or_sequence(mixed_types)
        assert result == {"name": "Bob", "age": 30, "active": True, "tags": ["user", "admin"]}

        # Sequence with complex properties
        complex_seq = [
            {"properties": {"id": 1, "data": {"nested": "value1"}}},
            {"properties": {"id": 2, "data": {"nested": "value2"}}},
        ]
        result = normalize_entity_or_sequence(complex_seq, seq_mode="strict")
        expected = [
            {"id": 1, "data": {"nested": "value1"}},
            {"id": 2, "data": {"nested": "value2"}},
        ]
        assert result == expected
