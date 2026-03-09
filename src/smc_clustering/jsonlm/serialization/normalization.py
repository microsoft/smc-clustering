"""Entity and sequence normalization utilities.

Provides centralized handling of legacy "properties" wrapper removal and consistent
sequence processing policies. Ensures deterministic behavior across dataset loading,
training, and evaluation workflows.
"""

from typing import Any, Literal


def unwrap_properties_entity(obj: dict[str, Any]) -> dict[str, Any]:
    """Remove legacy "properties" wrapper from a single entity if present.

    Args:
        obj: A dictionary that may contain a "properties" wrapper.

    Returns:
        The unwrapped entity dictionary.

    Raises:
        TypeError: If obj is not a dictionary.
    """
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict, got {type(obj).__name__}")

    if "properties" in obj:
        return obj["properties"]

    return obj


def unwrap_properties_sequence(
    obj: list[dict[str, Any]],
    mode: Literal["strict", "lenient"] = "strict",
) -> list[dict[str, Any]]:
    """Remove legacy "properties" wrapper from entity sequences.

    Args:
        obj: A list of dictionaries that may contain "properties" wrappers.
        mode: Processing mode:
            - "strict": All items must have properties OR none have properties.
                       If first item has properties, all items must have them.
            - "lenient": Items may have mixed properties presence.
                        Extract properties where present, leave others unchanged.

    Returns:
        List of unwrapped entity dictionaries.

    Raises:
        TypeError: If obj is not a list or contains non-dict items.
        ValueError: In strict mode, if properties presence is inconsistent.
    """
    if not isinstance(obj, list):
        raise TypeError(f"Expected list, got {type(obj).__name__}")

    if not obj:
        return obj

    if not all(isinstance(item, dict) for item in obj):
        raise TypeError("All items in sequence must be dictionaries")

    if mode == "strict":
        # Check first item to determine if we expect properties on all
        if "properties" in obj[0]:
            # All items must have properties in strict mode
            if not all("properties" in item for item in obj):
                raise ValueError("In strict mode, if first item has 'properties', all items must have 'properties'")
            return [item["properties"] for item in obj]
        # No items should have properties in strict mode
        if any("properties" in item for item in obj):
            raise ValueError("In strict mode, if first item lacks 'properties', no items should have 'properties'")
        return obj

    if mode == "lenient":
        # Extract properties where present, leave others unchanged
        return [item.get("properties", item) for item in obj]

    raise ValueError(f"Invalid mode: {mode!r}. Must be 'strict' or 'lenient'")


def normalize_entity_or_sequence(
    obj: dict[str, Any] | list[dict[str, Any]],
    seq_mode: Literal["strict", "lenient"] = "strict",
) -> dict[str, Any] | list[dict[str, Any]]:
    """Normalize entities or sequences by removing legacy "properties" wrappers.

    This is the main entry point for normalizing JSON objects that may be either
    single entities (dict) or entity sequences (list of dicts).

    Args:
        obj: A dictionary (single entity) or list of dictionaries (entity sequence).
        seq_mode: Mode for sequence processing (see unwrap_properties_sequence).

    Returns:
        Normalized entity or sequence with properties wrappers removed.

    Raises:
        TypeError: If obj is neither dict nor list, or if list contains non-dicts.
        ValueError: In strict seq_mode, if properties presence is inconsistent.
    """
    if isinstance(obj, dict):
        return unwrap_properties_entity(obj)
    if isinstance(obj, list):
        return unwrap_properties_sequence(obj, mode=seq_mode)
    raise TypeError(f"Expected dict or list, got {type(obj).__name__}")
