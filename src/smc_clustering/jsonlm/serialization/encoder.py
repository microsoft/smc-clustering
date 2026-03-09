"""Canonicalization and textual serialization/parsing for JSON-entity dicts.

This module converts raw Python entities of type dict[str, list[str]] into a deterministic, canonical form and a stable
training string that includes structural sentinels (<K>, <V>) alongside JSON punctuation. Parsing performs strict
validation and returns a canonicalized dict. Determinism here makes downstream model scores identifiable and comparable.
"""

from __future__ import annotations

import json


# Each lexed token is a tuple (TYPE, VALUE). Only STRING carries a meaningful VALUE; others use the lexeme as TYPE.
Token = tuple[str, str | None]


def canonicalize_entity(entity: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return a canonicalized copy of an entity (sort keys; sort & dedup string values).

    Validates that:
      * entity is a dict[str, list[str]]
      * all keys are str
      * all values are lists of str

    The returned dict has lexicographically sorted keys, and each value list is de-duplicated and lexicographically
    sorted. This ensures a unique representative for set-semantics MVP.

    Args:
        entity: Input mapping from keys to list-of-string values.

    Returns:
        A new canonicalized mapping.

    Raises:
        ValueError: If types are invalid (non-dict, non-list values, non-string items).
    """
    if not isinstance(entity, dict):
        raise ValueError(f"Entity must be a dict[str, list[str]], got: {type(entity).__name__}")

    norm: dict[str, list[str]] = {}
    for k, vs in entity.items():
        # Validate key and value types.
        if not isinstance(k, str):
            raise ValueError(f"Entity keys must be str, got key type: {type(k).__name__}")
        if not isinstance(vs, list):
            raise ValueError(f"Entity values must be lists, got type for key '{k}': {type(vs).__name__}")
        bad_items = [type(x).__name__ for x in vs if not isinstance(x, str)]
        if bad_items:
            raise ValueError(f"All list items must be str for key '{k}', bad types: {bad_items}")

        # Set semantics MVP: dedup + sort.
        unique_sorted = sorted(set(vs))
        norm[k] = unique_sorted

    # Sort keys lexicographically for canonical ordering.
    out = {k: norm[k] for k in sorted(norm.keys())}
    return out


def entity_to_string(entity: dict[str, list[str]]) -> str:
    """Serialize an entity to a stable training string with <K>/<V> sentinels and JSON punctuation.

    The emitted format is deterministic and whitespace-stable; tokens are separated by single spaces.

    Example:
        { <K> "author" : [ <V> "Ada" , <V> "Lovelace" ] , <K> "tags" : [ <V> "ai" , <V> "ml" ] }

    This is not standard JSON due to <K>/<V> markers. Quoted strings are emitted as a single token (with quotes) and use
    JSON escaping rules.

    Args:
        entity: Raw entity mapping.

    Returns:
        A deterministic string representation suitable for tokenization/training.
    """
    can = canonicalize_entity(entity)

    tokens: list[str] = []
    tokens.append("{")

    if len(can) == 0:
        tokens.append("}")
        return " ".join(tokens)

    for i, (k, values) in enumerate(can.items()):
        if i > 0:
            tokens.append(",")

        # Emit keyval: <K> "key" : [ ... ]
        tokens.append("<K>")
        tokens.append(_string_literal(k))
        tokens.append(":")
        tokens.append("[")
        if len(values) == 0:
            tokens.append("]")
        else:
            # Emit values: <V> "val" ( , <V> "val" )*
            for i, v in enumerate(values):
                if i > 0:
                    tokens.append(",")
                tokens.append("<V>")
                tokens.append(_string_literal(v))
            tokens.append("]")

    tokens.append("}")
    return " ".join(tokens)


def entities_to_string_as_set(entities: list[dict[str, list[str]]]) -> str:
    """Serialize a sequence of entities (interpreted as a bag)to a stable string.

    Args:
        entities: List of entity mappings to serialize.

    Returns:
        A deterministic string representation of the entity sequence.

    Example:
        entities_to_string([{"a": ["x"]}, {"b": ["y"]}])
        -> "{ <K> \"a\" : [ <V> \"x\" ] } { <K> \"b\" : [ <V> \"y\" ] }"
    """
    if not entities:
        return ""

    def _entity_sort_key(e: dict[str, list[str]]) -> str:
        can_ent = canonicalize_entity(e)
        if not can_ent:
            return ""
        parts: list[str] = []
        for k in can_ent:
            vals = can_ent[k]
            part = f"{k}:{','.join(vals)}" if vals else f"{k}:"
            parts.append(part)
        return "|".join(parts)

    entities_sorted = sorted(entities, key=_entity_sort_key)
    serialized_entities = [entity_to_string(entity) for entity in entities_sorted]
    return " ".join(serialized_entities)


def parse_sequence(text: str) -> list[dict[str, list[str]]]:
    """Parse a training string containing multiple entities back into a list of canonical dicts.

    Uses lexer-based brace counting to identify entity boundaries, then calls the existing parse_entity
    function for each individual entity. Only counts { and } tokens outside of STRING tokens.

    Args:
        text: The training string containing one or more entities.

    Returns:
        List of canonicalized entity dicts.

    Raises:
        ValueError: If the input violates the expected grammar or contains invalid syntax.

    Example:
        parse_sequence("{ <K> \"a\" : [ <V> \"x\" ] } { <K> \"b\" : [ <V> \"y\" ] }")
        -> [{"a": ["x"]}, {"b": ["y"]}]
    """
    if not text or not text.strip():
        return []

    tokens = _lex(text)
    entities: list[dict[str, list[str]]] = []
    start_idx = 0

    while start_idx < len(tokens):
        # Find the end of the first entity by counting { and } tokens
        brace_count = 0
        entity_end_idx = start_idx

        for i in range(start_idx, len(tokens)):
            token_type, _ = tokens[i]
            if token_type == "{":
                brace_count += 1
            elif token_type == "}":
                brace_count -= 1
                if brace_count == 0:
                    entity_end_idx = i + 1
                    break

        if brace_count != 0:
            raise ValueError("Unterminated entity in sequence")

        # Parse the entity using the pre-lexed tokens
        entity_tokens = tokens[start_idx:entity_end_idx]
        entity = parse_entity("", _tokens=entity_tokens)
        entities.append(entity)

        # Move to the next entity
        start_idx = entity_end_idx

    return entities


def parse_entity(text: str, *, _tokens: list[Token] | None = None) -> dict[str, list[str]]:
    """Parse a training string with <K>/<V> sentinels back into a canonical entity dict.

    The parser accepts the deterministic format produced by `entity_to_string`. It is whitespace-insensitive, requires
    <K> before keys and <V> before values, and enforces JSON-quoted strings with proper escaping. Duplicate keys, if
    present, are merged (values concatenated) and canonicalized at the end.

    Args:
        text: The training string to parse.
        _tokens: Pre-lexed tokens (internal use only). If provided, text is ignored.

    Returns:
        Canonicalized dict[str, list[str]].

    Raises:
        ValueError: If the input violates the expected grammar or contains invalid quoting/escapes.
    """
    tokens = _tokens if _tokens is not None else _lex(text)
    pos = 0

    def take(expected: str) -> None:
        nonlocal pos
        if pos >= len(tokens) or tokens[pos][0] != expected:
            got = tokens[pos][0] if pos < len(tokens) else "<EOF>"
            raise ValueError(f"Expected token {expected!r}, got {got!r} at position {pos}")
        pos += 1

    def peek() -> str | None:
        return tokens[pos][0] if pos < len(tokens) else None

    result: dict[str, list[str]] = {}

    # Grammar:
    #   object -> { ( keyval ( , keyval )* )? }
    #   keyval -> <K> string : [ ( <V> string ( , <V> string )* )? ]
    take("{")
    if peek() == "}":
        take("}")
        return canonicalize_entity(result)

    # keyval ( , keyval )*
    while True:
        take("<K>")
        # STRING token contains the decoded Python str value at index 1.
        key_tok = tokens[pos]
        if key_tok[0] != "STRING":
            raise ValueError(
                f"Expected string literal for key after <K>, got {key_tok[0]!r} at position {pos}"
            )
        key = key_tok[1]  # type: ignore[index]
        pos += 1

        take(":")
        take("[")
        values: list[str] = []
        if peek() == "]":
            take("]")
        else:
            while True:
                take("<V>")
                val_tok = tokens[pos]
                if val_tok[0] != "STRING":
                    raise ValueError(
                        f"Expected string literal for value after <V>, got {val_tok[0]!r} at position {pos}",
                    )
                value = val_tok[1]  # type: ignore[index]
                pos += 1
                values.append(value)
                if peek() == ",":
                    take(",")
                    continue
                if peek() == "]":
                    take("]")
                    break
                got = peek()
                raise ValueError(f'Expected "," or "]" after value, got {got!r} at position {pos}')

        # Merge semantics if duplicate keys appear; we canonicalize later.
        if key not in result:
            result[key] = values
        else:
            result[key].extend(values)

        if peek() == ",":
            take(",")
            continue
        if peek() == "}":
            take("}")
            break
        got = peek()
        raise ValueError(f'Expected "," or "}}" after key/value pair, got {got!r} at position {pos}')

    # Always return canonicalized mapping for determinism.
    return canonicalize_entity(result)


def _string_literal(value: str) -> str:
    """Return a single JSON string literal token (including surrounding quotes)."""
    assert isinstance(value, str), "String literal must be built from str"
    dumped = json.dumps(value, ensure_ascii=False)  # e.g., "\"A\\nb\""
    assert dumped[0] == '"'
    assert dumped[-1] == '"'
    return dumped


def _lex(text: str) -> list[tuple[str, str | None]]:
    """Lex the training string into tokens: punctuation, sentinels, and JSON strings."""
    i = 0
    n = len(text)
    tokens: list[tuple[str, str | None]] = []

    while i < n:
        ch = text[i]

        # Skip whitespace.
        if ch.isspace():
            i += 1
            continue

        # Single-character punctuation.
        if ch in "{}[]:,":
            tokens.append((ch, None))
            i += 1
            continue

        # Sentinels <K> or <V>.
        if ch == "<":
            if text.startswith("<K>", i):
                tokens.append(("<K>", None))
                i += 3
                continue
            if text.startswith("<V>", i):
                tokens.append(("<V>", None))
                i += 3
                continue
            raise ValueError(f"Unknown sentinel starting at position {i}: {text[i : i + 4]!r}")

        # JSON string literal.
        if ch == '"':
            # Scan to next unescaped quote.
            j = i + 1
            while j < n:
                if text[j] == '"':
                    k = j - 1
                    backslashes = 0
                    while k >= i + 1 and text[k] == "\\":
                        backslashes += 1
                        k -= 1
                    if backslashes % 2 == 0:
                        break
                j += 1
            if j >= n or text[j] != '"':
                raise ValueError(f"Unterminated JSON string starting at position {i}")

            raw = text[i : j + 1]
            try:
                py_str = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string literal at {i}: {e.msg}") from e
            tokens.append(("STRING", py_str))
            i = j + 1
            continue

        raise ValueError(f"Unexpected character at position {i}: {ch!r}")

    return tokens
