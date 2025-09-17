"""
Utilities for reading JSONL files that contain dict[str, list[str]] entities.

This module provides a strict, streaming JSONL reader that yields Python dicts line by line. Errors include file and
line-number context to make troubleshooting data issues easy. The functions assume UTF-8 and ignore empty lines.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from json.decoder import JSONDecodeError


def read_jsonl_entities(path: str) -> Iterator[dict[str, list[str]]]:
    """Yield dict[str, list[str]] entities from a JSONL file, one per non-empty line.

    Each non-empty line is parsed via json.loads. Lines consisting only of whitespace are skipped. The function raises a
    ValueError with file/line context if a line fails JSON parsing or does not decode to a dict.

    Args:
        path: Filesystem path to a UTF-8 encoded JSONL file.

    Yields:
        Parsed entities as dict[str, list[str]].

    Raises:
        ValueError: If a line cannot be decoded as JSON or is not a dict.
    """
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")

    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue  # ignore empty/whitespace-only lines
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    obj = obj[0]
                if "properties" in obj:
                    obj = obj["properties"]
            except JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {path}:{lineno}: {e.msg}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected a JSON object in {path}:{lineno}, got {type(obj).__name__}")
            # Note: value-type validation happens later in canonicalization/serialization path.
            yield obj
