"""
Tests for strict JSONL reading: streaming parse, type checking, and error messages.

We verify that valid lines are yielded as dicts, empty lines are skipped, and malformed lines raise ValueError
with file/line-number context to make debugging easy.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from jsonlm.utils.io import read_jsonl_entities


def test_read_jsonl_entities_streams_and_types() -> None:
    """Valid dict lines are yielded; empty lines are ignored."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"a": ["x", "y"]}\n')
            f.write("\n")
            f.write('{"b": []}\n')

        got = list(read_jsonl_entities(path))
        assert isinstance(got[0], dict) and got[0] == {"a": ["x", "y"]}
        assert isinstance(got[1], dict) and got[1] == {"b": []}
        assert len(got) == 2
    finally:
        os.remove(path)


def test_read_jsonl_entities_malformed_line_has_context() -> None:
    """Malformed JSON raises ValueError with path and line number."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"ok": ["x"]}\n')
            f.write('{"bad": [oops]}\n')  # invalid JSON
        with pytest.raises(ValueError) as exc:
            _ = list(read_jsonl_entities(path))
        msg = str(exc.value)
        assert path in msg and ":2" in msg
        assert "parse error" in msg.lower() or "json" in msg.lower()
    finally:
        os.remove(path)
