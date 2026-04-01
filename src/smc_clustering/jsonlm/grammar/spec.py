# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Grammar state definitions for the constrained JSON-with-<K>/<V> language.

The grammar models a single JSON object whose values are arrays of strings. Strings are serialized with leading/trailing
quote tokens, and their interiors are tokenized by byte-level BPE. The state machine is intentionally small and
stackless because the language has only one-level arrays (per key) and no nested objects.
"""

from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    """Enumeration of parser states for constrained token sequencing."""

    START = auto()  # Only BOS has been seen; expect '{'.
    AFTER_LBRACE = auto()  # After '{'; expect '<K>' or '}'.
    AFTER_K = auto()  # After '<K>'; expect opening '"'.
    IN_KEY_STRING = auto()  # Inside key string: BPE pieces or closing '"'.
    AFTER_KEY_STRING = auto()  # After closing key '"'; expect ':'.
    AFTER_COLON = auto()  # After ':'; expect '['.
    AFTER_LBRACKET = auto()  # After '['; expect '<V>' or ']'.
    AFTER_V = auto()  # After '<V>'; expect opening '"'.
    IN_VALUE_STRING = auto()  # Inside value string: BPE pieces or closing '"'.
    AFTER_VALUE_STRING = auto()  # After closing value '"'; expect ',' or ']'.
    AFTER_RBRACKET = auto()  # After ']' of values array; expect ',' or '}'.
    END = auto()  # After '}' at the top level; only EOS allowed.
