"""Defines project-wide token string constants for JSON punctuation and structural sentinels.

The constants are later assembled into an ordered list for Vocabulary.
"""

from __future__ import annotations


# JSON punctuation (single-character tokens).
LBRACE: str = "{"
RBRACE: str = "}"
LBRACKET: str = "["
RBRACKET: str = "]"
COLON: str = ":"
COMMA: str = ","
QUOTE: str = '"'  # The literal double-quote character.

# Structural sentinels (multi-character, angle-bracketed control tokens).
K_SENTINEL: str = "<K>"
V_SENTINEL: str = "<V>"

# Sequence-control tokens.
BOS: str = "<BOS>"
EOS: str = "<EOS>"
PAD: str = "<PAD>"

# Canonical, fixed ordering of specials used across the codebase.
# We place punctuation first (to simplify grammar masks), then structural sentinels, then sequence-control tokens.
SPECIAL_TOKENS: list[str] = [
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
    COLON,
    COMMA,
    QUOTE,
    K_SENTINEL,
    V_SENTINEL,
    BOS,
    EOS,
    PAD,
]
