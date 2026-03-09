"""
Torch Dataset that reads JSONL entities or entity sequences, canonicalizes on-the-fly, and encodes to token IDs.

The dataset builds a byte-offset index for each provided JSONL file so it can seek to a line on demand without loading
everything into memory. For each item, it parses JSON into either dict[str, list[str]] or list[dict[str, list[str]]],
serializes using the format with `<K>` and `<V>`, then encodes to joint IDs via the tokenizer (optionally adding BOS/EOS).
Supports both single entities per line and sequences of entities per line with Kleene-plus grammar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from json.decoder import JSONDecodeError

import torch
from torch.utils.data import Dataset

from jsonlm.serialization.encoder import entities_to_string_as_set, entity_to_string
from jsonlm.serialization.normalization import normalize_entity_or_sequence
from jsonlm.tokenization.tokenizer import JsonLMTokenizer


@dataclass(slots=True)
class _LineRef:
    """Reference to a single JSONL line by file path, byte offset, and 1-based line number."""

    path: str
    offset: int
    lineno: int


class EntityDataset(Dataset):
    """Dataset that yields encoded ID tensors from one or more JSONL files.

    Items are read lazily: the constructor scans files once to record byte offsets. Each `__getitem__` seeks to the
    indexed offset, reads one line, parses JSON, serializes with `<K>/<V>`, and encodes with the provided tokenizer.
    Supports both single entities (dict) and entity sequences (list of dicts) per line using Kleene-plus grammar.

    Shapes:
        Returned tensor has shape [T] (1-D) with dtype torch.long and includes BOS/EOS if `add_bos_eos=True`.

    Args:
        paths: List of JSONL file paths.
        tokenizer: The tokenizer used to encode serialized strings.
        add_bos_eos: Whether to add BOS/EOS during encoding (default True).
    """

    def __init__(
        self,
        paths: list[str],
        tokenizer: JsonLMTokenizer,
        max_length: int = -1,
        add_bos_eos: bool = True,
    ) -> None:
        """Initialize the dataset."""
        super().__init__()
        if not paths:
            raise ValueError("EntityDataset requires at least one JSONL path.")
        self.paths = list(paths)
        self.tokenizer = tokenizer
        self.add_bos_eos = add_bos_eos
        self._index: list[_LineRef] = []
        self._build_index(max_length)

    def _build_index(self, max_length: int) -> None:
        """Scan all files once and build a byte-offset index of non-empty lines."""
        for p in self.paths:
            with open(p, "rb") as f:
                lineno = 0
                while True:
                    start = f.tell()
                    raw = f.readline()
                    if not raw:
                        break
                    lineno += 1
                    # Keep line if it contains non-whitespace content.
                    if 0 < len(raw.strip()) <= (max_length if max_length > 0 else len(raw)):
                        self._index.append(_LineRef(path=p, offset=start, lineno=lineno))

    def __len__(self) -> int:
        """Return the total number of non-empty lines across all indexed files."""
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return encoded IDs for the JSON entity on the indexed line as a 1-D LongTensor."""
        if idx < 0 or idx >= len(self._index):
            raise IndexError(f"Index out of range: {idx}")

        ref = self._index[idx]
        # Read exactly one line in binary, then decode as utf-8 strictly.
        with open(ref.path, "rb") as f:
            f.seek(ref.offset)
            raw = f.readline()
        try:
            line = raw.decode("utf-8").rstrip("\n\r")
        except UnicodeDecodeError as e:
            raise ValueError(f"UTF-8 decoding error in {ref.path}:{ref.lineno}") from e

        # Parse JSON; produce helpful errors with line context.
        try:
            obj = json.loads(line)
        except JSONDecodeError as e:
            raise ValueError(f"JSON parse error in {ref.path}:{ref.lineno}: {e.msg}") from e

        # Handle both single entities (dict) and entity sequences (list of dicts)
        if isinstance(obj, dict):
            # Normalize entity by removing legacy "properties" wrapper if present
            normalized_obj = normalize_entity_or_sequence(obj, seq_mode="strict")
            assert isinstance(normalized_obj, dict), "Single entity normalization should return dict"
            # Single entity: serialize with entity_to_string
            s = entity_to_string(normalized_obj)
        elif isinstance(obj, list):
            # Entity sequence: validate all elements are dicts and serialize with entities_to_string
            if not all(isinstance(item, dict) for item in obj):
                raise ValueError(f"List items must all be dicts in {ref.path}:{ref.lineno}")
            # Normalize sequence by removing legacy "properties" wrappers if present
            normalized_obj = normalize_entity_or_sequence(obj, seq_mode="strict")
            assert isinstance(normalized_obj, list), "Sequence normalization should return list"
            s = entities_to_string_as_set(normalized_obj)
        else:
            raise ValueError(f"Expected a JSON object or array in {ref.path}:{ref.lineno}, got {type(obj).__name__}")

        # Encode to IDs with BOS before sequence and EOS after sequence
        ids: list[int] = self.tokenizer.encode(s, add_bos_eos=self.add_bos_eos)

        # Return as 1-D LongTensor. Shape: [T].
        t = torch.tensor(ids, dtype=torch.long)
        assert t.dim() == 1, "Encoded sequence must be 1-D"
        return t
