# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tokenization package exposing Vocabulary and (later) tokenizer utilities.

We keep vocabulary (special tokens, ids, and lookups) separate from the tokenizer implementation so that
grammar and serialization can depend on a stable set of IDs even before BPE training. The `Vocabulary`
dataclass provides those stable mappings and convenience accessors.
"""

from __future__ import annotations

from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


__all__ = ["Vocabulary"]
