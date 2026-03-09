"""
Trainer for the hybrid tokenizer that uses Byte-Level BPE inside quoted strings.

The trainer extracts raw string-literal contents from the serialized corpus and fits a byte-level BPE model on them.
We configure the BPE with the full byte alphabet and byte fallback to guarantee coverage for unseen characters.
"""

from __future__ import annotations

from collections.abc import Iterable

from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def train_tokenizer(corpus: Iterable[str], vocabulary: Vocabulary, bpe_vocab_size: int = 800) -> JsonLMTokenizer:
    """Train a hybrid tokenizer (specials + byte-level BPE for string interiors) from a serialized corpus."""
    return JsonLMTokenizer.train_from_corpus(corpus=corpus, vocabulary=vocabulary, bpe_vocab_size=bpe_vocab_size)
