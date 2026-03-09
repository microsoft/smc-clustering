"""
Hybrid tokenizer: specials for JSON structure + Byte-Level BPE for string contents.

This tokenizer lexes the serialized training string (with <K>/<V> sentinels), keeps structure tokens as single IDs,
and applies Byte-Level BPE to the *interior* of quoted strings. This design avoids OOVs while preserving explicit JSON
structure tokens. Decoding reconstructs the original serialized, space-separated form.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tokenizers import Tokenizer  # type: ignore[import-not-found]
from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # type: ignore[import-not-found]
from tokenizers.models import BPE  # type: ignore[import-not-found]
from tokenizers.pre_tokenizers import ByteLevel  # type: ignore[import-not-found]
from tokenizers.trainers import BpeTrainer  # type: ignore[import-not-found]

from jsonlm.serialization import encoder
from jsonlm.tokenization.vocab import Vocabulary


@dataclass(slots=True)
class JsonLMTokenizer:
    """Tokenizer that preserves structure tokens and applies BPE to quoted-string interiors.

    The global vocabulary is the concatenation of:
      * Specials (from Vocabulary): '{', '}', '[', ']', ':', ',', '"', '<K>', '<V>', '<BOS>', '<EOS>', '<PAD>'
      * BPE subword pieces for string contents (byte-level) — appended after specials.

    Attributes:
        vocabulary: Base specials-first Vocabulary.
        bpe: HuggingFace tokenizers `Tokenizer` trained for byte-level BPE on string contents.
        specials_size: Number of special tokens (len(vocabulary)).
        bpe_size: Size of the BPE sub-vocabulary (bpe.get_vocab_size()).
    """

    vocabulary: Vocabulary
    bpe: Tokenizer
    specials_size: int
    bpe_size: int

    @classmethod
    def train_from_corpus(
        cls,
        corpus: Iterable[str],
        vocabulary: Vocabulary,
        bpe_vocab_size: int = 800,
    ) -> JsonLMTokenizer:
        """Train a Byte-Level BPE over the *contents* of quoted strings extracted from a serialized corpus.

        Uses an initial alphabet of all bytes and enables byte fallback to guarantee coverage for unseen characters.

        Args:
            corpus: Iterable of serialized entity strings (space-separated with <K>/<V>, quotes, punctuation).
            vocabulary: The base specials-first Vocabulary.
            bpe_vocab_size: Target size of the BPE sub-vocabulary for string interiors.

        Returns:
            A JsonLMTokenizer ready for encode/decode without OOVs.

        Raises:
            ValueError: If the corpus has no quoted strings to train on.
        """

        # Create a generator that yields string literal contents on-the-fly
        def string_literals_generator():
            found_any = False
            for line in corpus:
                toks = encoder._lex(line)  # ("STRING", text) for literals
                for typ, val in toks:
                    if typ == "STRING":
                        assert isinstance(val, str)
                        found_any = True
                        yield val

            if not found_any:
                raise ValueError("No string literals found in corpus to train BPE.")

        # Train Byte-Level BPE: full byte alphabet + byte fallback => zero OOV risk.
        bpe = Tokenizer(BPE(unk_token="[UNK]", byte_fallback=True))
        bpe.pre_tokenizer = ByteLevel(add_prefix_space=False)
        bpe.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=bpe_vocab_size,
            special_tokens=[],  # specials are managed externally in Vocabulary
            initial_alphabet=ByteLevel.alphabet(),  # ensure all bytes are representable
        )
        bpe.train_from_iterator(string_literals_generator(), trainer=trainer)

        return cls(vocabulary=vocabulary, bpe=bpe, specials_size=len(vocabulary), bpe_size=bpe.get_vocab_size())

    def __len__(self) -> int:
        """Return size of the joint vocabulary (specials + BPE)."""
        return self.specials_size + self.bpe_size

    def _bpe_to_joint_id(self, bpe_id: int) -> int:
        """Map a BPE piece id to a joint id space offset after specials."""
        assert 0 <= bpe_id < self.bpe_size
        return self.specials_size + bpe_id

    def _joint_to_bpe_id(self, joint_id: int) -> int:
        """Map a joint id (after specials) back to BPE piece id."""
        bpe_id = joint_id - self.specials_size
        assert 0 <= bpe_id < self.bpe_size
        return bpe_id

    def _is_special_id(self, tok_id: int) -> bool:
        """Return True if tok_id belongs to the specials prefix."""
        return 0 <= tok_id < self.specials_size

    def _is_bpe_id(self, tok_id: int) -> bool:
        """Return True if tok_id belongs to the BPE segment."""
        return self.specials_size <= tok_id < self.specials_size + self.bpe_size

    def encode(self, text: str, add_bos_eos: bool = True) -> list[int]:
        """Encode a serialized training string into joint IDs, optionally adding BOS/EOS.

        This method lexes `text`, emits structure tokens as specials, and BPE-encodes the interior of string literals
        between opening/closing quotes. It preserves exact round-trip behavior (decode(strip_bos_eos=True) reproduces
        the input string).

        Args:
            text: Serialized entity string produced by `entity_to_string`.
            add_bos_eos: Whether to prepend BOS and append EOS IDs.

        Returns:
            List of joint token IDs.

        Raises:
            ValueError: If `text` cannot be lexed according to the expected grammar.
        """
        toks = encoder._lex(text)
        ids: list[int] = []

        if add_bos_eos:
            ids.append(self.vocabulary.bos_id)

        for typ, val in toks:
            if typ in ("{", "}", "[", "]", ":", ",", "<K>", "<V>"):
                ids.append(self.vocabulary.token_id(typ))
            elif typ == "STRING":
                # Emit opening quote special, BPE pieces for interior, then closing quote special.
                ids.append(self.vocabulary.quote_id)
                assert isinstance(val, str)
                pieces = self.bpe.encode(val)
                ids.extend(self._bpe_to_joint_id(pid) for pid in pieces.ids)
                ids.append(self.vocabulary.quote_id)
            else:
                raise ValueError(f"Unexpected token type from lexer: {typ!r}")

        if add_bos_eos:
            ids.append(self.vocabulary.eos_id)

        return ids

    def decode(self, ids: list[int], strip_bos_eos: bool = False) -> str:
        """Decode joint IDs back into the original serialized training string.

        This reconstructs quoted string literals by collecting BPE pieces bracketed by quote specials and emitting a
        single JSON string literal token via the serializer's JSON escaping rules.

        Args:
            ids: List of joint token IDs.
            strip_bos_eos: Whether to drop leading BOS and trailing EOS if present.

        Returns:
            The original space-separated serialized string.

        Raises:
            IndexError: If IDs fall out of range for either specials or BPE segment.
            ValueError: If quotes are unbalanced or interior segments are malformed.
        """
        seq = ids
        if strip_bos_eos and seq:
            if seq[0] == self.vocabulary.bos_id:
                seq = seq[1:]
            if seq and seq[-1] == self.vocabulary.eos_id:
                seq = seq[:-1]

        tokens_out: list[str] = []
        i = 0
        n = len(seq)

        while i < n:
            tok_id = seq[i]
            if not (0 <= tok_id < len(self)):
                raise IndexError(f"Token ID out of joint-vocab range: {tok_id}")

            if self._is_special_id(tok_id):
                tok = self.vocabulary.id_token(tok_id)
                if tok == '"':
                    # Begin quoted string: collect BPE pieces until the next quote.
                    i += 1
                    piece_ids: list[int] = []
                    while i < n and seq[i] != self.vocabulary.quote_id:
                        if not self._is_bpe_id(seq[i]):
                            raise ValueError("Non-BPE token inside quoted string during decode.")
                        piece_ids.append(self._joint_to_bpe_id(seq[i]))
                        i += 1
                    if i >= n or seq[i] != self.vocabulary.quote_id:
                        raise ValueError("Unterminated quoted string during decode.")
                    # Convert BPE ids back to string and re-emit as a single JSON literal.
                    val = self.bpe.decode(piece_ids)
                    tokens_out.append(encoder._string_literal(val))
                    i += 1  # consume closing quote
                else:
                    tokens_out.append(tok)
                    i += 1
            else:
                raise ValueError("BPE piece encountered outside of quoted string during decode.")

        return " ".join(tokens_out)
