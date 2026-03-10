# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Padding-aware collate utilities for batching BOS…EOS sequences.

We batch variable-length [T] LongTensors returned by EntityDataset by padding each row to the batch maximum length.
To keep the constrained-loss well-defined and grammar-friendly, we fill the padded tail with **EOS** (not PAD). The
Lightning module masks the post-EOS region to allow only EOS, so targets in the padded tail being EOS is consistent.

Shapes:
    Input batch (list): [N] of tensors with shapes [T_i].
    Output tensor:      [N, max_i T_i] (long), BOS at [:,0], EOS appears at or before the last column.

Note: We still place PAD in places where a "real" PAD might be needed in the future, but for the MVP we simply emit EOS
in the tail and let the grammar masks handle it cleanly.
"""

from __future__ import annotations

import torch

from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer


def pad_collate(batch: list[torch.Tensor], tokenizer: JsonLMTokenizer) -> torch.Tensor:
    """Pad a list of [T] BOS…EOS sequences to a [B, L_max] LongTensor.

    The padded region is filled with EOS so that (a) the next-token targets are EOS and (b) our grammar masks can
    allow only EOS there. This avoids "no allowed tokens" and keeps loss finite/easy.

    Args:
        batch: List of 1-D LongTensors, each containing BOS…EOS.
        tokenizer: The tokenizer whose vocabulary provides BOS/EOS/PAD ids.

    Returns:
        A 2-D LongTensor of shape [B, L_max].
    """
    assert batch, "pad_collate received an empty batch."
    eos_id = tokenizer.vocabulary.eos_id
    pad_id = tokenizer.vocabulary.pad_id  # kept for potential future use

    max_len = max(seq.shape[0] for seq in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)  # [B, L_max]

    for i, seq in enumerate(batch):
        L = int(seq.shape[0])
        out[i, :L] = seq
        if max_len > L:
            # Fill padded tail with EOS so both inputs/targets are EOS there.
            out[i, L:] = eos_id

    return out
