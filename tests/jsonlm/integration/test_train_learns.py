# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Integration test that a tiny constrained LM can improve on repeated training data."""

# tests/test_end_to_end_training.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from smc_clustering.jsonlm.api import logprob_entity
from smc_clustering.jsonlm.data.collate import pad_collate
from smc_clustering.jsonlm.data.dataset import EntityDataset
from smc_clustering.jsonlm.models.lit_module import LitConstrainedLM
from smc_clustering.jsonlm.models.transformer import TransformerConfig, TransformerLM
from smc_clustering.jsonlm.serialization.encoder import entity_to_string
from smc_clustering.jsonlm.serialization.normalization import normalize_entity_or_sequence
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _write_jsonl(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def _corpus_lines(paths: list[Path]):
    """Yield serialized training strings for tokenizer training."""
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # normalize & serialize single-entity lines
                norm = normalize_entity_or_sequence(obj, seq_mode="strict")
                assert isinstance(norm, dict)
                yield entity_to_string(norm)


def _avg_logprob(
    items: list[dict], model: torch.nn.Module, tok: JsonLMTokenizer, normalize: str = "sum"
) -> float:
    with torch.no_grad():
        vals = [
            float(logprob_entity(x, model=model.eval(), tokenizer=tok, normalize=normalize))
            for x in items
        ]
    return float(np.mean(vals))


def test_tiny_model_learns(tmp_path: Path):
    """Verify a tiny end-to-end training run improves model fit on simple data."""
    # Reproducibility & CPU-only
    seed_everything(1234, workers=True)
    torch.set_grad_enabled(True)

    # 1) Tiny synthetic dataset (one simple pattern, repeated)
    # Using a single pattern helps ensure quick, measurable improvement.
    train_items = [{"title": ["alpha"], "tag": ["x", "y"]}] * 64
    val_items = [{"title": ["alpha"], "tag": ["x", "y"]}] * 8
    train_path = tmp_path / "data" / "train.jsonl"
    val_path = tmp_path / "data" / "val.jsonl"
    _write_jsonl(train_path, train_items)
    _write_jsonl(val_path, val_items)

    # 2) Tokenizer (byte-level BPE on quoted string interiors)
    vocab = Vocabulary.from_default()
    tok = train_tokenizer(
        corpus=_corpus_lines([train_path]),
        vocabulary=vocab,
        bpe_vocab_size=64,  # tiny
    )

    # 3) Datasets & loaders
    max_len = 128
    train_ds = EntityDataset([str(train_path)], tokenizer=tok, max_length=max_len, add_bos_eos=True)
    val_ds = EntityDataset([str(val_path)], tokenizer=tok, max_length=max_len, add_bos_eos=True)

    collate = lambda batch: pad_collate(batch, tokenizer=tok)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)

    # 4) Tiny transformer config
    cfg = TransformerConfig(
        vocab_size=len(tok),
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=max_len,
        dropout=0.0,
        tie_embeddings=True,
        use_bias=False,
        pos_encoding="rope",
        norm_type="rms",
        ffn_activation="swiglu",
    )
    model = TransformerLM(cfg)

    # 5) Lightning wrapper (no warmup; learn structure & EOS at full weight)
    lit = LitConstrainedLM(
        model=model,
        tokenizer=tok,
        lr=3e-3,
        weight_decay=0.0,
        warmup_steps=0,
        max_steps_override=None,
        struct_weight=1.0,
        downweight_eos=False,
    )

    # 6) Baseline log-probability on a small sample of the train set
    sample_for_scoring = train_items[:8]
    pre = _avg_logprob(sample_for_scoring, model=model, tok=tok, normalize="sum")

    # 7) Train briefly on CPU
    trainer = Trainer(
        max_epochs=5,
        accelerator="cpu",
        devices=1,
        precision="32",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=50,
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 8) Post-training log-probability
    post = _avg_logprob(sample_for_scoring, model=model, tok=tok, normalize="sum")

    # 9) Assert improvement (higher log-probability == better)
    # A small but reliable margin to avoid flakiness on CI.
    assert post > pre + 1.0, (
        f"Expected training to improve log-probability: pre={pre:.3f}, post={post:.3f}"
    )
