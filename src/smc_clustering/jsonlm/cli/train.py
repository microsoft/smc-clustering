"""Training CLI for grammar-constrained JSON-entity language modeling.

This script:
  1) Trains the Byte-Level BPE tokenizer on the interiors of quoted strings from the TRAIN JSONL files.
  2) Builds EntityDatasets and pad-aware DataLoaders (EOS-filled tails).
  3) Instantiates a TinyTransformerLM, wrapped by LitConstrainedLM for constrained NLL.
  4) Trains with PyTorch Lightning, checkpointing the best 'val/loss'.
  5) Saves artifacts: vocabulary tokens, BPE model JSON, and transformer config JSON.

Example:
    uv run ./src/jsonlm/cli/train.py --train ./data_vm/datasets/fragment_set_generation/train/rebel_fragment_set_generation_dataset.jsonl --val ./data_vm/datasets/fragment_set_generation/dev/rebel_fragment_set_generation_dataset.jsonl --save_dir runs/exp1 --max_epochs 10 --decode_every 500 --device cuda --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from functools import partial

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader


try:
    from pytorch_lightning.loggers import WandbLogger

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from smc_clustering.jsonlm.api import decode_entity
from smc_clustering.jsonlm.data.collate import pad_collate
from smc_clustering.jsonlm.data.dataset import EntityDataset
from smc_clustering.jsonlm.models.lit_module import LitConstrainedLM
from smc_clustering.jsonlm.models.transformer import TransformerConfig, TransformerLM
from smc_clustering.jsonlm.serialization.encoder import entities_to_string_as_set, entity_to_string
from smc_clustering.jsonlm.serialization.normalization import normalize_entity_or_sequence
from smc_clustering.jsonlm.tokenization.trainer import train_tokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _is_wandb_configured() -> bool:
    """Check if wandb is available and properly configured."""
    if not _WANDB_AVAILABLE:
        return False

    try:
        import wandb

        # Try to check if we can initialize wandb without actually initializing
        # This is done by checking if we have an API key or are in offline mode
        from wandb.sdk.wandb_settings import Settings

        Settings()
        # If we can't get basic settings or there's no API key configured, wandb won't work
        api_key = wandb.api.api_key
        return api_key is not None and api_key != ""
    except Exception:  # noqa: BLE001
        # Any exception during wandb configuration check means it's not properly set up
        return False


def _save_artifacts(
    save_dir: str, vocab: Vocabulary, tokenizer_bpe_json: str, cfg: TransformerConfig
) -> None:
    """Persist artifacts: `vocab.json`, `bpe.json`, and `config.json` in `save_dir`."""
    os.makedirs(save_dir, exist_ok=True)
    # Save Vocabulary as ordered token list (strings).
    vocab_path = os.path.join(save_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab.as_list(), f, ensure_ascii=False, indent=2)

    # Save HF tokenizer.json (already written by caller to a temporary location).
    bpe_target = os.path.join(save_dir, "bpe.json")
    if tokenizer_bpe_json != bpe_target:
        # Copy by reading & writing to avoid shutil dependency.
        with (
            open(tokenizer_bpe_json, encoding="utf-8") as src,
            open(bpe_target, "w", encoding="utf-8") as dst,
        ):
            dst.write(src.read())

    # Save transformer config (dataclass -> dict).
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def _train_corpus_lines(paths: Sequence[str]) -> Iterable[str]:
    """Yield serialized training strings (with sentinels) from JSONL paths for tokenizer training."""
    import json

    for p in paths:
        with open(p, encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.rstrip("\n\r")
                if not line.strip():
                    continue  # ignore empty/whitespace-only lines
                try:
                    obj = json.loads(line)
                    if isinstance(obj, list):
                        # Entity sequence: serialize with entities_to_string
                        if not all(isinstance(item, dict) for item in obj):
                            raise ValueError(f"List items must all be dicts in {p}:{lineno}")
                        # Normalize sequence by removing legacy "properties" wrappers if present
                        normalized_obj = normalize_entity_or_sequence(obj, seq_mode="strict")
                        assert isinstance(normalized_obj, list), (
                            "Sequence normalization should return list"
                        )
                        yield entities_to_string_as_set(normalized_obj)
                    elif isinstance(obj, dict):
                        # Normalize entity by removing legacy "properties" wrapper if present
                        normalized_obj = normalize_entity_or_sequence(obj, seq_mode="strict")
                        assert isinstance(normalized_obj, dict), (
                            "Single entity normalization should return dict"
                        )
                        # Single entity: serialize with entity_to_string
                        yield entity_to_string(normalized_obj)
                    else:
                        raise ValueError(
                            f"Expected a JSON object or array in {p}:{lineno}, got {type(obj).__name__}"
                        )
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON parse error in {p}:{lineno}: {e.msg}") from e


class PeriodicDecodeCallback(Callback):
    """Every `every_n_steps`, print a constrained-greedy decode (as a canonical dict) for sanity."""

    def __init__(self, tokenizer, every_n_steps: int = 0, max_steps: int = 128) -> None:
        """Initialize the callback."""
        super().__init__()
        self.tokenizer = tokenizer
        self.every = int(every_n_steps)
        self.max_steps = max_steps

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LitConstrainedLM,
        outputs: torch.Tensor,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Print a decode every `every_n_steps`."""
        if self.every <= 0:
            return
        global_step = int(trainer.global_step)
        if global_step == 0 or (global_step % self.every) != 0:
            return
        try:
            d = decode_entity(pl_module.model, tokenizer=self.tokenizer, max_steps=self.max_steps)
            print(f"[decode@step={global_step}] {d!r}")
        except Exception as e:  # keep training robust to decode mishaps  # noqa: BLE001
            print(f"[decode@step={global_step}] ERROR: {e}")


def build_argparser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI."""
    p = argparse.ArgumentParser(description="Train a constrained LM over JSON entities.")
    p.add_argument("--train", nargs="+", required=True, help="Path(s) to train JSONL file(s).")
    p.add_argument("--val", nargs="+", required=True, help="Path(s) to val JSONL file(s).")
    p.add_argument("--save_dir", required=True, help="Directory to save artifacts and checkpoints.")
    p.add_argument(
        "--resume_from", default=None, help="Path to Lightning checkpoint (.ckpt) to resume from."
    )

    # Tokenizer / vocab
    p.add_argument(
        "--bpe_vocab_size",
        type=int,
        default=1200,
        help="Byte-Level BPE vocab size for string interiors.",
    )

    # Model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=10)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=768)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--tie_embeddings", action="store_true", default=True)
    p.add_argument("--no_tie_embeddings", action="store_false", dest="tie_embeddings")
    p.add_argument("--use_bias", action="store_true", default=False)

    # Optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--precision", choices=["bf16-true", "16-mixed", "32"], default="bf16-true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=1500)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument(
        "--max_steps_override",
        type=int,
        default=-1,
        help="Max optimization steps; use -1 to disable step limit (Lightning default).",
    )
    p.add_argument("--seed", type=int, default=523)
    p.add_argument(
        "--struct_weight",
        type=float,
        default=0.4,
        help="Weight for structure/EOS tokens in loss (1.0 = no down-weight).",
    )
    p.add_argument(
        "--no_downweight_eos",
        action="store_true",
        default=False,
        help="Disable down-weighting of EOS tokens.",
    )

    # Misc
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--decode_every", type=int, default=200, help="If >0, print a decode every N train steps."
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


def main(argv: list[str] | None = None) -> None:
    """Train a constrained language model over JSON entities."""
    # Configure logging once per entry; INFO is useful for CLI progress.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Parsing arguments")
    args = build_argparser().parse_args(argv)

    # Seeding
    logger.info("Setting random seeds (seed=%d)", args.seed)
    seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002
    torch.manual_seed(args.seed)

    # Build base Vocabulary and train tokenizer on serialized train corpus.
    logger.info(
        "Training tokenizer (bpe_vocab_size=%d) from %d train file(s)",
        args.bpe_vocab_size,
        len(args.train),
    )
    vocab = Vocabulary.from_default()
    tok = train_tokenizer(
        _train_corpus_lines(args.train), vocabulary=vocab, bpe_vocab_size=args.bpe_vocab_size
    )

    # Save tokenizer artifacts (HF tokenizer save returns a path).
    tmp_bpe_path = os.path.join(args.save_dir, "_tmp_bpe.json")
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info("Saving temporary tokenizer model to %s", tmp_bpe_path)
    tok.bpe.save(tmp_bpe_path)

    # Prepare model config
    logger.info(
        "Preparing transformer config (d_model=%d, n_layers=%d, n_heads=%d, d_ff=%d, max_seq_len=%d, dropout=%.3f, tie_embeddings=%s, use_bias=%s)",
        args.d_model,
        args.n_layers,
        args.n_heads,
        args.d_ff,
        args.max_seq_len,
        args.dropout,
        str(args.tie_embeddings),
        str(args.use_bias),
    )

    cfg = TransformerConfig(
        vocab_size=len(tok),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        tie_embeddings=args.tie_embeddings,
        use_bias=args.use_bias,
    )
    logger.info("Saving artifacts (vocab.json, bpe.json, config.json) to %s", args.save_dir)
    _save_artifacts(args.save_dir, vocab, tmp_bpe_path, cfg)

    # Datasets / loaders
    logger.info("Indexing training and validation datasets from JSONL files")
    train_ds = EntityDataset(
        paths=list(args.train), tokenizer=tok, max_length=args.max_seq_len, add_bos_eos=True
    )
    val_ds = EntityDataset(
        paths=list(args.val), tokenizer=tok, max_length=args.max_seq_len, add_bos_eos=True
    )
    try:  # noqa: SIM105
        logger.info("Dataset sizes: train=%d, val=%d", len(train_ds), len(val_ds))
    except Exception:  # noqa: BLE001, S110
        # len should work, but keep training robust if it doesn't for some reason
        pass

    logger.info(
        "Creating DataLoaders (batch_size=%d, num_workers=%d)",
        args.batch_size,
        args.num_workers,
    )
    collate = partial(pad_collate, tokenizer=tok)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    # Model + Lightning module
    logger.info("Initializing model and Lightning module")
    model = TransformerLM(cfg)
    max_steps_override = args.max_steps_override if args.max_steps_override is not None else -1

    lit = LitConstrainedLM(
        model=model,
        tokenizer=tok,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps_override=max_steps_override if max_steps_override > 0 else None,
        struct_weight=args.struct_weight,
        downweight_eos=(not args.no_downweight_eos),
    )

    # Device selection
    if args.device == "auto":
        use_gpu = torch.cuda.is_available()
    elif args.device == "cuda":
        use_gpu = True
    else:
        use_gpu = False
    logger.info("Selected accelerator: %s", "gpu" if use_gpu else "cpu")

    # Callbacks: checkpoint best val/loss + early stopping + optional periodic decode
    logger.info(
        "Setting up callbacks (ModelCheckpoint, EarlyStopping%s)",
        ", PeriodicDecode" if int(args.decode_every) > 0 else "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )
    es_cb = EarlyStopping(monitor="val/loss", mode="min", patience=3)
    dec_cb = PeriodicDecodeCallback(tokenizer=tok, every_n_steps=int(args.decode_every), max_steps=128)

    # Check if wandb is available and configured, fallback to CSV logger if not
    if _is_wandb_configured():
        logger.info(
            "Initializing logger (WandbLogger) [project=%s, run=%s]",
            "jsonlm",
            os.path.basename(args.save_dir.rstrip("/")),
        )
        pl_logger = WandbLogger(
            project="jsonlm",
            name=os.path.basename(args.save_dir.rstrip("/")),
            save_dir=args.save_dir,
        )
    else:
        logger.info("Wandb not available or configured, using CSVLogger")
        pl_logger = CSVLogger(
            save_dir=args.save_dir,
            name="training_logs",
        )

    logger.info(
        "Constructing Trainer (max_epochs=%d, max_steps=%d, log_every_n_steps=%d)",
        args.max_epochs,
        max_steps_override,
        50,
    )

    trainer = Trainer(
        default_root_dir=args.save_dir,
        max_epochs=args.max_epochs,
        max_steps=max_steps_override,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        callbacks=[ckpt_cb, es_cb, dec_cb],
        log_every_n_steps=50,
        logger=pl_logger,
    )

    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)
    logger.info("Starting training loop")
    trainer.fit(
        lit, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_from
    )

    # Cleanup tmp file if present
    try:
        if os.path.exists(tmp_bpe_path):
            logger.info("Removing temporary tokenizer file: %s", tmp_bpe_path)
            os.remove(tmp_bpe_path)
    except OSError:
        pass

    print(f"Training complete. Artifacts/checkpoints saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
