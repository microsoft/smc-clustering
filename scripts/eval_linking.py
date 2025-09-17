"""
Evaluate JSON-LM linking performance in MS-KeBAB.

Example usage:
    uv run scripts/eval_linking.py --artifacts ./data_vm/artifacts --ckpt ./data_vm/artifacts/last.ckpt --task_instance Linking-REBEL-Incremental-Test --offset 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter as pc

import numpy as np
import torch
from kebab import mskebab
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task
from tokenizers import Tokenizer as HFTokenizer

from jsonlm.models.scoring import compute_deltas_batched
from jsonlm.models.transformer import TinyTransformerLM, TransformerConfig
from jsonlm.tokenization.tokenizer import JsonLMTokenizer
from jsonlm.tokenization.vocab import Vocabulary


def _load_artifacts(artifacts_dir: str) -> tuple[JsonLMTokenizer, TransformerConfig]:
    """Load tokenizer and model config from artifacts directory."""
    vocab_path = os.path.join(artifacts_dir, "vocab.json")
    bpe_path = os.path.join(artifacts_dir, "bpe.json")
    cfg_path = os.path.join(artifacts_dir, "config.json")

    with open(vocab_path, encoding="utf-8") as f:
        tokens = json.load(f)
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError("vocab.json must be a JSON list of token strings.")
    vocab = Vocabulary.from_tokens(tokens)

    bpe = HFTokenizer.from_file(bpe_path)
    tok = JsonLMTokenizer(vocabulary=vocab, bpe=bpe, specials_size=len(vocab), bpe_size=bpe.get_vocab_size())

    with open(cfg_path, encoding="utf-8") as f:
        cfg = TransformerConfig(**json.load(f))

    return tok, cfg


def _load_model(ckpt_path: str, cfg: TransformerConfig, device: torch.device) -> TinyTransformerLM:
    """Load model from checkpoint."""
    model = TinyTransformerLM(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if "state_dict" in ckpt:
        state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] Missing keys: {missing}, Unexpected keys: {unexpected}")
    return model


def _load_linking_pairs(task_instance: Task) -> list[tuple[dict, dict]]:
    """Yield pairs (A,B) from MS-KeBAB."""
    ent_pairs: Iterable[tuple[tuple[Entity, Entity], bool]] = task_instance.read_items()
    pairs = [
        (e1.properties if isinstance(e1, Entity) else [f.properties for f in e1], e2.properties)
        for (e1, e2), _ in ent_pairs
    ]
    return list(pairs)


def build_argparser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description="Evaluate JSON-LM in MS-KeBAB Linking task.")
    p.add_argument("--artifacts", default="./artifacts", help="Directory with vocab.json, bpe.json, config.json")
    p.add_argument("--ckpt", default="./artifacts/last.ckpt", help="Checkpoint (.ckpt or raw state_dict)")
    p.add_argument("--task_instance", type=str, default="Linking-REBEL-Test", help="MS-KeBAB Linking task instance")
    p.add_argument(
        "--skip_calibration",
        action="store_true",
        help="Whether or not to calibrate the scores",
    )
    p.add_argument("--out", default="./output/scores.txt", help="Output file path for Δ values")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for processing")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Additive logit offset used when scoring entity linking pairs (was previously hardcoded)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """Run the evaluation."""
    args = build_argparser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Arguments: {args}")

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    logging.info(f"Using device: {device}")

    tok, cfg = _load_artifacts(args.artifacts)
    logging.info(f"Loaded tokenizer and config from {args.artifacts}")

    model = _load_model(args.ckpt, cfg, device=device)
    logging.info(f"Loaded model from {args.ckpt}")

    benchmark = mskebab.get_default_benchmark()
    task_instance = benchmark.tasks_by_name[args.task_instance]
    logging.info(f"Loaded task instance {args.task_instance}")

    pairs = _load_linking_pairs(task_instance)
    logging.info(f"Loaded {len(pairs)} pairs from task instance {args.task_instance}")

    if torch.cuda.is_available():
        with torch.amp.autocast("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            logging.info("Computing deltas with CUDA AMP...")

            start = pc()

            deltas = compute_deltas_batched(
                pairs, model=model, tokenizer=tok, offset=args.offset, batch_size=args.batch_size, device=device
            )

            end = pc()
            torch.cuda.synchronize(device)
            peak_vram_alloc = torch.cuda.max_memory_allocated(device)
    else:
        start = pc()

        deltas = compute_deltas_batched(
            pairs, model=model, tokenizer=tok, offset=args.offset, batch_size=args.batch_size, device=device
        )

        end = pc()
        peak_vram_alloc = 0.0

    elapsed = float(end - start)
    logging.info(f"Time (s): {elapsed:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savetxt(args.out, deltas)

    metrics = task_instance.evaluate(Path(args.out))

    metrics["Total Time (s)"] = elapsed
    metrics["Candidates per Second (C/s)"] = len(pairs) / elapsed
    metrics["Peak GPU Memory (GB)"] = peak_vram_alloc / 1.0e9

    for key, val in metrics.items():
        logging.info(f"{key}: {val:.6f}")
        print(f"{key}: {val:.6f}")


if __name__ == "__main__":
    main()
