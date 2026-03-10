# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

r"""Δ-scoring CLI: read JSONL of [entity_1, entity_2] and write Δ = logP(A∪B) − logP(A) − logP(B).

Input format:
    Each line is a JSON array with two objects, e.g.:
        [{"a": ["x"]}, {"a": ["x", "y"], "b": ["z"]}]

Output formats:
    * txt   (default): one floating-point Δ per line, aligned with input lines.
    * jsonl: one object per line: { "delta": <float> }  (kept minimal on purpose).

Artifacts expected in --artifacts:
  - vocab.json   (Vocabulary tokens)
  - bpe.json     (HF Tokenizers JSON for BPE)
  - config.json  (TransformerConfig)

Examples:
    python -m jsonlm.cli.deltas \\
        --artifacts runs/exp1 --ckpt runs/exp1/last.ckpt \\
        --pairs data/pairs.jsonl --out out/deltas.txt --batch_size 64 --device auto
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import torch
from tokenizers import Tokenizer as HFTokenizer  # type: ignore[import-not-found]

from smc_clustering.jsonlm.models.scoring import compute_deltas_batched
from smc_clustering.jsonlm.models.transformer import TransformerConfig, TransformerLM
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _load_artifacts(artifacts_dir: str) -> tuple[JsonLMTokenizer, TransformerConfig]:
    artifacts_path = Path(artifacts_dir)
    vocab_path = artifacts_path / "vocab.json"
    bpe_path = artifacts_path / "bpe.json"
    cfg_path = artifacts_path / "config.json"

    with vocab_path.open(encoding="utf-8") as f:
        tokens = json.load(f)
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError("vocab.json must be a JSON list of token strings.")
    vocab = Vocabulary.from_tokens(tokens)

    bpe = HFTokenizer.from_file(str(bpe_path))
    tok = JsonLMTokenizer(
        vocabulary=vocab, bpe=bpe, specials_size=len(vocab), bpe_size=bpe.get_vocab_size()
    )

    with cfg_path.open(encoding="utf-8") as f:
        cfg = TransformerConfig(**json.load(f))
    return tok, cfg


def _load_model(ckpt_path: str, cfg: TransformerConfig, device: torch.device) -> TransformerLM:
    model = TransformerLM(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if "state_dict" in ckpt:
        state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] Missing keys: {missing}, Unexpected keys: {unexpected}")
    return model


def _read_pairs(path: str) -> Iterable[tuple[dict, dict]]:
    """Yield pairs (A,B) from a JSONL where each line is a 2-element array of objects."""
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                arr = json.loads(stripped_line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Parse error in {path}:{lineno}: {e}") from e
            if not (
                isinstance(arr, list)
                and len(arr) == 2
                and isinstance(arr[0], dict)
                and isinstance(arr[1], dict)
            ):
                raise ValueError(
                    f"Parse error in {path}:{lineno}: Line must be a JSON array of two objects."
                )
            yield arr[0], arr[1]


def _write_out_txt(out_path: str, values: Iterable[float]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{v:.8f}\n")


def _write_out_jsonl(out_path: str, values: Iterable[float]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for v in values:
            json.dump({"delta": float(v)}, f, ensure_ascii=False)
            f.write("\n")


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description="Compute Δ for JSONL pairs [A, B].")
    p.add_argument("--artifacts", required=True, help="Dir with vocab.json, bpe.json, config.json")
    p.add_argument("--ckpt", required=True, help="Checkpoint (.ckpt or raw state_dict)")
    p.add_argument("--pairs", required=True, help="Input JSONL file with [A, B] per line")
    p.add_argument("--out", required=True, help="Output file path for Δ values")
    p.add_argument(
        "--format", choices=["txt", "jsonl"], default="txt", help="Output format (default: txt)"
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


def main(argv: list[str] | None = None) -> None:
    """Run the delta-scoring CLI."""
    args = build_argparser().parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    tok, cfg = _load_artifacts(args.artifacts)
    model = _load_model(args.ckpt, cfg, device=device)

    pairs = list(_read_pairs(args.pairs))
    deltas = compute_deltas_batched(
        pairs, model=model, tokenizer=tok, batch_size=args.batch_size, device=device
    )

    for pair, delta in zip(pairs, deltas, strict=False):
        print(f"Δ({pair[0]}, {pair[1]}) = {delta:.8f}")

    if args.format == "txt":
        _write_out_txt(args.out, deltas)
    else:
        _write_out_jsonl(args.out, deltas)

    print(f"Wrote {len(deltas)} deltas to {args.out} ({args.format}).")


if __name__ == "__main__":
    main()
