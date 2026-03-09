# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Sampling CLI: generate entities from a trained checkpoint (greedy or stochastic).

Usage example:
    python -m jsonlm.cli.sample --artifacts ./runs/exp1 --ckpt ./runs/exp1/last.ckpt --num 10 --mode sample --top_p 0.9 --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer as HFTokenizer  # type: ignore[import-not-found]

from smc_clustering.jsonlm.models.decode import decode_greedy, decode_sample
from smc_clustering.jsonlm.models.transformer import TransformerConfig, TransformerLM
from smc_clustering.jsonlm.serialization.encoder import parse_entity
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample valid JSON entities from a trained model.")
    p.add_argument("--artifacts", required=True, help="Dir with vocab.json, bpe.json, config.json")
    p.add_argument("--ckpt", required=True, help="Lightning .ckpt or raw state_dict")
    p.add_argument("--num", type=int, default=1, help="Number of samples to generate")
    p.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    p.add_argument("--max_steps", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0, help="0 disables top-k")
    p.add_argument("--top_p", type=float, default=1.0, help="1.0 disables nucleus")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    artifacts_dir = Path(args.artifacts)
    with (artifacts_dir / "vocab.json").open(encoding="utf-8") as f:
        vocab = Vocabulary.from_tokens(json.load(f))
    bpe = HFTokenizer.from_file(str(artifacts_dir / "bpe.json"))
    tok = JsonLMTokenizer(
        vocabulary=vocab, bpe=bpe, specials_size=len(vocab), bpe_size=bpe.get_vocab_size()
    )
    with (artifacts_dir / "config.json").open(encoding="utf-8") as f:
        cfg = TransformerConfig(**json.load(f))

    model = TransformerLM(cfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state = (
        {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
        if "state_dict" in ckpt
        else state
    )
    _ = model.load_state_dict(state, strict=False)

    for i in range(args.num):
        if args.mode == "greedy":
            text = decode_greedy(model=model, tokenizer=tok, max_steps=args.max_steps, device=device)
        else:
            top_k = args.top_k if args.top_k > 0 else None
            top_p = args.top_p if args.top_p < 1.0 else None
            text = decode_sample(
                model=model,
                tokenizer=tok,
                max_steps=args.max_steps,
                device=device,
                temperature=args.temperature,
                top_k=top_k,
                top_p=top_p,
                seed=args.seed,
            )

        print()
        try:
            print(parse_entity(text))
        except Exception as e:
            print(f"(parse error?) {e}")


if __name__ == "__main__":
    main()
