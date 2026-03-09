"""Evaluation CLI: score entities and (optionally) compute Δ on pairs.

This script reloads a trained model and tokenizer artifacts, then:
  * Computes average log-likelihood (sum / mean / bits-per-token) on a JSONL file; and/or
  * Computes Δ = logP(AuB) - logP(A) - logP(B) for tab-separated JSON pairs.

Artifacts expected in --artifacts:
  - vocab.json   (ordered list of special tokens; matches Vocabulary.from_default() for MVP)
  - bpe.json     (HF Tokenizers JSON for Byte-Level BPE)
  - config.json  (TransformerConfig as JSON)

Examples:
    # Score a dataset
    python -m jsonlm.cli.eval --artifacts runs/exp1 --ckpt runs/exp1/last.ckpt --data data/val.jsonl

    # Compute Δ over pairs
    python -m jsonlm.cli.eval --artifacts runs/exp1 --ckpt runs/exp1/best.ckpt --pairs pairs.tsv
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

import torch
from tokenizers import Tokenizer as HFTokenizer  # type: ignore[import-not-found]

from smc_clustering.jsonlm import constants
from smc_clustering.jsonlm.api import delta, logprob_entity, logprob_sequence
from smc_clustering.jsonlm.models.transformer import TransformerConfig, TransformerLM
from smc_clustering.jsonlm.serialization.normalization import normalize_entity_or_sequence
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


def _load_artifacts(artifacts_dir: str) -> tuple[JsonLMTokenizer, TransformerConfig]:
    vocab_path = f"{artifacts_dir}/vocab.json"
    bpe_path = f"{artifacts_dir}/bpe.json"
    cfg_path = f"{artifacts_dir}/config.json"

    with open(vocab_path, encoding="utf-8") as f:
        tokens = json.load(f)
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError("vocab.json must be a JSON list of token strings.")
    # Sanity: ensure the required specials are present.
    for req in (
        constants.BOS,
        constants.EOS,
        constants.PAD,
        constants.K_SENTINEL,
        constants.V_SENTINEL,
        constants.QUOTE,
    ):
        if req not in tokens:
            raise ValueError(f"vocab.json missing required special token: {req}")
    vocab = Vocabulary.from_tokens(tokens)

    bpe = HFTokenizer.from_file(bpe_path)
    tok = JsonLMTokenizer(
        vocabulary=vocab, bpe=bpe, specials_size=len(vocab), bpe_size=bpe.get_vocab_size()
    )

    with open(cfg_path, encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = TransformerConfig(**cfg_dict)
    return tok, cfg


def _load_model_from_ckpt(ckpt_path: str, cfg: TransformerConfig, device: torch.device) -> TransformerLM:
    """Load TinyTransformerLM weights from a Lightning checkpoint or plain state_dict."""
    model = TransformerLM(cfg)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        # Strip "model." prefix used inside LitConstrainedLM
        new_state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing or unexpected:
            print(f"[warn] Missing keys: {missing}, Unexpected keys: {unexpected}")
    else:
        # Assume it's a raw state_dict for the underlying model.
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            print(f"[warn] Missing keys: {missing}, Unexpected keys: {unexpected}")
    model.eval()
    return model


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained JSON-entity LM.")
    p.add_argument(
        "--artifacts", required=True, help="Directory containing vocab.json, bpe.json, config.json."
    )
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.ckpt or state_dict).")
    p.add_argument("--data", default=None, help="JSONL file to score (dict or list[dict] per line).")
    p.add_argument("--pairs", default=None, help="TSV with two JSON objects per line to compute Δ.")
    p.add_argument("--normalize", choices=["sum", "mean", "bpt"], default="sum")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


def _iter_pairs(path: str) -> Iterable[tuple[dict, dict]]:
    """Yield pairs A,B from a TSV file where each column is a JSON object string."""
    import json as _json

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                a_str, b_str = line.split("$", 1)
                a = _json.loads(a_str)
                b = _json.loads(b_str)
                if not isinstance(a, dict) or not isinstance(b, dict):
                    raise ValueError("Both columns must be JSON objects.")
            except Exception as e:
                raise ValueError(f"Failed to parse pairs at line {lineno}: {e}") from e
            yield a, b


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load artifacts + model
    tok, cfg = _load_artifacts(args.artifacts)
    model = _load_model_from_ckpt(args.ckpt, cfg, device=device)

    # Score dataset (if provided)
    if args.data is not None:
        total = 0.0
        count = 0
        with torch.no_grad():
            # Read raw JSON lines to handle both dict and list[dict]
            with open(args.data, encoding="utf-8") as f:
                for lineno, raw in enumerate(f, start=1):
                    line = raw.rstrip("\n\r")
                    if not line.strip():
                        continue  # ignore empty/whitespace-only lines
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            # Normalize entity by removing legacy "properties" wrapper if present
                            normalized_obj = normalize_entity_or_sequence(obj, seq_mode="lenient")
                            assert isinstance(normalized_obj, dict), (
                                "Single entity normalization should return dict"
                            )
                            # Single entity: use logprob_entity
                            lp = logprob_entity(
                                normalized_obj,
                                model=model,
                                tokenizer=tok,
                                normalize=args.normalize,
                                device=device,
                            )
                        elif isinstance(obj, list):
                            # Entity sequence: use logprob_sequence with include_eos=False
                            if not all(isinstance(item, dict) for item in obj):
                                raise ValueError(f"List items must all be dicts in {args.data}:{lineno}")

                            # Normalize sequence by removing legacy "properties" wrappers if present (lenient mode)
                            normalized_obj = normalize_entity_or_sequence(obj, seq_mode="lenient")
                            assert isinstance(normalized_obj, list), (
                                "Sequence normalization should return list"
                            )
                            lp = logprob_sequence(
                                normalized_obj,
                                model=model,
                                tokenizer=tok,
                                include_eos=False,
                                normalize=args.normalize,
                                device=device,
                            )
                            print(
                                f"[debug] {args.data}:{lineno}  items={len(normalized_obj)}  {lp=:.6f}"
                            )
                        else:
                            raise ValueError(
                                f"Expected a JSON object or array in {args.data}:{lineno}, got {type(obj).__name__}",
                            )
                        total += float(lp)
                        count += 1
                    except json.JSONDecodeError as e:
                        raise ValueError(f"JSON parse error in {args.data}:{lineno}: {e.msg}") from e
        if count == 0:
            print(f"No entities found in {args.data}.")
        else:
            avg = total / count
            print(f"[data] {args.data}  items={count}  avg_{args.normalize}={avg:.6f}")

    # Compute Δ on pairs (if provided)
    if args.pairs is not None:
        vals: list[float] = []
        with torch.no_grad():
            for a, b in _iter_pairs(args.pairs):
                d = delta(a, b, model=model, tokenizer=tok)
                vals.append(d)
                print(f"[debug] {args.pairs}\tΔ={d:.6f}\tA={a}\tB={b}")
        if not vals:
            print(f"No pairs found in {args.pairs}.")
        else:
            import math

            mean_delta = sum(vals) / len(vals)
            std_delta = math.sqrt(sum((x - mean_delta) ** 2 for x in vals) / max(1, len(vals) - 1))
            print(f"[pairs] {args.pairs}  n={len(vals)}  meanΔ={mean_delta:.6f}  stdev={std_delta:.6f}")


if __name__ == "__main__":
    main()
