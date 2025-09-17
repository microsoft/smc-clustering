"""
End-to-end smoke test for the training CLI.

This test creates a tiny JSONL train/val set, runs the jsonlm.cli.train entrypoint with
a tiny Transformer configuration (1 epoch, batch_size=2, CPU), and then asserts that the
expected artifacts/checkpoints were created in the output directory.

The goal is not model quality—just to catch wiring regressions (tokenizer training,
DataLoader + padding, Lightning Trainer, checkpointing, and artifact saving).
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonlm.cli import train as train_cli


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_train_cli_runs_and_saves_artifacts(tmp_path: Path) -> None:
    """Run the training CLI on a micro dataset and verify saved files exist."""
    # Prepare tiny dataset
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    save_dir = tmp_path / "runs" / "exp_e2e"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Two small entities for train; one for val.
    train_lines = [
        '{"a": ["x", "y"], "b": ["c"]}',
        '{"title": ["Notes"], "tags": ["ai", "ml"]}',
    ]
    val_lines = [
        '{"a": ["x"], "b": ["c"]}',
    ]
    _write_jsonl(train_path, train_lines)
    _write_jsonl(val_path, val_lines)

    args = [
        "--train",
        str(train_path),
        "--val",
        str(val_path),
        "--save_dir",
        str(save_dir),
        "--bpe_vocab_size",
        "32",
        "--batch_size",
        "2",
        "--max_epochs",
        "1",
        "--device",
        "cpu",
        # Small Transformer config
        "--d_model",
        "32",
        "--n_layers",
        "1",
        "--n_heads",
        "4",
        "--d_ff",
        "64",
        "--max_seq_len",
        "64",
        "--dropout",
        "0.0",
        # Faster convergence not required; keep defaults for lr/weight_decay
        "--decode_every",
        "0",
        "--num_workers",
        "0",
        "--seed",
        "123",
    ]
    # Import-and-call keeps the test lightweight and avoids subprocess overhead.
    train_cli.main(args)

    # Check expected artifacts exist
    vocab_json = save_dir / "vocab.json"
    bpe_json = save_dir / "bpe.json"
    cfg_json = save_dir / "config.json"
    last_ckpt = save_dir / "last.ckpt"

    assert vocab_json.exists(), "vocab.json was not created"
    assert bpe_json.exists(), "bpe.json was not created"
    assert cfg_json.exists(), "config.json was not created"
    assert last_ckpt.exists(), "last.ckpt was not created"

    # Optional: quick sanity on config contents.
    cfg = json.loads(cfg_json.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict) and "vocab_size" in cfg and cfg["d_model"] == 32
