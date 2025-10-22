from __future__ import annotations

import json
from pathlib import Path

from jsonlm.cli import deltas as deltas_cli
from jsonlm.cli import train as train_cli


def _write_jsonl(p: Path, lines: list[str]) -> None:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_deltas_cli_runs(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    out_path = tmp_path / "deltas.txt"
    save_dir = tmp_path / "runs" / "exp"

    _write_jsonl(
        train_path,
        [
            '{"a": ["x", "y"], "b": ["c"]}',
            '{"title": ["Notes"], "tags": ["ai", "ml"]}',
        ],
    )
    _write_jsonl(val_path, ['{"a": ["x"], "b": ["c"]}'])

    # Train tiny model
    train_cli.main(
        [
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
        ],
    )

    # Build minimal pairs (two lines)
    _write_jsonl(
        pairs_path,
        [
            f"{json.dumps([{'a': ['x']}, {'a': ['x', 'y']}])}",
            f"{json.dumps([{'b': ['c']}, {'title': ['Notes'], 'tags': ['ai']}])}",
        ],
    )

    deltas_cli.main(
        [
            "--artifacts",
            str(save_dir),
            "--ckpt",
            str(save_dir / "last.ckpt"),
            "--pairs",
            str(pairs_path),
            "--out",
            str(out_path),
            "--format",
            "txt",
            "--batch_size",
            "2",
            "--device",
            "cpu",
        ],
    )

    text = out_path.read_text(encoding="utf-8").strip()
    assert len(text.splitlines()) == 2
    # sanity: deltas are finite floats
    for line in text.splitlines():
        float(line)
