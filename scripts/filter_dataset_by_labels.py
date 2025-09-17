"""
Filter a JSONL dataset by labels, keeping at most N unique labels.

Example:
    uv run scripts/filter_dataset_by_labels.py         --dataset ./data_vm/datasets/rebel_clustering_dataset.jsonl         --labels ./data_vm/datasets/rebel_clustering_ground_truth.jsonl         --max-unique 10         --out ./data_vm/datasets/clustering_small/data/rebel_clustering_dataset.jsonl         --out-labels ./data_vm/datasets/clustering_small/data/rebel_clustering_ground_truth.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the script."""
    p = argparse.ArgumentParser(description="Filter a JSONL dataset by labels")
    p.add_argument("--dataset", required=True, help="Path to input JSONL dataset")
    p.add_argument("--labels", required=True, help="Path to labels file (one label per line, in same order as dataset)")
    p.add_argument(
        "--max-unique",
        type=int,
        required=True,
        help="Maximum number of unique labels to accept. New labels after this are rejected.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSONL path. Defaults to <dataset_stem>_labels_<max_unique>_filtered.jsonl",
    )
    p.add_argument(
        "--out-labels",
        default=None,
        help="Output labels path. Defaults to <dataset_stem>_labels_<max_unique>_filtered.labels.jsonl",
    )
    return p


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def _read_labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n").strip() for line in f]


def _derive_default_paths(dataset_path: Path, max_unique: int) -> tuple[Path, Path]:
    stem = dataset_path.stem
    dataset_out = dataset_path.with_name(f"{stem}_labels_{max_unique}_filtered.jsonl")
    labels_out = dataset_path.with_name(f"{stem}_labels_{max_unique}_filtered.labels.jsonl")
    return dataset_out, labels_out


def filter_dataset(
    dataset_path: str | Path,
    labels_path: str | Path,
    max_unique: int,
    out_dataset_path: str | Path | None = None,
    out_labels_path: str | Path | None = None,
) -> tuple[int, int, int, int]:
    """Filter the dataset and write out filtered JSONL and labels.

    Returns: (processed_count, kept_count, skipped_count, unique_labels_kept)
    """
    if max_unique <= 0:
        raise ValueError("--max-unique must be positive")

    dataset_path = Path(dataset_path)
    labels_path = Path(labels_path)

    if out_dataset_path is None or out_labels_path is None:
        default_dataset_out, default_labels_out = _derive_default_paths(dataset_path, max_unique)
        out_dataset_path = Path(out_dataset_path) if out_dataset_path else default_dataset_out
        out_labels_path = Path(out_labels_path) if out_labels_path else default_labels_out
    else:
        out_dataset_path = Path(out_dataset_path)
        out_labels_path = Path(out_labels_path)

    logging.info("Counting dataset lines...")
    dataset_lines = _count_lines(dataset_path)
    logging.info(f"Dataset lines: {dataset_lines}")

    logging.info("Reading labels file...")
    labels = _read_labels(labels_path)
    logging.info(f"Labels lines: {len(labels)}")

    if len(labels) != dataset_lines:
        logging.error(
            "Dataset and labels files have different number of lines. They must correspond one-to-one. Aborting."
        )
        raise SystemExit(2)

    accepted_labels: set[str] = set()
    kept = 0
    skipped = 0
    processed = 0

    out_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    out_labels_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        dataset_path.open(encoding="utf-8") as din,
        out_dataset_path.open("w", encoding="utf-8") as dout_data,
        out_labels_path.open("w", encoding="utf-8") as dout_labels,
    ):
        for raw_line, label in zip(din, labels, strict=True):
            processed += 1
            # Decide whether to keep this record
            if label in accepted_labels:
                dout_data.write(raw_line)
                dout_labels.write(label + "\n")
                kept += 1
            else:
                if len(accepted_labels) < max_unique:
                    accepted_labels.add(label)
                    dout_data.write(raw_line)
                    dout_labels.write(label + "\n")
                    kept += 1
                else:
                    skipped += 1

            if processed % 100000 == 0:
                logging.info(
                    f"Processed {processed:,} lines, kept {kept:,}, skipped {skipped:,}, accepted labels: {len(accepted_labels):,}"
                )

    logging.info(
        f"Finished. Processed {processed} lines, kept {kept}, skipped {skipped}. Unique labels kept: {len(accepted_labels)}."
    )
    logging.info(f"Accepted labels ({len(accepted_labels)}): sample -> {list(accepted_labels)[:10]}")

    return processed, kept, skipped, len(accepted_labels)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the filtering operation. Returns an exit code (0 on success)."""
    args = build_argparser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        processed, kept, skipped, unique = filter_dataset(
            args.dataset,
            args.labels,
            args.max_unique,
            args.out,
            args.out_labels,
        )
    except (OSError, ValueError, SystemExit):
        logging.exception("Error while filtering")
        return 1

    logging.info(
        f"Done. Wrote filtered dataset to: {args.out or '<derived>'} and labels to: {args.out_labels or '<derived>'}"
    )
    logging.info(f"Stats -> processed: {processed}, kept: {kept}, skipped: {skipped}, unique_labels: {unique}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
