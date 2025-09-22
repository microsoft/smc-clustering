"""
Filter a JSONL dataset by labels, keeping at most N unique labels.

If a "confusing entities" map is provided (JSONL lines of [entity_id, [confusing_entity_ids...]]), then accepting a
label will also accept all of its confusing labels (provided doing so does not exceed the unique limit). This is done
atomically: either the whole cluster (label + confusing) fits under the limit and is accepted, or the label is skipped.

Example:
    uv run scripts/filter_dataset_by_labels.py \
        --dataset ./data_vm/datasets/clustering/test/rebel_clustering_dataset.jsonl \
        --labels ./data_vm/datasets/clustering/test/rebel_clustering_ground_truth.jsonl \
        --confusing-map ./data_vm/datasets/rebel_confusing_entities_map.jsonl \
        --max-unique 50 \
        --min-confusing-cluster-size 2 \
        --max-confusing-cluster-entities 5 \
        --out ./data_vm/datasets/clustering_f/test/rebel_clustering_dataset.jsonl \
        --out-labels ./data_vm/datasets/clustering_f/test/rebel_clustering_ground_truth.jsonl

    uv run scripts/filter_dataset_by_labels.py \
        --dataset ./data_vm/datasets/clustering/test/rebel_clustering_dataset.jsonl \
        --labels ./data_vm/datasets/clustering/test/rebel_clustering_ground_truth.jsonl \
        --max-unique 100 \
        --out ./data_vm/datasets/clustering_f/test/rebel_clustering_dataset.jsonl \
        --out-labels ./data_vm/datasets/clustering_f/test/rebel_clustering_ground_truth.jsonl

"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


CLUSTER_LOG_SAMPLE = 25  # number of cluster members to show in acceptance log
_MIN_QUOTED_LEN = 2  # minimal length to consider stripping surrounding quotes


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
    p.add_argument(
        "--confusing-map",
        default="rebel_confusing_entities_map.jsonl",
        help="Optional path to JSONL map of entity_id -> list(confusing entity_ids). Default: rebel_confusing_entities_map.jsonl. Use empty string to disable.",
    )
    p.add_argument(
        "--min-confusing-cluster-size",
        type=int,
        default=2,
        help="Minimum size (number of distinct labels) a confusing cluster must have (after filtering) to trigger grouped acceptance. Clusters smaller than this act as singletons.",
    )
    p.add_argument(
        "--max-confusing-cluster-entities",
        type=int,
        default=4,
        help="Optional cap on number of labels taken from a confusing cluster (0 = no cap). Root label always included; remaining members truncated deterministically (sorted order).",
    )
    return p


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def _read_labels(path: Path) -> list[str]:
    def _normalize(label: str) -> str:
        # Remove surrounding single or double quotes if present: "Q1" -> Q1, 'Q1' -> Q1
        if len(label) >= _MIN_QUOTED_LEN and label[0] == label[-1] and label[0] in {'"', "'"}:
            return label[1:-1]
        return label

    with path.open(encoding="utf-8") as f:
        return [_normalize(line.rstrip("\n").strip()) for line in f]


def _derive_default_paths(dataset_path: Path, max_unique: int) -> tuple[Path, Path]:
    stem = dataset_path.stem
    dataset_out = dataset_path.with_name(f"{stem}_labels_{max_unique}_filtered.jsonl")
    labels_out = dataset_path.with_name(f"{stem}_labels_{max_unique}_filtered.labels.jsonl")
    return dataset_out, labels_out


def _load_confusing_map(path: Path, allowed_entities: set[str] | None = None) -> dict[str, set[str]]:
    """Load confusing entities map from JSONL lines where each line is [entity_id, [confusing_ids...]].

    Filtering semantics when `allowed_entities` is provided:
      - Root entities not in the allowed set are only kept if at least one confusing id is in the allowed set; otherwise dropped.
      - After loading, the cluster is intersected with the allowed set; if the root is removed by intersection, the cluster is dropped.
      - Returned mapping always includes the (retained) root in its own cluster.

    Returns dict[root] = set(confusing_entities_including_root). Missing/unreadable file -> empty dict.
    """
    if not path or str(path) == "":  # disabled
        return {}
    if not path.exists():
        logging.info("Confusing map file '%s' not found; proceeding without it.", path)
        return {}

    mapping: dict[str, set[str]] = {}
    processed = 0
    try:
        with path.open(encoding="utf-8") as fin:
            for line_no, raw_line in enumerate(fin, 1):
                processed += 1
                if processed % 500_000 == 0:
                    logging.info(f"Processed {processed:,} lines of confusing map...")

                line = raw_line.strip()
                if not line:
                    continue
                try:
                    root, conf_list = json.loads(line)
                    if not isinstance(root, str) or not isinstance(conf_list, list):
                        raise TypeError("Invalid line structure; expected [str, list]")
                    confusing_set = {c for c in conf_list if isinstance(c, str)}
                    confusing_set.add(root)
                    if allowed_entities is not None:
                        if root not in allowed_entities and not (confusing_set & allowed_entities):
                            continue
                        confusing_set &= allowed_entities
                        if root not in confusing_set:
                            continue
                    mapping[root] = confusing_set
                except Exception as e:  # noqa: BLE001
                    logging.warning("Failed to parse confusing map line %d: %s", line_no, e)
    except OSError as e:
        logging.info("Could not read confusing map file '%s': %s; proceeding without it.", path, e)
        return {}

    return mapping


def filter_dataset(
    dataset_path: str | Path,
    labels_path: str | Path,
    max_unique: int,
    out_dataset_path: str | Path | None = None,
    out_labels_path: str | Path | None = None,
    confusing_map_path: str | Path | None = None,
    min_confusing_cluster_size: int = 1,
    max_confusing_cluster_entities: int = 0,
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

    # Build set of all labels (still useful for default singleton mapping for unseen labels)
    all_labels = set(labels)

    if min_confusing_cluster_size <= 0:
        raise ValueError("--min-confusing-cluster-size must be positive")

    confusing_map: dict[str, set[str]] = {}
    if confusing_map_path:
        confusing_map = _load_confusing_map(Path(confusing_map_path), allowed_entities=all_labels)
        if confusing_map:
            logging.info("Loaded confusing map clusters: %d", len(confusing_map))
        else:
            logging.info("Confusing map disabled or empty; proceeding with per-label acceptance.")
            if min_confusing_cluster_size > 1:
                logging.info(
                    "min-confusing-cluster-size=%d requested but no clusters available; behaving as singleton mode.",
                    min_confusing_cluster_size,
                )

    # Construct label -> cluster mapping directly from map; unseen labels are singletons.
    label_to_cluster: dict[str, set[str]] = {root: set(cluster) for root, cluster in confusing_map.items()}
    missing_singletons: list[str] = []
    for lbl in all_labels:
        if lbl not in label_to_cluster:
            label_to_cluster[lbl] = {lbl}
            missing_singletons.append(lbl)

    if missing_singletons:
        sample = ", ".join(sorted(missing_singletons)[:CLUSTER_LOG_SAMPLE])
        if len(missing_singletons) > CLUSTER_LOG_SAMPLE:
            sample += "..."
        logging.warning(
            "%d labels not present in confusing map; treated as singletons. Sample: %s",
            len(missing_singletons),
            sample,
        )

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
            # Decide whether to keep this record (possibly introducing entire cluster)
            if label in accepted_labels:
                dout_data.write(raw_line)
                dout_labels.write(json.dumps(label) + "\n")
                kept += 1
                continue

            cluster = label_to_cluster.get(label, {label})
            new_labels = cluster - accepted_labels

            if not new_labels:
                # All labels already accepted
                dout_data.write(raw_line)
                dout_labels.write(json.dumps(label) + "\n")
                kept += 1
                continue

            cluster_size = len(cluster)
            effective_cluster = cluster if cluster_size >= min_confusing_cluster_size else {label}

            original_effective_size = len(effective_cluster)
            if max_confusing_cluster_entities and len(effective_cluster) > max_confusing_cluster_entities:
                # Deterministic truncation: always include the triggering label, then add others in sorted order
                others = sorted(x for x in effective_cluster if x != label)
                take = max_confusing_cluster_entities - 1  # already including label
                take = max(take, 0)
                truncated = {label, *others[:take]}
                effective_cluster = truncated

            new_effective = effective_cluster - accepted_labels

            if len(accepted_labels) + len(new_effective) <= max_unique:
                accepted_labels.update(new_effective)
                dout_data.write(raw_line)
                dout_labels.write(json.dumps(label) + "\n")
                kept += 1
                if len(new_effective) > 1:
                    logging.info(
                        "Accepted cluster via label '%s': orig_size=%d used_size=%d (added %d new). Members added: %s",
                        label,
                        original_effective_size,
                        len(effective_cluster),
                        len(new_effective),
                        ", ".join(sorted(new_effective)[:CLUSTER_LOG_SAMPLE])
                        + ("..." if len(new_effective) > CLUSTER_LOG_SAMPLE else ""),
                    )
                else:
                    logging.debug("Accepted singleton label '%s'", label)
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
            args.confusing_map if getattr(args, "confusing_map", None) else None,
            args.min_confusing_cluster_size,
            args.max_confusing_cluster_entities,
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
