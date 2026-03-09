# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Aggregate sweep metrics over seeds.

Parses a metrics file produced by `sweep_eval_clustering_smc.py` and
aggregates requested metrics across seeds (or any excluded keys).

Example usage:

  uv run scripts/aggregate_metrics.py \
      --input ./sweep_outputs/metrics.txt \
      --metrics [LL,LP] \
      --exclude seed \
      --sort-by LL

Output format (tab separated):

  Configuration    LL              LP
  [alpha=1.0, max_evals=10, max_particles=10, split=0]\t-1100+-50\t-13111+-132

Notes:
  * By default uses population standard deviation (divide by N) since
    we typically treat the collected seeds as the full evaluation set.
    Use --sample-std to switch to sample standard deviation.
  * Blocks with `<run failed ...>` or `<no metrics captured>` are skipped.
  * Missing metrics for a configuration are reported as `NA`.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


def _parse_list_like(value: str) -> list[str]:
    v = value.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        # split on commas (do not attempt JSON coercion to stay simple)
        return [tok.strip() for tok in inner.split(",") if tok.strip()]
    if "," in v:
        return [tok.strip() for tok in v.split(",") if tok.strip()]
    return [v]


def build_argparser() -> argparse.ArgumentParser:
    """Create and return the argument parser for the aggregation script."""
    p = argparse.ArgumentParser(description="Aggregate sweep metrics over seeds")
    p.add_argument("--input", required=True, help="Path to metrics.txt produced by sweep script")
    p.add_argument(
        "--metrics",
        required=True,
        help="Metric name or list expression (e.g. LL or [LL,LP,f1])",
    )
    p.add_argument(
        "--exclude",
        default="seed",
        help="Comma/list of key names to exclude from grouping (default: seed)",
    )
    p.add_argument(
        "--sort-by",
        default=None,
        help="Metric name to sort configurations by (mean value). Defaults to first metric requested.",
    )
    p.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order (default ascending).",
    )
    p.add_argument(
        "--precision",
        type=int,
        default=None,
        help="If set, use fixed number of decimal places for mean/std formatting.",
    )
    p.add_argument(
        "--sample-std",
        action="store_true",
        help="Use sample standard deviation (N-1) instead of population.",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum number of values required for a configuration (default 1).",
    )
    p.add_argument(
        "--na-token",
        default="NA",
        help="Token to display when a metric is missing for a configuration.",
    )
    p.add_argument(
        "--no-header",
        action="store_true",
        help="Suppress header line in output.",
    )
    p.add_argument(
        "--show-counts",
        action="store_true",
        help="Append sample count (n=...) after each formatted metric cell.",
    )
    return p


def _format_stat(values: list[float], precision: int | None, sample_std: bool) -> str:
    if not values:
        return "NA"
    if len(values) == 1:
        mean = values[0]
        std = 0.0
    else:
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if sample_std and len(values) > 1 else statistics.pstdev(values)
    if precision is not None:
        fmt = f"{{:.{precision}f}}"
        return f"{fmt.format(mean)}  ±  {fmt.format(std)}"
    # auto precision heuristic
    THRESH_3 = 1000
    THRESH_2 = 100
    THRESH_1 = 10

    def auto_dec(x: float) -> int:
        ax = abs(x)
        if ax >= THRESH_3:
            return 0
        if ax >= THRESH_2:
            return 1
        if ax >= THRESH_1:
            return 2
        return 3

    dec = max(auto_dec(mean), auto_dec(std))
    fmt = f"{{:.{dec}f}}"
    return f"{fmt.format(mean)}  ±  {fmt.format(std)}"


def _normalize_metrics_arg(raw: str) -> list[str]:
    return _parse_list_like(raw)


def _normalize_exclude_arg(raw: str) -> set[str]:
    if not raw:
        return set()
    return set(_parse_list_like(raw))


def parse_metrics_file(
    path: Path, exclude: set[str]
) -> tuple[dict[str, dict[str, list[float]]], set[str]]:
    """Parse metrics file and return grouped metric values.

    Returns:
        grouped: mapping from configuration key (string repr) to {metric: [values]}
        all_metrics: set of all metric names observed
    """
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_metrics: set[str] = set()

    current_header: dict[str, str] | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                current_header = None
                continue
            if line.startswith("[") and line.endswith("]"):
                inside = line[1:-1]
                parts = [p.strip() for p in inside.split(",") if p.strip()]
                header_dict: dict[str, str] = {}
                for part in parts:
                    if "=" not in part:
                        continue
                    k, v = part.split("=", 1)
                    header_dict[k.strip()] = v.strip()
                current_header = header_dict
                continue
            if line.startswith("<"):
                # Skip status or failure annotations
                continue
            if current_header is None:
                # stray metric line without header
                continue
            if ":" not in line:
                continue
            name, val_str = line.split(":", 1)
            name = name.strip()
            val_str = val_str.strip()
            try:
                value = float(val_str)
            except ValueError:
                continue
            # Build grouping key excluding requested keys
            effective_items = [
                (k, current_header[k]) for k in sorted(current_header) if k not in exclude
            ]
            conf_key = "[" + ", ".join(f"{k}={v}" for k, v in effective_items) + "]"
            grouped[conf_key][name].append(value)
            all_metrics.add(name)

    return grouped, all_metrics


def sort_configurations(
    configs: Iterable[str],
    data: dict[str, dict[str, list[float]]],
    metric: str,
    descending: bool,
) -> list[str]:
    """Return configuration keys sorted by mean of a given metric."""

    def key_fn(conf: str) -> float:
        vals = data[conf].get(metric, [])
        if not vals:
            return math.inf if not descending else -math.inf
        return statistics.fmean(vals)

    return sorted(configs, key=key_fn, reverse=descending)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_argparser().parse_args(argv)
    metrics = _normalize_metrics_arg(args.metrics)
    if not metrics:
        raise SystemExit("--metrics produced empty list")
    exclude = _normalize_exclude_arg(args.exclude)
    metrics_path = Path(args.input)
    if not metrics_path.is_file():
        raise SystemExit(f"Input file not found: {metrics_path}")

    grouped, all_metrics = parse_metrics_file(metrics_path, exclude)
    if not grouped:
        raise SystemExit("No data parsed from metrics file")

    # Warn if requested metrics missing
    missing = [m for m in metrics if m not in all_metrics]
    if missing:
        print(f"Warning: requested metrics not found: {', '.join(missing)}", flush=True)

    sort_metric = args.sort_by or metrics[0]
    ordered = sort_configurations(grouped.keys(), grouped, sort_metric, args.descending)

    # Header
    if not args.no_header:
        print("Configuration\t" + "\t".join(metrics))

    for conf in ordered:
        metric_values = []
        counts_ok = True
        for m in metrics:
            vals = grouped[conf].get(m, [])
            if len(vals) < args.min_count:
                counts_ok = False
            metric_values.append(vals)
        if not counts_ok:
            continue  # skip insufficient data
        cells: list[str] = []
        for vals in metric_values:
            if not vals:
                cell = args.na_token
            else:
                cell = _format_stat(vals, args.precision, args.sample_std)
                if args.show_counts:
                    cell += f" (n={len(vals)})"
            cells.append(cell)
        print(conf + "\t" + "\t".join(cells))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
