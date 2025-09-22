"""Sweep runner for eval_clustering_smc.

Allows specifying scalar or list-valued arguments to generate a Cartesian
product of runs.

Examples:
    (test)
    uv run scripts/sweep_eval_clustering_smc.py \
        --config ./scripts/config/benchmark_conf.json \
        --artifacts ./data_vm/artifacts \
        --ckpt ./data_vm/artifacts/best.ckpt \
        --offset 6.2146 \
        --task_instance Clustering-REBEL-200 \
        --alpha 1.0 \
        --seed [0,1] \
        --max_particles 10 \
        --max_evals 0 \
        --split_interval [0,1]

    uv run scripts/sweep_eval_clustering_smc.py \
        --config ./scripts/config/benchmark_conf.json \
        --artifacts ./data_vm/artifacts \
        --ckpt ./data_vm/artifacts/best.ckpt \
        --offset 6.2146 \
        --task_instance Clustering-REBEL-200 \
        --alpha 1.0 \
        --seed [0,1,2,3,4,5,6,7,8,9,10] \
        --max_particles 100 \
        --max_evals 100 \
        --split_interval [0,1]

    uv run scripts/sweep_eval_clustering_smc.py \
        --config ./scripts/config/benchmark_conf.json \
        --artifacts ./data_vm/artifacts \
        --ckpt ./data_vm/artifacts/best.ckpt \
        --offset 6.2146 \
        --task_instance Clustering-REBEL-200 \
        --alpha 1.0 \
        --seed [0,1,2,3,4,5,6,7,8,9,10] \
        --max_particles 50 \
        --max_evals 50 \
        --split_interval [0,1]

    uv run scripts/sweep_eval_clustering_smc.py \
        --config ./scripts/config/benchmark_conf.json \
        --artifacts ./data_vm/artifacts \
        --ckpt ./data_vm/artifacts/best.ckpt \
        --offset 6.2146 \
        --task_instance Clustering-REBEL-200 \
        --alpha 1.0 \
        --seed [0,1,2,3,4,5,6,7,8,9,10] \
        --max_particles 10 \
        --max_evals 10 \
        --split_interval [0,1]

Use --dry-run to only print the planned commands.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shlex
import subprocess


# Parameters we allow sweeping over (others are treated as scalar pass-through)
SWEEP_PARAMS = {
    "alpha",
    "seed",
    "prior_scale",
    "max_particles",
    "max_evals",
    "threshold",
    "split_interval",
    "batch_size",
}


def _parse_list_like(value: str) -> list[str]:
    """Parse a value that might encode a list.

    Supports:
      - JSON-like lists: [1,2,3]
      - Comma separated: 1,2,3
      - Single scalar: 5
    Returns list of string tokens (no type coercion yet).
    """
    v = value.strip()
    if v.startswith("[") and v.endswith("]"):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass  # fall back to manual
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [tok.strip() for tok in inner.split(",") if tok.strip()]
    if "," in v:
        return [tok.strip() for tok in v.split(",") if tok.strip()]
    return [v]


def build_argparser() -> argparse.ArgumentParser:
    """Construct argument parser for sweep script.

    Returns:
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    p = argparse.ArgumentParser(description="Sweep runner for eval_clustering_smc.py")
    p.add_argument("--script", default="scripts/eval_clustering_smc.py", help="Path to evaluation script")
    p.add_argument("--out-root", default="./sweep_outputs", help="Root directory for sweep outputs")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    # Collect arbitrary passthrough args after '--'
    p.add_argument("--", dest="passthrough", nargs=argparse.REMAINDER, help="Arguments after -- are passed verbatim")

    # Known arguments (may be swept)
    p.add_argument("--task_instance", required=True, help="Task instance name (always included in output path)")
    for arg in sorted(SWEEP_PARAMS):
        p.add_argument(f"--{arg}", required=False, help=f"Value or list for {arg}")

    # Additional common args passed directly if provided
    p.add_argument("--config", help="Config path")
    p.add_argument("--artifacts", help="Artifacts directory")
    p.add_argument("--ckpt", help="Checkpoint path")
    p.add_argument("--surrogate", help="Surrogate model path")
    p.add_argument("--device", help="Device override")
    p.add_argument("--offset", help="Offset override")

    return p


def _expand_param(name: str, value: str | None) -> list[str]:
    if value is None:
        return []
    if name in SWEEP_PARAMS:
        return _parse_list_like(value)
    return [value]


def _cartesian_product(param_values: dict[str, list[str]]):
    """Yield dicts for the Cartesian product of parameter values."""
    keys = sorted(param_values)
    lists = [param_values[k] for k in keys]
    for combo in itertools.product(*lists):
        yield dict(zip(keys, combo, strict=True))


def _sanitize_value(val: str) -> str:
    return val.replace(".", "_")


def main(argv: list[str] | None = None) -> int:
    """Entry point for running a sweep.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument list (defaults to sys.argv).

    Returns:
    -------
    int
        Process exit code (0 on success).
    """
    args = build_argparser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Gather sweep parameters
    sweep_inputs: dict[str, list[str]] = {}
    for name in SWEEP_PARAMS:
        raw = getattr(args, name)
        if raw is not None:
            expanded = _expand_param(name, raw)
            if expanded:
                sweep_inputs[name] = expanded

    if not sweep_inputs:
        logging.error("No sweep parameters provided; specify at least one of: %s", ", ".join(sorted(SWEEP_PARAMS)))
        return 2

    # Generate all combinations
    combos = list(_cartesian_product(sweep_inputs))
    logging.info("Planned %d runs", len(combos))

    os.makedirs(args.out_root, exist_ok=True)
    metrics_path = os.path.join(args.out_root, "metrics.txt")
    if not os.path.exists(metrics_path):
        # Initialize empty metrics file (no header per spec)
        with open(metrics_path, "w", encoding="utf-8"):
            pass

    base_cmd_prefix = ["uv", "run", args.script]

    # Add pass-through scalar args that are not sweeped but provided
    passthrough_scalars: list[tuple[str, str]] = []
    for extra in ["config", "artifacts", "ckpt", "surrogate", "device", "offset"]:
        val = getattr(args, extra)
        if val is not None:
            passthrough_scalars.append((f"--{extra}", str(val)))

    extra_passthrough = args.passthrough or []

    for combo in combos:
        # Build output name
        parts = ["eval_smc_clustering", _sanitize_value(args.task_instance)]
        for k in sorted(combo):
            parts.append(f"{k}_{_sanitize_value(combo[k])}")
        out_dir = os.path.join(args.out_root, "_".join(parts))
        os.makedirs(out_dir, exist_ok=True)

        cmd = [*base_cmd_prefix, "--task_instance", args.task_instance, "--out", out_dir]
        for k, v in combo.items():
            cmd += [f"--{k}", v]
        for k, v in passthrough_scalars:
            cmd += [k, v]
        cmd += extra_passthrough

        logging.info("Run: %s", " ".join(shlex.quote(c) for c in cmd))
        if not args.dry_run:
            log_path = os.path.join(out_dir, "log.txt")
            metrics_lines: list[str] = []
            in_metrics = False
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write("Command: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
                lf.flush()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    lf.write(line)
                    if not in_metrics and line.strip() == "METRICS":
                        in_metrics = True
                        continue
                    if in_metrics:
                        metrics_lines.append(line.rstrip("\n"))
                proc.wait()
                exit_code = proc.returncode or 0
            # Append metrics section
            header = ", ".join(f"{k}={combo[k]}" for k in sorted(combo))
            with open(metrics_path, "a", encoding="utf-8") as mf:
                mf.write(f"[{header}]\n")
                if metrics_lines:
                    for ml in metrics_lines:
                        mf.write(ml + "\n")
                else:
                    mf.write("<no metrics captured>\n")
                if exit_code != 0:
                    mf.write(f"<run failed exit_code={exit_code}>\n")
                mf.write("\n")
            if exit_code != 0:
                logging.error("Run failed (exit=%d). See %s", exit_code, log_path)
            else:
                logging.info("Completed successfully -> %s", log_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
