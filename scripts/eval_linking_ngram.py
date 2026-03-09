"""Evaluate n-gram surrogate model linking performance in MS-KeBAB.

Example usage:
    uv run scripts/eval_linking_ngram.py --artifacts ./data_vm/artifacts --task_instance Linking-REBEL-Test --prior_scale 0.01
    uv run scripts/eval_linking_ngram.py --artifacts ./data_vm/artifacts --task_instance Linking-REBEL-Validation --prior_scale 0.01
    uv run scripts/eval_linking_ngram.py --artifacts ./data_vm/artifacts --task_instance Linking-REBEL-Incremental-Test --prior_scale 0.01
    uv run scripts/eval_linking_ngram.py --artifacts ./data_vm/artifacts --task_instance Linking-REBEL-Incremental-Validation --prior_scale 0.01

The required pickle with prior counts is expected at:
    <artifacts>/wikipedia_names_2gram_counts.pickle
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter as pc

import numpy as np
from kebab import mskebab
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task

from smc_clustering.clustering.surrogate_models import Bigram, CountDict, get_ngram_counts


class NameBigram(Bigram):
    """Retrieves name property for use in the bigram model."""

    def __init__(self, prior_scale, prior_counts):
        super().__init__(prior_scale, prior_counts)

    def post_predictive(self, obs, n, summary):
        if type(obs) is list:
            name = obs[0].properties["name"]
        else:
            name = obs.properties["name"]

        return super().post_predictive(name, n, summary)


def _load_linking_pairs(task_instance: Task) -> list[tuple[Entity, Entity]]:
    """Load raw (Entity, Entity) pairs from the task."""
    ent_pairs: Iterable[tuple[tuple[Entity, Entity], bool]] = task_instance.read_items()
    return [(e1, e2) for (e1, e2), _ in ent_pairs]


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(description="Evaluate n-gram surrogate model in MS-KeBAB Linking task.")
    p.add_argument(
        "--artifacts",
        default="./data_vm/artifacts",
        help="Directory with wikipedia_names_2gram_counts.pickle",
    )
    p.add_argument(
        "--task_instance", type=str, default="Linking-REBEL-Test", help="MS-KeBAB Linking task instance"
    )
    p.add_argument(
        "--out", default="./output/scores_ngram.txt", help="Output file path for delta values"
    )
    p.add_argument(
        "--prior_scale", type=float, default=0.01, help="Dirichlet prior scale (alpha multiplier)"
    )
    p.add_argument(
        "--n", type=int, default=2, choices=[2], help="Order of the n-gram model (only bigram supported)"
    )
    return p


def _load_ngram_prior(artifacts_dir: str) -> CountDict:
    """Load prior bigram counts (wikipedia_names_2gram_counts.pickle)."""
    path = os.path.join(artifacts_dir, "wikipedia_names_2gram_counts.pickle")
    with open(path, "rb") as f:
        # Trusted artifact within repo context; pickle acceptable here.
        count_dict = pickle.load(f)
    if "<UNK>" not in count_dict:
        raise ValueError("Prior counts pickle must include '<UNK>' entry")
    prior_counts = CountDict(count_dict["<UNK>"], count_dict)
    return prior_counts


def compute_deltas_ngram(pairs: list[tuple[Entity, Entity]], surrogate: NameBigram) -> np.ndarray:
    """Compute linking deltas using the n-gram surrogate.

    For each pair (A, B) we compute:
        delta = log p(A U B) - (log p(A) + log p(B))
    where p(.) is the marginal likelihood under the Dirichlet-multinomial bigram model.
    """

    def log_evidence(entity: Entity) -> float:
        names = entity.properties.get("name", [])
        counts = get_ngram_counts(names, surrogate.n) if names else get_ngram_counts([""], surrogate.n)
        return float(surrogate._evidence(None, counts))

    deltas = []
    for left, right in pairs:
        merged_entity = Entity.merge([left, right])
        log_A = log_evidence(left)
        log_B = log_evidence(right)
        log_AB = log_evidence(merged_entity)
        deltas.append(log_AB - (log_A + log_B))

    return np.asarray(deltas, dtype=np.float64)


def main(argv: list[str] | None = None) -> None:
    """Entry point for n-gram linking evaluation."""
    args = build_argparser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Arguments: %s", args)

    BIGRAM_N = 2
    if args.n != BIGRAM_N:  # only bigram currently supported
        raise NotImplementedError("Only bigram (n=2) supported.")

    benchmark = mskebab.get_default_benchmark()
    task_instance = benchmark.tasks_by_name[args.task_instance]
    logging.info("Loaded task instance %s", args.task_instance)

    pairs = _load_linking_pairs(task_instance)
    logging.info("Loaded %d pairs from task instance %s", len(pairs), args.task_instance)

    prior_counts = _load_ngram_prior(args.artifacts)
    logging.info("Loaded prior n-gram counts")

    surrogate = NameBigram(args.prior_scale, prior_counts)

    start = pc()
    deltas = compute_deltas_ngram(pairs, surrogate=surrogate)
    end = pc()

    elapsed = float(end - start)
    logging.info("Time (s): %.4f", elapsed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savetxt(args.out, deltas)

    metrics = task_instance.evaluate(Path(args.out))
    metrics["Total Time (s)"] = elapsed
    metrics["Candidates per Second (C/s)"] = len(pairs) / elapsed if elapsed > 0 else float("inf")

    for key, val in metrics.items():
        try:
            logging.info("%s: %.6f", key, float(val))
            print(f"{key}: {float(val):.6f}")
        except (TypeError, ValueError):
            logging.info("%s: %s", key, val)
            print(f"{key}: {val}")


if __name__ == "__main__":  # pragma: no cover
    main()
