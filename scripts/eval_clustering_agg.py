# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Evaluate AGG + JSON-LM clustering performance in MS-KeBAB.

Example usage:
    uv run scripts/eval_clustering_agg.py --config ./scripts/config/benchmark_conf.json --artifacts ./data/artifacts --ckpt ./data/artifacts/best.ckpt --offset 6.2146 --task_instance Clustering-REBEL-50
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pickle
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np
import torch
from kebab import mskebab
from kebab.contracts.entity import Entity
from tokenizers import Tokenizer as HFTokenizer
from torch import nn

from smc_clustering.clustering.agglomerative import Clusterer
from smc_clustering.clustering.cluster import Cluster, DirichletProcess
from smc_clustering.clustering.surrogate_models import Bigram, CountDict, get_ngram_counts
from smc_clustering.jsonlm.models.scoring import score_entities_batched
from smc_clustering.jsonlm.models.transformer import TransformerConfig, TransformerLM
from smc_clustering.jsonlm.tokenization.tokenizer import JsonLMTokenizer
from smc_clustering.jsonlm.tokenization.vocab import Vocabulary


class ListWrapper:
    """Wrapper that exposes entity lists through the clustering API."""

    def __init__(self, data: list[Entity]) -> None:
        """Initialize ListWrapper with the provided entity list."""
        self.data = data

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of the wrapped data."""
        return (len(self.data),)

    def __getitem__(self, row_ids: int | slice | np.ndarray) -> list[Entity]:
        """Return items selected from the wrapped entity list."""
        if type(row_ids) is int:
            return [self.data[row_ids]]
        if type(row_ids) is slice:
            return self.data[row_ids]
        return [self.data[idx] for idx in row_ids]


class NameBigram(Bigram):
    """Retrieves name property for use in the bigram model."""

    def __init__(self, prior_scale: float, prior_counts: CountDict) -> None:
        """Initialize NameBigram with the given prior counts."""
        super().__init__(prior_scale, prior_counts)

    def post_predictive(
        self, obs: Entity | list[Entity], n: np.ndarray, summary: list[collections.Counter[str]]
    ) -> np.ndarray:
        """Score an entity or entity list using name bigrams."""
        name = obs[0].properties["name"] if type(obs) is list else obs.properties["name"]

        return super().post_predictive(name, n, summary)


class NameBigramCluster(Cluster):
    """Cluster subclass with summary statistics for an n-gram model, looks at name property for counts."""

    def __init__(
        self,
        data_ids: frozenset[int],
        n: int = 2,
        counts: collections.Counter[str] | None = None,
        data: Entity | list[Entity] | None = None,
    ) -> None:
        """Initialize NameBigramCluster with cached name n-gram counts."""
        super().__init__(data_ids)
        self.n = n
        if counts is not None:
            self.counts = counts
        elif data is not None:
            if isinstance(data, Entity):
                data = [data]
            self.counts = get_ngram_counts([entity.properties["name"] for entity in data], self.n)
        else:
            self.counts = collections.Counter()

    @property
    def summary(self) -> collections.Counter[str]:
        """Return the cached name n-gram counts."""
        return self.counts

    def merge_point(self, data_id: int, data: list[Entity]) -> NameBigramCluster:
        """Return a new cluster with updated name n-gram counts."""
        new_counts = self.counts + get_ngram_counts(
            [entity.properties["name"] for entity in data], self.n
        )
        return NameBigramCluster(self.data.union({data_id}), self.n, counts=new_counts)


def _load_artifacts(artifacts_dir: str) -> tuple[JsonLMTokenizer, TransformerConfig]:
    """Load tokenizer and model config from artifacts directory."""
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
    """Load model from checkpoint."""
    model = TransformerLM(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if "state_dict" in ckpt:
        state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] Missing keys: {missing}, Unexpected keys: {unexpected}")
    return model


def score_entities(
    _rng: jax.Array,
    clusters: list[list[Entity]],
    model: nn.Module,
    tokenizer: JsonLMTokenizer,
    batch_size: int = 256,
    offset: float = 0.0,
    max_cluster_size: int = 100,
) -> list[float]:
    """Estimate cluster log-likelihoods."""
    # entities = [Entity.merge(cluster) if len(cluster) > 1 else cluster[0] for cluster in clusters]
    # entities = [e.properties for e in entities]

    entities = []
    for cluster in clusters:
        if len(cluster) < max_cluster_size:
            entities.append(cluster)
        else:
            entities.append(
                cluster[: int(max_cluster_size // 2)] + cluster[-int(max_cluster_size // 2) :]
            )
            logging.warning(
                f"Cluster too large: {len(cluster)}. Entity 1 = {cluster[0].properties['name']}, ... Entity N = {cluster[-1].properties['name']}"
            )

    entities = [[e.properties for e in cluster] for cluster in entities]
    scores = score_entities_batched(
        entities, model=model, tokenizer=tokenizer, offset=offset, batch_size=batch_size
    )

    return np.array(scores)


def build_argparser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description="Evaluate JSON-LM in MS-KeBAB Linking task.")
    p.add_argument(
        "--config", default="./config/benchmark_conf.json", help="Path to benchmark config file"
    )
    p.add_argument(
        "--artifacts", default="./artifacts", help="Directory with vocab.json, bpe.json, config.json"
    )
    p.add_argument(
        "--ckpt", default="./artifacts/last.ckpt", help="Checkpoint (.ckpt or raw state_dict)"
    )
    p.add_argument(
        "--surrogate",
        default="./data/artifacts/rebel_2gram_counts.pickle",
        help="Surrogate model parameters",
    )
    p.add_argument(
        "--task_instance",
        type=str,
        default="Clustering-REBEL-Small",
        help="MS-KeBAB Linking task instance",
    )
    p.add_argument("--out", default="./output", help="Output path for results")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for processing")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--offset",
        type=float,
        default=0,
        help="Additive logit offset used when scoring entities (was previously hardcoded)",
    )

    # Clustering arguments
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--prior_scale", type=float, default=25)
    p.add_argument(
        "--cluster_batch_size",
        type=int,
        default=None,
        help="Number of clusters sampled at each iteration (if None, compute all cluster merges)",
    )
    p.add_argument(
        "--increment", type=int, default=10, help="Number of iterations to do between each evaluation"
    )
    p.add_argument("--max_iter", type=int, default=np.inf, help="Maximum number of iterations")
    p.add_argument("--max_t", type=int, default=np.inf, help="Maximum runtime")
    p.add_argument(
        "--stop_cond",
        type=float,
        default=100,
        help="Convergence condition - stop if there is no change to the best clustering for this many iterations",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    """Run the evaluation."""
    args = build_argparser().parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Arguments: {args}")

    # Set up the task and get the data
    benchmark = mskebab.Benchmark(Path(args.config))
    task_instance = benchmark.tasks_by_name[args.task_instance]
    logging.info(f"Loaded task instance {args.task_instance}")

    data = [e for e, _ in task_instance.read_items()]
    logging.info(f"Loaded {len(data)} entities for clustering")

    # shuffle the dataset, compute the inverse shuffling
    shuffle_rng, rng = jax.random.split(rng)
    shuffled_idx = jax.random.permutation(shuffle_rng, len(data))
    unshuffled_idx = np.zeros_like(shuffled_idx)
    unshuffled_idx[shuffled_idx] = np.arange(len(data))

    data = ListWrapper([data[i] for i in shuffled_idx])

    # Set up the JSON-LM model
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    logging.info(f"Using device: {device}")

    tok, cfg = _load_artifacts(args.artifacts)
    logging.info(f"Loaded tokenizer and config from {args.artifacts}")

    model = _load_model(args.ckpt, cfg, device=device)
    logging.info(f"Loaded model from {args.ckpt}")

    prior = DirichletProcess(args.alpha)

    batched_score_eval = partial(
        score_entities, model=model, tokenizer=tok, batch_size=args.batch_size, offset=args.offset
    )

    if args.cluster_batch_size is None:
        cluster_batch_size = len(data.data)
        experiment_name = f"s{args.seed}_alpha{args.alpha}_agg_full"

    else:
        cluster_batch_size = args.cluster_batch_size
        experiment_name = f"s{args.seed}_alpha{args.alpha}_agg_batch{cluster_batch_size}"

    clusterer = Clusterer(
        data=data, score_fn=batched_score_eval, prior=prior, cluster_batch_size=cluster_batch_size
    )

    # Run clustering
    best = -np.inf
    t = 0
    total_iters = 0
    iters_since_change = 0
    while t < args.max_t and total_iters < args.max_iter and iters_since_change < args.stop_cond:
        rng, cl_rng = jax.random.split(rng)
        start = time.time()
        clusterer.cluster(cl_rng, max_iter=args.increment)
        t += time.time() - start
        total_iters += args.increment

        if clusterer.objective > best:
            best = clusterer.objective
            iters_since_change = 0
        else:
            iters_since_change += args.increment

        # save clustering and metrics
        ll = sum([clusterer.score_cache[cl.hash] for cl in clusterer.clusters])
        lp = clusterer.objective

        clustering = clusterer.list_cluster_labels()
        clustering = [
            clustering[idx] for idx in unshuffled_idx
        ]  # cluster labels for the data in the original order

        clustering_path = out_dir / f"{experiment_name}_it{total_iters}_clustering"
        with clustering_path.open("w", encoding="utf-8") as f:
            for cluster in clustering:
                f.write(str(cluster) + "\n")

        metrics = task_instance.evaluate(clustering_path)
        metrics["t"] = t
        metrics["total_evals"] = len(clusterer.score_cache)
        metrics["LL"] = ll
        metrics["LP"] = lp

        print(f"Iteration {total_iters}, time {t}")
        for key, val in metrics.items():
            logging.info(f"{key}: {val:.6f}")
            print(f"{key}: {val:.6f}")

        with (out_dir / f"{experiment_name}_it{total_iters}_metrics.pickle").open("wb") as f:
            pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save final clustering and metrics
    ll = sum([clusterer.score_cache[cl.hash] for cl in clusterer.clusters])
    lp = clusterer.objective

    clustering = clusterer.list_cluster_labels()
    clustering = [
        clustering[idx] for idx in unshuffled_idx
    ]  # cluster labels for the data in the original order

    final_clustering_path = out_dir / f"{experiment_name}_final_clustering"
    with final_clustering_path.open("w", encoding="utf-8") as f:
        for cluster in clustering:
            f.write(str(cluster) + "\n")

    metrics = task_instance.evaluate(final_clustering_path)
    metrics["t"] = t
    metrics["total_evals"] = len(clusterer.score_cache)
    metrics["LL"] = ll
    metrics["LP"] = lp

    with (out_dir / f"{experiment_name}_final_metrics.pickle").open("wb") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
