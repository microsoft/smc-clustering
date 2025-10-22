"""
Evaluate MCMC + JSON-LM clustering performance in MS-KeBAB.

Example usage:
    uv run scripts/eval_clustering_mcmc.py --config ./scripts/config/benchmark_conf.json --artifacts ./data_vm/artifacts --ckpt ./data_vm/artifacts/best.ckpt --offset 6.2146 --task_instance Clustering-REBEL-50
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
from kebab import mskebab
from kebab.contracts.entity import Entity
from tokenizers import Tokenizer as HFTokenizer
from torch import nn

from diffusion_linking.clustering import Cluster
from diffusion_linking.mcmc_clustering import GibbsClusterer
from diffusion_linking.smc_clustering import DirichletProcess
from diffusion_linking.surrogate_models import Bigram, CountDict, get_ngram_counts
from jsonlm.models.scoring import score_entities_batched
from jsonlm.models.transformer import TransformerConfig, TransformerLM
from jsonlm.tokenization.tokenizer import JsonLMTokenizer
from jsonlm.tokenization.vocab import Vocabulary


class ListWrapper:
    # Allows easier retrieval of cluster data from lists
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        return (len(self.data),)

    def __getitem__(self, row_ids):
        if type(row_ids) is int:
            return [self.data[row_ids]]
        if type(row_ids) is slice:
            return self.data[row_ids]
        return [self.data[idx] for idx in row_ids]


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


class NameBigramCluster(Cluster):
    """Cluster subclass with summary statistics for an n-gram model, looks at name property for counts"""

    def __init__(self, data_ids, n=2, counts=None, data=None):
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
    def summary(self):
        return self.counts

    def merge_point(self, data_id, data):
        new_counts = self.counts + get_ngram_counts([entity.properties["name"] for entity in data], self.n)
        return NameBigramCluster(self.data.union({data_id}), self.n, counts=new_counts)


def _load_artifacts(artifacts_dir: str) -> tuple[JsonLMTokenizer, TransformerConfig]:
    """Load tokenizer and model config from artifacts directory."""
    vocab_path = os.path.join(artifacts_dir, "vocab.json")
    bpe_path = os.path.join(artifacts_dir, "bpe.json")
    cfg_path = os.path.join(artifacts_dir, "config.json")

    with open(vocab_path, encoding="utf-8") as f:
        tokens = json.load(f)
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError("vocab.json must be a JSON list of token strings.")
    vocab = Vocabulary.from_tokens(tokens)

    bpe = HFTokenizer.from_file(bpe_path)
    tok = JsonLMTokenizer(vocabulary=vocab, bpe=bpe, specials_size=len(vocab), bpe_size=bpe.get_vocab_size())

    with open(cfg_path, encoding="utf-8") as f:
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
    rng: Any,
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
            entities.append(cluster[: int(max_cluster_size // 2)] + cluster[-int(max_cluster_size // 2) :])
            logging.warning(
                f"Cluster too large: {len(cluster)}. Entity 1 = {cluster[0].properties['name']}, ... Entity N = {cluster[-1].properties['name']}"
            )

    entities = [[e.properties for e in cluster] for cluster in entities]
    scores = score_entities_batched(entities, model=model, tokenizer=tokenizer, offset=offset, batch_size=batch_size)

    return np.array(scores)


def build_argparser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description="Evaluate JSON-LM in MS-KeBAB Linking task.")
    p.add_argument("--config", default="./config/benchmark_conf.json", help="Path to benchmark config file")
    p.add_argument("--artifacts", default="./artifacts", help="Directory with vocab.json, bpe.json, config.json")
    p.add_argument("--ckpt", default="./artifacts/last.ckpt", help="Checkpoint (.ckpt or raw state_dict)")
    p.add_argument(
        "--surrogate", default="./data_vm/artifacts/rebel_2gram_counts.pickle", help="Surrogate model parameters"
    )
    p.add_argument("--task_instance", type=str, default="Clustering-REBEL-Small", help="MS-KeBAB Linking task instance")
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
    p.add_argument("--increment", type=int, default=10, help="Number of iterations to do between each evaluation")
    p.add_argument("--max_iter", type=int, default=np.inf, help="Maximum number of iterations")
    p.add_argument("--max_t", type=int, default=np.inf, help="Maximum runtime")
    p.add_argument(
        "--stop_cond",
        type=float,
        default=500,
        help="Convergence condition - stop if there is no change to the best clustering for this many iterations",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    """Run the evaluation."""
    args = build_argparser().parse_args(argv)

    rng = jax.random.PRNGKey(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Arguments: {args}")

    # Set up the task and get the data
    benchmark = mskebab.Benchmark(Path(args.config))
    task_instance = benchmark.tasks_by_name[args.task_instance]
    logging.info(f"Loaded task instance {args.task_instance}")

    data = list(e for e, _ in task_instance.read_items())
    logging.info(f"Loaded {len(data)} entities for clustering")

    # shuffle the dataset, compute the inverse shuffling
    shuffle_rng, rng = jax.random.split(rng)
    shuffled_idx = jax.random.permutation(shuffle_rng, len(data))
    unshuffled_idx = np.zeros_like(shuffled_idx)
    unshuffled_idx[shuffled_idx] = np.arange(len(data))

    data = ListWrapper([data[i] for i in shuffled_idx])

    # Set up the JSON-LM model
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    logging.info(f"Using device: {device}")

    tok, cfg = _load_artifacts(args.artifacts)
    logging.info(f"Loaded tokenizer and config from {args.artifacts}")

    model = _load_model(args.ckpt, cfg, device=device)
    logging.info(f"Loaded model from {args.ckpt}")

    # Set up the SMC clustering components
    with open(args.surrogate, "rb") as f:
        count_dict = pickle.load(f)

    logging.info(f"Loaded n-gram counts: {len(count_dict)} elements")

    prior_counts = CountDict(count_dict["<UNK>"], count_dict)
    prior = DirichletProcess(args.alpha)
    surrogate = NameBigram(args.prior_scale, prior_counts)

    batched_score_eval = partial(
        score_entities, model=model, tokenizer=tok, batch_size=args.batch_size, offset=args.offset
    )

    clusterer = GibbsClusterer(
        data=data,
        score_fn=batched_score_eval,
        prior=prior,
        ClusterClass=NameBigramCluster,
        # data=data, score_fn=batched_score_eval, prior=prior, surrogate=surrogate, ClusterClass=NameBigramCluster
    )
    experiment_name = f"s{args.seed}_alpha{args.alpha}_mcmc"

    os.makedirs(args.out, exist_ok=True)

    # Run clustering
    best = -np.inf
    t = 0
    total_iters = 0
    iters_since_change = 0
    while t < args.max_t and total_iters < args.max_iter and iters_since_change < args.stop_cond:
        rng, cl_rng = jax.random.split(rng)
        start = time.time()
        clusterer.cluster(cl_rng, sweeps=args.increment)
        t += time.time() - start
        total_iters += args.increment

        if clusterer.best_weight > best:
            best = clusterer.best_weight
            iters_since_change = 0
        else:
            iters_since_change += args.increment

        # save clustering and metrics
        ll = sum([clusterer.score_cache[cl.hash] for cl in clusterer.best])
        lp = clusterer.best_weight

        clustering = clusterer.list_cluster_labels()
        clustering = [clustering[idx] for idx in unshuffled_idx]  # cluster labels for the data in the original order

        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, experiment_name + f"_it{total_iters}_clustering"), "w", encoding="utf-8") as f:
            for cluster in clustering:
                f.write(str(cluster) + "\n")

        metrics = task_instance.evaluate(Path(os.path.join(args.out, experiment_name + f"_it{total_iters}_clustering")))
        metrics["t"] = t
        metrics["total_evals"] = len(clusterer.score_cache)
        metrics["LL"] = ll
        metrics["LP"] = lp

        print(f"Iteration {total_iters}, time {t}")
        for key, val in metrics.items():
            logging.info(f"{key}: {val:.6f}")
            print(f"{key}: {val:.6f}")

        with open(os.path.join(args.out, experiment_name + f"_it{total_iters}_metrics.pickle"), "wb") as f:
            pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save final clustering and metrics
    ll = sum([clusterer.score_cache[cl.hash] for cl in clusterer.best])
    lp = clusterer.best_weight

    clustering = clusterer.list_cluster_labels()
    clustering = [clustering[idx] for idx in unshuffled_idx]  # cluster labels for the data in the original order

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, experiment_name + "_final_clustering"), "w", encoding="utf-8") as f:
        for cluster in clustering:
            f.write(str(cluster) + "\n")

    print("METRICS")

    metrics = task_instance.evaluate(Path(os.path.join(args.out, experiment_name + "_final_clustering")))
    metrics["t"] = t
    metrics["total_evals"] = len(clusterer.score_cache)
    metrics["LL"] = ll
    metrics["LP"] = lp

    with open(os.path.join(args.out, experiment_name + "_final_metrics.pickle"), "wb") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
