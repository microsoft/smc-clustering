# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Agglomerative clustering utilities for scored entity clusters.

This module maintains cluster scores in a cache and repeatedly merges the best-scoring pair from a sampled batch of candidate merges.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from smc_clustering.clustering.cluster import Cluster


class Clusterer:
    """Agglomerative clusterer over cached cluster scores."""

    def __init__(
        self,
        data: np.ndarray,
        score_fn: Callable,
        link_threshold: float = 0,
        cluster_batch_size: int = 16,
        prior: Callable | None = None,
        score_cache: dict[int, float] | None = None,
    ):
        """Initialize Clusterer with the given parameters."""
        self.data = data
        self.score_fn = score_fn
        self.link_threshold = link_threshold
        self.cluster_batch_size = cluster_batch_size
        self.clusters = [Cluster({d}) for d in range(data.shape[0])]
        self.score_cache = {} if score_cache is None else score_cache
        self.prior = prior if prior is not None else lambda _: 0
        self.objective = None

    def compute_scores(
        self, rng: jax.Array, clusters: list[frozenset[int]], force_recompute: bool = False
    ) -> int:
        """Compute scores for a list of clusters.

        The cache avoids recomputing scores for clusters that have already been evaluated.
        """
        # remove the score from the cache if we're forcing a recompute
        if force_recompute:
            [self.score_cache.pop(hash(cluster), None) for cluster in clusters]

        compute_clusters = [cluster for cluster in clusters if hash(cluster) not in self.score_cache]
        if len(compute_clusters) == 0:
            return 0

        hashes = [hash(cluster) for cluster in compute_clusters]
        scores = self.score_fn(
            rng, [self.data[np.fromiter(cluster, dtype=np.int64)] for cluster in compute_clusters]
        )
        for score, hash_ in zip(scores, hashes, strict=False):
            self.score_cache[hash_] = score

        return len(compute_clusters)

    def generate_batch_ids(self, rng: jax.Array) -> tuple[jax.Array, list[tuple[Any, Any]]]:
        # select a batch at random
        """Sample a batch of cluster indices and their unique pairs."""
        indices = np.arange(len(self.clusters))
        batch_size = min(self.cluster_batch_size, len(self.clusters))
        batch_indices = jax.random.choice(rng, indices, (batch_size,), replace=False)
        unique_pairs = [(i, j) for i in batch_indices for j in batch_indices if i < j]
        return batch_indices, unique_pairs

    def cluster(
        self, rng: jax.Array, max_iter: int = 100, callback: Callable | None = None
    ) -> tuple[list[int], bool]:
        """Run the clustering procedure."""
        n_evals = []
        if self.objective is None:
            rng, compute_rng = jax.random.split(rng)
            model_evals = self.compute_scores(compute_rng, [cl.data for cl in self.clusters])
            n_evals.append(model_evals)
            self.objective = sum(
                [self.prior(np.array([cl.size])) + self.score_cache[cl.hash] for cl in self.clusters]
            )

        for iteration in (pbar := tqdm(range(max_iter))):
            rng, batch_rng = jax.random.split(rng)
            inds, ijs = self.generate_batch_ids(batch_rng)
            cluster_batch = [c.data for i, c in enumerate(self.clusters) if i in inds]
            proposed_clusters = [self.clusters[i].merge(self.clusters[j]).data for i, j in ijs]

            # assign the score to each cluster
            rng, score_rng = jax.random.split(rng)
            model_evals = self.compute_scores(score_rng, cluster_batch + proposed_clusters)
            n_evals.append(model_evals)

            # compute linking scores
            linking_scores = [
                self.score_cache[hash(pc)]
                + self.prior(np.array([len(pc)]))
                - self.prior(np.array([self.clusters[i].size, self.clusters[j].size]))
                - self.score_cache[self.clusters[i].hash]
                - self.score_cache[self.clusters[j].hash]
                for (i, j), pc in zip(ijs, proposed_clusters, strict=False)
            ]
            linking_scores = np.stack(linking_scores)

            # find the best pair in the batch, execute merge if threshold reached
            best_pair = ijs[np.argmax(linking_scores).item()]
            best_score = linking_scores.max()

            if best_score > self.link_threshold:
                i, j = best_pair

                # remove cluster i and j
                ci = self.clusters[i]
                cj = self.clusters[j]
                self.clusters.remove(ci)
                self.clusters.remove(cj)

                # append new cluster
                self.clusters.append(ci.merge(cj))

                self.objective = sum(
                    [self.prior(np.array([cl.size])) + self.score_cache[cl.hash] for cl in self.clusters]
                )
                pbar.set_postfix(
                    {"Best score": f"{best_score:.4g}", "Objective": f"{self.objective:.4g}"}
                )

            elif len(self.clusters) <= self.cluster_batch_size:
                # in this case the batch size is larger than the number of clusters, so we've exhausted all possible links
                print("potential links exhausted, exiting")
                return n_evals, True

            if callback is not None:
                callback(self.clusters, iteration)

        return n_evals, False

    def summary(self, print_cluster_data: bool = False):
        # Print out summary of clustering
        """Print a summary of the current clustering state."""
        clusters = sorted(self.clusters, key=lambda c: c.size, reverse=True)
        print(f"{len(clusters)} clusters, {self.data.shape[0]} points, {[c.size for c in clusters]}")
        if print_cluster_data:
            for c in clusters:
                print(f"LL:{self.score_cache[c.hash]:.2g}, ", self.data[c.ids])
            print()

    def list_cluster_labels(self) -> list[str]:
        """Return a list of cluster identifiers for each observation."""
        cluster_lookup = {}
        for cl in self.clusters:
            for i in cl.ids:
                cluster_lookup[i.item()] = str(cl.hash)
        return [cluster_lookup[i] for i in sorted(cluster_lookup.keys())]


def plot_callback(clusterer: Clusterer) -> None:
    # sort clusters by size
    """Plot the current clustering assignment."""
    clusters = sorted(clusterer.clusters, key=lambda c: c.size, reverse=True)
    plt.figure()
    for c in clusters:
        if c.size > 1:
            plt.plot(clusterer.data[c.ids, 0], clusterer.data[c.ids, 1], "o", markersize=3)
        else:
            plt.plot(clusterer.data[c.ids, 0], clusterer.data[c.ids, 1], "ks", markersize=3, alpha=0.5)
    plt.axis("equal")
    plt.show()
