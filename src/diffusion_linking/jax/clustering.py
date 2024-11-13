# Licensed under the MIT license.
from math import ceil

import jax
import jax.numpy as jnp


class Cluster:
    """
    A simple class to hold a cluster of points.
    A torch tensor of shape (cluster_size, dim) is used to store the data.
    A hash is used to identify the cluster, and is computed as a frozenset of the ids the fragments. This allows
    caching of cluster scores.
    """

    def __init__(self, data, hash_=None):
        self.data = data
        self._hash = hash_
        self.score = None

    @property
    def hash(self):
        if self._hash is None:
            return frozenset([id(self)])
        return self._hash

    @property
    def size(self):
        return self.data.shape[0]

    def merge(self, other):
        return Cluster(jnp.concat([self.data, other.data]), self.hash.union(other.hash))


class Clusterer:
    def __init__(self, data, score_fn, link_threshold=0, cluster_batch_size=16, batch_shape=(20, 20)):
        self.data = data
        self.score_fn = score_fn
        self.link_threshold = link_threshold
        self.cluster_batch_size = cluster_batch_size
        self.clusters = [Cluster(d) for d in data]
        self.score_cache = {}

        self.data_dim = data[0].shape[-1]
        self.batch_shape = batch_shape

    def compute_scores(self, rng, clusters, force_recompute=False):
        """
        For a list of clusters, compute the score for each cluster.

        We use a cache to avoid recomputing scores for clusters that have already been computed.

        The score function is assumed to take a batch of data and a batch of masks,
        and return a batch of scores. The shape of the data is (batch_size, max_cluster_size, dim),
        the shape of the mask is (batch_size, max_cluster_size), and the shape of the scores is (batch_size,).
        """
        # remove the score from the cache if we're forcing a recompute
        if force_recompute:
            [self.score_cache.pop(cluster.hash, None) for cluster in clusters]

        compute_clusters = [cluster for cluster in clusters if cluster.hash not in self.score_cache]
        if len(compute_clusters) == 0:
            for cluster in clusters:
                cluster.score = self.score_cache[cluster.hash]
            return
        # split clusters into batches and pad last batch (avoids recompilations)
        n_batches = ceil(len(compute_clusters) / self.batch_shape[0])
        compute_clusters = [
            compute_clusters[(self.batch_shape[0] * i) : min(self.batch_shape[0] * (i + 1), len(compute_clusters))]
            for i in range(n_batches)
        ]
        compute_clusters[-1] += [Cluster(jnp.full((0, self.data_dim), float("nan")))] * (
            self.batch_shape[0] - len(compute_clusters[-1])
        )

        for cluster_batch in compute_clusters:
            max_size = (
                self.batch_shape[1]
                + max([0, ceil((max([c.size for c in cluster_batch]) - self.batch_shape[1]) / 8)]) * 8
            )
            # prepare the data and masks
            data, masks, hashes = [], [], []
            for cluster in cluster_batch:
                hashes.append(cluster.hash)

                # prepare masks
                mask = jnp.concat([jnp.ones((cluster.size,)), jnp.zeros(max_size - cluster.size)])
                masks.append(mask)

                # append nans onto data to make it the same size
                padding = jnp.full((max_size - cluster.size, cluster.data.shape[1]), float("nan"))
                data.append(jnp.concat([cluster.data, padding], axis=0))

            # compute a batch of scores, if any to compute
            if len(data) > 0:
                scores = self.score_fn(rng, jnp.stack(data), jnp.stack(masks))

                # hash the scores so we don't need to recompute later
                for score, hash_ in zip(scores, hashes):
                    self.score_cache[hash_] = score

        # retrieve the scores from the cache
        for cluster in clusters:
            cluster.score = self.score_cache[cluster.hash]

    def generate_batch_ids(self, rng):
        # select a batch at random
        indices = jnp.arange(len(self.clusters))
        batch_size = min(self.cluster_batch_size, len(self.clusters))
        batch_indices = jax.random.choice(rng, indices, (batch_size,), replace=False)
        unique_pairs = [(i, j) for i in batch_indices for j in batch_indices if i < j]
        return batch_indices, unique_pairs

    def cluster(self, rng, max_iter=100, verbose=True, callback=None):
        for iteration in range(max_iter):
            rng, batch_rng = jax.random.split(rng)
            inds, ijs = self.generate_batch_ids(batch_rng)
            cluster_batch = [c for i, c in enumerate(self.clusters) if i in inds]
            proposed_clusters = [self.clusters[i].merge(self.clusters[j]) for i, j in ijs]

            # assign the score to each cluster
            rng, score_rng = jax.random.split(rng)
            self.compute_scores(score_rng, cluster_batch + proposed_clusters)

            # compute linking scores
            linking_scores = [
                pc.score - self.clusters[i].score - self.clusters[j].score for (i, j), pc in zip(ijs, proposed_clusters)
            ]
            linking_scores = jnp.stack(linking_scores)

            # find the best pair in the batch, execute merge if threshold reached
            best_pair = ijs[jnp.argmax(linking_scores).item()]
            best_score = linking_scores.max()
            if best_score > self.link_threshold:
                if verbose:
                    print(f"iteration {iteration}, linking score: {best_score:.4f}")
                i, j = best_pair

                # remove cluster i and j
                ci = self.clusters[i]
                cj = self.clusters[j]
                self.clusters.remove(ci)
                self.clusters.remove(cj)

                # append new cluster
                self.clusters.append(ci.merge(cj))

            else:
                if verbose:
                    print(f"iteration {iteration}, no links found!, best merging score: {best_score:.4f}")
                if len(self.clusters) <= self.cluster_batch_size:
                    # in this case the batch size is larger than the number of clusters, so we've exhausted all possible links
                    print("potential links exhausted, exiting")
                    break

            if callback is not None:
                callback(self.clusters, iteration)
