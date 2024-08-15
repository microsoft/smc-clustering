# Licensed under the MIT license.

import numpy as np
import torch


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
        return Cluster(torch.cat([self.data, other.data]), self.hash.union(other.hash))


class Clusterer:
    def __init__(self, data, score_fn, link_threshold=0, cluster_batch_size=16):
        self.data = data
        self.score_fn = score_fn
        self.link_threshold = link_threshold
        self.cluster_batch_size = cluster_batch_size
        self.clusters = [Cluster(d) for d in data]
        self.score_cache = {}

    def compute_scores(self, clusters, force_recompute=False):
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

        # prepare the data and masks
        max_size = max([c.size for c in clusters])
        data, masks, hashes = [], [], []
        for cluster in clusters:
            if cluster.hash in self.score_cache:
                continue

            hashes.append(cluster.hash)

            # prepare masks
            mask = torch.zeros(max_size, dtype=torch.bool)
            mask[: cluster.size] = True
            masks.append(mask)

            # append nans onto data to make it the same size
            padding = torch.zeros(max_size - cluster.size, cluster.data.shape[1]).fill_(float('nan'))
            data.append(torch.cat([cluster.data, padding], dim=0))

        # compute a batch of scores, if any to compute
        if len(data) > 0:
            scores = self.score_fn(torch.stack(data), torch.stack(masks))

            # hash the scores so we don't need to recompute later
            for score, hash_ in zip(scores, hashes):
                self.score_cache[hash_] = score

        # retrieve the scores from the cache
        for cluster in clusters:
            cluster.score = self.score_cache[cluster.hash]

    def generate_batch_ids(self):
        # select a batch at random
        indices = np.arange(len(self.clusters))
        batch_size = min(self.cluster_batch_size, len(self.clusters))
        batch_indices = np.random.choice(indices, batch_size, replace=False)
        unique_pairs = [(i, j) for i in batch_indices for j in batch_indices if i < j]
        return batch_indices, unique_pairs

    def cluster(self, max_iter=100, verbose=True, callback=None):

        for iteration in range(max_iter):

            inds, ijs = self.generate_batch_ids()
            cluster_batch = [c for i, c in enumerate(self.clusters) if i in inds]
            proposed_clusters = [self.clusters[i].merge(self.clusters[j]) for i, j in ijs]

            # assign the score to each cluster
            self.compute_scores(cluster_batch + proposed_clusters)

            # compute linking scores
            linking_scores = [
                pc.score - self.clusters[i].score - self.clusters[j].score for (i, j), pc in zip(ijs, proposed_clusters)
            ]
            linking_scores = torch.stack(linking_scores)

            # find the best pair in the batch, execute merge if threshold reached
            best_pair = ijs[torch.argmax(linking_scores).item()]
            best_score = linking_scores.max()
            if best_score > self.link_threshold:
                if verbose:
                    print(f'iteration {iteration}, linking score: {best_score:.4f}')
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
                    print(f'iteration {iteration}, no links found!, best merging score: {best_score:.4f}')
                if len(self.clusters) <= self.cluster_batch_size:
                    # in this case the batch size is larger than the number of clusters, so we've exhausted all possible links
                    print('potential links exhausted, exiting')
                    break

            if callback is not None:
                callback(self.clusters, iteration)
