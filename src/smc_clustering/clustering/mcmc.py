# Licensed under the MIT license.
import logging

import jax
import numpy as np
import scipy
from tqdm import tqdm

from smc_clustering.clustering.cluster import Cluster


logger = logging.getLogger(__name__)


class GibbsClusterer:
    def __init__(self, data, score_fn, prior, surrogate=None, ClusterClass=Cluster, score_cache=None):
        self.data = data
        self.score_fn = score_fn
        self.prior = prior
        self.surrogate = surrogate
        self.ClusterClass = ClusterClass
        self.clusters = [ClusterClass({d}, data=self.data[d]) for d in range(data.shape[0])]
        self.logpost = None
        self.best_clustering = self.clusters.copy()
        self.best_logpost = None
        self.score_cache = {} if score_cache is None else score_cache

    def compute_scores(self, rng, clusters, force_recompute=False):
        """
        For a list of clusters, compute the score for each cluster
        We use a cache to avoid recomputing scores for clusters that have already been computed.

        """
        # remove the score from the cache if we're forcing a recompute
        if force_recompute:
            [self.score_cache.pop(hash(cluster), None) for cluster in clusters]

        compute_clusters = [cluster for cluster in clusters if hash(cluster) not in self.score_cache]
        if len(compute_clusters) == 0:
            return 0

        hashes = [hash(cluster) for cluster in compute_clusters]
        scores = self.score_fn(rng, [self.data[np.fromiter(cluster, dtype=np.int64)] for cluster in compute_clusters])
        for score, hash_ in zip(scores, hashes, strict=False):
            self.score_cache[hash_] = score

        return len(compute_clusters)

    def update_exact(self, rng, i):
        """
        Gibbs update for assignment i
        """
        compute_clusters = []
        hashes = []
        weights = np.zeros(len(self.clusters) + 1)
        weights[-1] = self.prior.marginal(self.data.shape[0], 0).item()
        old_k = None
        for k, C_k in enumerate(self.clusters):
            N_k = C_k.size
            if i in C_k.data:
                hashes.append(C_k.hash)
                old_k = k
                if N_k > 1:
                    compute_clusters.append(C_k.data - {i})
                    weights[k] = self.prior.marginal(self.data.shape[0], N_k - 1).item()
                else:
                    weights = weights[:-1]
                    weights[k] = self.prior.marginal(self.data.shape[0], N_k - 1).item()
            else:
                hashes.append(hash(C_k.add(i)))
                compute_clusters.append(C_k.add(i))
                weights[k] = self.prior.marginal(self.data.shape[0], N_k).item() - self.score_cache[C_k.hash]

        rng, compute_rng = jax.random.split(rng)
        model_evals = self.compute_scores(compute_rng, compute_clusters)

        if len(weights) == len(hashes):
            # old cluster was a singleton
            weights += np.array([self.score_cache[h] for h in hashes])

        else:
            weights[:-1] += np.array([self.score_cache[h] for h in hashes])
            weights[old_k] -= self.score_cache[hash(self.clusters[old_k].data - {i})]
            weights[-1] += self.score_cache[hash(frozenset({i}))]

        rng, sample_rng = jax.random.split(rng)
        new_k = jax.random.choice(
            sample_rng, len(weights), (1,), p=np.exp(weights - scipy.special.logsumexp(weights))
        ).item()

        if new_k != old_k:
            # Update state
            if new_k < len(self.clusters):
                self.clusters[new_k] = self.clusters[new_k].merge_point(i, self.data[i])
            else:
                self.clusters.append(self.ClusterClass({i}, data=self.data[i]))
            self.logpost = sum([self.prior(np.array([cl.size])) + self.score_cache[cl.hash] for cl in self.clusters])

            if self.clusters[old_k].size > 1:
                self.clusters[old_k] = self.ClusterClass(self.clusters[old_k].data - {i})
            else:
                del self.clusters[old_k]

        return model_evals, 0

    def update_mh(self, rng, i):
        """
        Metropolis-within-Gibbs update, using the surrogate model as the proposal distribution
        """
        model_evals = 0
        summary_stats = []
        cluster_sizes = []
        weights = np.zeros(len(self.clusters) + 1)
        old_k = None
        for k, C_k in enumerate(self.clusters):
            N_k = C_k.size
            if i in C_k.data:
                old_k = k
                if N_k > 1:
                    new_data_ids = C_k.data - {i}
                    summary_stats.append(
                        self.ClusterClass(
                            new_data_ids, data=self.data[np.fromiter(new_data_ids, dtype=np.int64)]
                        ).summary
                    )
                    cluster_sizes.append(N_k - 1)
                    weights[k] = self.prior.marginal(self.data.shape[0], N_k - 1).item()
                else:
                    summary_stats.append(self.ClusterClass({}).summary)
                    cluster_sizes.append(N_k - 1)
                    weights = weights[:-1]
                    weights[k] = self.prior.marginal(self.data.shape[0], N_k - 1).item()
            else:
                summary_stats.append(C_k.summary)
                cluster_sizes.append(C_k.size)
                weights[k] = self.prior.marginal(self.data.shape[0], N_k).item()

        if len(weights) == len(self.clusters) + 1:
            # add in singleton assignment if not aleardy included
            summary_stats.append(self.ClusterClass({}).summary)
            cluster_sizes.append(0)
            weights[-1] = self.prior.marginal(self.data.shape[0], 0).item()

        sur_LL = self.surrogate.post_predictive(self.data[i], np.array(cluster_sizes), summary_stats)
        weights += sur_LL
        surrogate_evals = len(cluster_sizes)

        rng, sample_rng = jax.random.split(rng)
        new_k = jax.random.choice(
            sample_rng, len(weights), (1,), p=np.exp(weights - scipy.special.logsumexp(weights))
        ).item()

        model_evals = 0
        if new_k != old_k:
            # Metropolis-Hastings accept step
            compute_clusters = []

            if self.clusters[old_k].size > 1:
                old_cluster = self.clusters[old_k].data - {i}
                compute_clusters.append(old_cluster)
            else:
                old_cluster = None

            new_cluster = self.clusters[new_k].add(i) if new_k < len(self.clusters) else frozenset({i})
            compute_clusters.append(new_cluster)

            rng, compute_rng = jax.random.split(rng)
            model_evals = self.compute_scores(rng, compute_clusters)

            ll_new_cluster_without_i = self.score_cache[self.clusters[new_k].hash] if new_k < len(self.clusters) else 0
            ll_old_cluster_without_i = self.score_cache[hash(old_cluster)] if old_cluster is not None else 0

            a = (
                self.score_cache[hash(new_cluster)]
                - ll_new_cluster_without_i
                - sur_LL[new_k]
                - self.score_cache[self.clusters[old_k].hash]
                + ll_old_cluster_without_i
                + sur_LL[old_k]
            )
            rng, mh_rng = jax.random.split(rng)
            if a > np.log(jax.random.uniform(mh_rng)):
                # Upate state
                if new_k < len(self.clusters):
                    self.clusters[new_k] = self.clusters[new_k].merge_point(i, self.data[i])
                else:
                    self.clusters.append(self.ClusterClass({i}, data=self.data[i]))
                self.logpost = sum(
                    [self.prior(np.array([cl.size])) + self.score_cache[cl.hash] for cl in self.clusters]
                )

                if self.clusters[old_k].size > 1:
                    new_ids = self.clusters[old_k].data - {i}
                    self.clusters[old_k] = self.ClusterClass(
                        new_ids, data=self.data[np.fromiter(new_ids, dtype=np.int64)]
                    )
                else:
                    del self.clusters[old_k]

        return model_evals, surrogate_evals

    def cluster(self, rng, sweeps=100, callback=None):
        n_evals = []
        if self.logpost is None:
            rng, compute_rng = jax.random.split(rng)
            evals = self.compute_scores(compute_rng, [cl.data for cl in self.clusters])
            n_evals.append([evals, 0])
            self.logpost = sum([self.prior(np.array([cl.size])) + self.score_cache[cl.hash] for cl in self.clusters])
            self.best_clustering = self.clusters.copy()
            self.best_logpost = self.logpost.copy()

        if self.surrogate is not None:
            update_step = self.update_mh
        else:
            update_step = self.update_exact

        for iteration in (pbar := tqdm(range(sweeps))):
            rng, shuffle_rng = jax.random.split(rng)
            ids = jax.random.permutation(shuffle_rng, self.data.shape[0])
            for i, idx in enumerate(ids):
                rng, update_rng = jax.random.split(rng)
                model_evals, surrogate_evals = update_step(update_rng, idx.item())
                n_evals.append([model_evals, surrogate_evals])

                if self.logpost >= self.best_logpost:
                    self.best_logpost = self.logpost.copy()
                    self.best_clustering = self.clusters.copy()

                pbar.set_postfix(
                    {
                        "Sweep progress": f"{i + 1}/{len(ids)}",
                        "Best": f"{self.best_logpost:.4g}",
                        "Current": f"{self.logpost:.4g}",
                    }
                )

            if callback is not None:
                callback(self.clusters, iteration)

        print()
        return n_evals

    def summary(self, print_cluster_data=False):
        # Print out summary of clustering
        print(f"Current (weight {self.logpost:.4g}):")
        clusters = sorted(self.clusters, key=lambda c: c.size, reverse=True)
        print(f"{len(clusters)} clusters, {self.data.shape[0]} points, {[c.size for c in clusters]}")
        if print_cluster_data:
            for c in clusters:
                print(
                    f"LL:{self.score_cache[c.hash]:.2g}, ",
                    [str(obs) for obs in self.data[c.ids]] if print_cluster_data else "",
                )
            print()

        print(f"Estimated MAP (weight {self.best_logpost:.4g}):")
        clusters = sorted(self.best_clustering, key=lambda c: c.size, reverse=True)
        print(f"{len(clusters)} clusters, {self.data.shape[0]} points, {[c.size for c in clusters]}")
        if print_cluster_data:
            for c in clusters:
                print(
                    f"LL:{self.score_cache[c.hash]:.2g}, ",
                    [str(obs) for obs in self.data[c.ids]] if print_cluster_data else "",
                )
            print()

    def list_cluster_labels(self, best=True):
        """
        Return a list of the cluster IDs for each observation

        """
        cluster_lookup = {}
        for cl in self.best_clustering if best else self.clusters:
            for i in cl.ids:
                cluster_lookup[i.item()] = str(cl.hash)
        return [cluster_lookup[i] for i in sorted(cluster_lookup.keys())]
