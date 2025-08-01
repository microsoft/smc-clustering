# Licensed under the MIT license.
import collections
import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm


logger = logging.getLogger(__name__)
# ====================== Priors ======================


class Uniform:
    """
    Uniform prior on cluster sizes
    """

    def __call__(self, n_obs, cluster_size, n_clusters):
        return jnp.zeros((cluster_size.shape[0]))


class DirichletProcess:
    """
    Dirichlet process prior on cluster sizes
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, n_obs, cluster_size, n_clusters):
        # Prior probability of a clustering, based on cluster size and hyperparameter alpha
        return jnp.where(
            cluster_size > 0, jnp.log(cluster_size / (n_obs + self.alpha)), jnp.log(self.alpha / (n_obs + self.alpha))
        )


class PitmanYorProcess:
    """
    Pitman-Yor process prior on cluster sizes
    """

    def __init__(self, alpha, d):
        self.alpha = alpha
        self.d = d

    def __call__(self, n_obs, cluster_size, n_clusters):
        # Prior probability of a clustering, based on cluster size and hyperparameters
        return jnp.where(
            cluster_size > 0,
            jnp.log((cluster_size - self.d) / (n_obs + self.alpha)),
            jnp.log((self.alpha + self.d * n_clusters) / (n_obs + self.alpha)),
        )


# ====================== Clusterer ======================


class SMCClustererState:
    """
    Represents particles as sets of cluster hashes, with a shared lookup table
    for the ids of the datapoints in each cluster.
    """

    def __init__(self, data, ClusterClass, score_cache=None):
        self.data = data
        self.ClusterClass = ClusterClass
        empty_cluster = ClusterClass([])
        initial_cluster = ClusterClass([0], data=self.data[0])
        self.cluster_partition = [{initial_cluster.hash}]
        self.clusters = {empty_cluster.hash: empty_cluster, initial_cluster.hash: initial_cluster}
        self.particles = [[{initial_cluster.hash}]]
        self.weights = [jnp.zeros((1,))]
        if score_cache is not None:
            self.score_cache = score_cache
            self.score_cache[empty_cluster.hash] = 0
        else:
            self.score_cache = {empty_cluster.hash: 0}
        self.n_obs = 1

    def get_descendants(self, particle_id):
        """
        Compute possible descendants of a particle, retrieve summary statistics and weights
        """
        old_cluster_ids = list(self.particles[particle_id[0]][particle_id[1]]) + [self.ClusterClass([]).hash]
        old_clusters = [
            self.clusters[cluster_hash] for cluster_hash in self.particles[particle_id[0]][particle_id[1]]
        ] + [self.ClusterClass([])]

        n = [cluster.size for cluster in old_clusters]
        summary = [cluster.summary for cluster in old_clusters]

        return (
            [[particle_id[1]] * len(old_clusters), old_cluster_ids],
            n,
            summary,
            [self.weights[particle_id[0]][particle_id[1]]] * len(old_clusters),
        )

    def merge_point(self, particle_id, datapoint_id, cluster_hash):
        """
        Add a datapoint to its assigned cluster on a given particle
        """
        new_hash = hash(self.clusters[cluster_hash].add(datapoint_id))  # this could be computed directly
        self.particles[particle_id[0]][particle_id[1]].add(new_hash)

        if new_hash not in self.clusters:
            # create new cluster
            self.clusters[new_hash] = self.clusters[cluster_hash].merge_point(datapoint_id, self.data[datapoint_id])

        # delete old cluster from particle
        if cluster_hash in self.particles[particle_id[0]][particle_id[1]]:
            self.particles[particle_id[0]][particle_id[1]].remove(cluster_hash)

    def update_particle_set(self, subprob, particle_ids, weights, cluster_assignments, datapoint_id):
        """
        Update the particle set with assignments for the new datapoint
        """

        # resample the particles
        self.particles[subprob] = [self.particles[subprob][i].copy() for i in particle_ids]
        self.weights[subprob] = weights

        # assign the new datapoint to its clusters
        for i, cluster_hash in enumerate(cluster_assignments):
            self.merge_point((subprob, i), datapoint_id, cluster_hash)

        # delete the clusters that are no longer in use
        used_clusters = set.union(*self.particles[subprob])
        unused_clusters = self.cluster_partition[subprob] - used_clusters - set({self.ClusterClass([]).hash})
        for cl in unused_clusters:
            del self.clusters[cl]
        self.cluster_partition[subprob] = used_clusters

    def update_and_merge_particle_set(self, subprobs, particle_ids, weights, cluster_assignments, datapoint_id):
        """
        Update the particle set with assignments for the new datapoint, merging subproblems together
        """
        # create particles
        new_particles = [
            set.union(*[self.particles[subprobs[i]][p[i]] for i in range(len(subprobs))]) for p in particle_ids
        ]

        subprobs = sorted(subprobs, reverse=True)
        for p in subprobs:
            del self.particles[p]
            del self.weights[p]

        self.particles.append(new_particles)
        self.weights.append(weights)
        subprob = len(self.particles) - 1

        # assign the new datapoint to its clusters
        for i, cluster_hash in enumerate(cluster_assignments):
            self.merge_point((subprob, i), datapoint_id, cluster_hash)

        # delete the clusters that are no longer in use
        used_clusters = set.union(*self.particles[subprob])
        unused_clusters = (
            set.union(*[self.cluster_partition[p] for p in subprobs])
            - used_clusters
            - set({self.ClusterClass([]).hash})
        )
        for p in subprobs:
            del self.cluster_partition[p]
        for cl in unused_clusters:
            del self.clusters[cl]
        self.cluster_partition.append(used_clusters)

    def add_subproblem(self, data_id):
        """
        Add datapoint to new subproblem by itself
        """
        new_cluster = self.ClusterClass([data_id], data=self.data[data_id])
        self.cluster_partition.append({new_cluster.hash})
        self.clusters[new_cluster.hash] = new_cluster
        self.particles.append([{new_cluster.hash}])
        self.weights.append(jnp.zeros((1,)))

    def retrieve_cluster_data(self, cluster_hash):
        """
        Return the datapoints in a given cluster
        """
        return self.data[self.clusters[cluster_hash].ids]

    def split_problem(self, s):
        """
        Attempt to split given problem into subproblems
        """

        # list datapoints on partition
        data_idx = np.concatenate([self.clusters[cl].ids for cl in self.particles[s][0]])
        ids = {data_idx[i]: i for i in range(len(data_idx))}

        # construct adjacency matrix
        n = len(data_idx)
        E = np.zeros((n, n))
        for cl in set.union(*self.particles[s]):
            for n_i, i in enumerate(self.clusters[cl].ids):
                for j in self.clusters[cl].ids[n_i:]:
                    E[ids[i], ids[j]] += 1
        E = scipy.sparse.csr_matrix(E)

        # compute subgraphs
        n_c, c = scipy.sparse.csgraph.connected_components(E, directed=False)
        if n_c > 1:
            # list out new subproblems
            data_partition = [set(data_idx[np.where(c == i)[0]]) for i in range(n_c)]
            cluster_partition = [
                set([cl for cl in self.cluster_partition[s] if self.clusters[cl].data.issubset(data_partition[i])])
                for i in range(n_c)
            ]

            # compute new particles and weights
            new_weights = []
            new_particles = []
            for subprob in cluster_partition:
                new_particles.append([])
                new_weights.append([])
                for p in self.particles[s]:
                    new_particle = p.intersection(subprob)
                    if new_particle not in new_particles[-1]:
                        new_particles[-1].append(new_particle)
                        new_weights[-1].append(
                            jax.scipy.special.logsumexp(
                                jnp.array(
                                    [
                                        self.weights[s][i]
                                        for i in range(len(self.particles[s]))
                                        if new_particle.issubset(self.particles[s][i])
                                    ]
                                )
                            )
                        )

            # update state
            del self.cluster_partition[s]
            del self.particles[s]
            del self.weights[s]
            self.cluster_partition += cluster_partition
            self.particles += new_particles
            self.weights += [jnp.array(w) for w in new_weights]

        return n_c

    def list_cluster_labels(self, metric=None):
        """
        Return a list of the cluster IDs for each observation,
        obtained from the top-weighted particle or best particle according to some metric

        """
        cluster_lookup = {}
        for s in range(len(self.weights)):
            if metric is None:
                top_particle = self.particles[s][np.argmax(self.weights[s])]

            else:
                expected_metric = []
                cluster_assignments = []
                for p in self.particles[s]:
                    cluster_lookup_p = {}
                    for cl in p:
                        for i in self.clusters[cl].ids:
                            cluster_lookup_p[i.item()] = self.clusters[cl].data
                    cluster_assignments.append(cluster_lookup_p)

                for p in range(len(self.particles[s])):
                    expected_metric.append(
                        sum(
                            [
                                jnp.exp(self.weights[s][i]) * metric(cluster_assignments[p], cluster_assignments[i])
                                for i in range(len(self.particles[s]))
                            ]
                        )
                    )

                top_particle = self.particles[s][np.argmax(expected_metric)]

            for cl in top_particle:
                for i in self.clusters[cl].ids:
                    cluster_lookup[i.item()] = str(cl)
        return [cluster_lookup[i] for i in sorted(cluster_lookup.keys())]


class SMCClusterer:
    def __init__(
        self,
        data,
        score_fn,
        max_evals,
        max_particles,
        prior,
        surrogate,
        ClusterClass,
        resample_fn,
        split_interval=None,
        surrogate_threshold=None,
        model_threshold=None,
        callback=None,
        score_cache=None,
        print_cluster_data=False,
    ):
        self.score_fn = score_fn
        self.prior = prior
        self.surrogate = surrogate
        self.max_evals = max_evals
        self.max_particles = max_particles
        self.resample = resample_fn
        self.split_interval = split_interval
        self.surrogate_threshold = surrogate_threshold
        self.model_threshold = model_threshold
        self.callback = callback
        self.print_cluster_data = print_cluster_data

        self.state = SMCClustererState(data, ClusterClass, score_cache=score_cache)

    def compute_scores(self, rng, clusters, force_recompute=False):
        """
        For a list of clusters, compute the score for each cluster
        We use a cache to avoid recomputing scores for clusters that have already been computed.

        """
        # remove the score from the cache if we're forcing a recompute
        if force_recompute:
            [self.state.score_cache.pop(hash(cluster), None) for cluster in clusters]

        compute_clusters = []
        hashes = []
        for c in clusters:
            if hash(c) not in self.state.score_cache and hash(c) not in hashes:
                compute_clusters.append(c)
                hashes.append(hash(c))

        if len(hashes) == 0:
            return

        scores = self.score_fn(
            rng, [self.state.data[np.fromiter(cluster, dtype=np.int64)] for cluster in compute_clusters]
        )
        for score, hash_ in zip(scores, hashes):
            self.state.score_cache[hash_] = score

    def update_step(self, rng, new_obs, verbose):
        """
        Update particle set given new observation
        """
        rng, update_rng, move_rng = jax.random.split(rng, 3)
        n_probs = len(self.state.weights)
        empty_hash = self.state.ClusterClass([]).hash
        evals = 0

        # Construct putative particle set
        putative_particles = []
        cluster_sizes = []
        n_clusters = []
        summary_stats = []
        old_weights = []
        for i in range(n_probs):
            putative_particles.append([[], []])
            cluster_sizes.append([])
            n_clusters.append([])
            summary_stats.append([])
            old_weights.append([])
            for j in range(len(self.state.weights[i])):
                p, n, s, w = self.state.get_descendants((i, j))
                putative_particles[i][0] += p[0]
                putative_particles[i][1] += p[1]
                cluster_sizes[i] += n
                n_clusters[i] += [len(self.state.particles[i][j])] * len(n)
                summary_stats[i] += s
                old_weights[i] += w

            putative_particles[i] = np.array(putative_particles[i], dtype=np.int64)
            cluster_sizes[i] = np.array(cluster_sizes[i])
            n_clusters[i] = np.array(n_clusters[i])
            old_weights[i] = np.array(old_weights[i])

        # Normalise old subproblem weights, compute prior
        putative_weights = [
            old_weights[i]
            + self.prior(self.state.n_obs, cluster_sizes[i], n_clusters[i])
            - jax.scipy.special.logsumexp(old_weights[i], axis=-1)
            for i in range(n_probs)
        ]
        # Evaluate surrogate model
        sur_LL = [self.surrogate.post_predictive(new_obs, cluster_sizes[i], summary_stats[i]) for i in range(n_probs)]
        single_LL = sur_LL[-1][-1]

        if n_probs == 1:
            p = jnp.zeros((), dtype=jnp.int32)
            putative_particles = putative_particles[0]
            putative_weights = putative_weights[0]
            sur_LL = sur_LL[0]
        else:
            test_weights = [putative_weights[i] + sur_LL[i] for i in range(n_probs)]
            test_weights = jnp.array(
                [jax.scipy.special.logsumexp(test_weights[i][cluster_sizes[i] > 0]) for i in range(len(test_weights))]
            )
            p_max = jnp.argmax(test_weights)

            # Remove duplicate singleton clusters, leaving one associated with the highest probability subproblem
            for p_i in range(n_probs):
                if p_i != p_max:
                    non_empty = cluster_sizes[p_i] > 0
                    putative_particles[p_i] = putative_particles[p_i][:, non_empty]
                    putative_weights[p_i] = putative_weights[p_i][non_empty]
                    sur_LL[p_i] = sur_LL[p_i][non_empty]

            p = jnp.concatenate([jnp.full(len(putative_weights[p_i]), p_i) for p_i in range(n_probs)])
            putative_particles = np.concatenate(putative_particles, axis=-1)
            putative_weights = jnp.concatenate(putative_weights)
            sur_LL = jnp.concatenate(sur_LL)

        putative_weights += sur_LL
        if self.surrogate_threshold is not None:
            # Discard assignments below threshold
            keep_ids = sur_LL - single_LL > self.surrogate_threshold
            putative_weights = putative_weights[keep_ids]
            putative_particles = putative_particles[:, keep_ids]
            sur_LL = sur_LL[keep_ids]
            if p.size > 1:
                p = p[keep_ids]

        if self.max_evals > 0:
            if len(putative_particles[1]) > self.max_evals:
                # Resample max_evals particles if number of new clusters exceeds max_evals
                rng, resample_rng = jax.random.split(rng)
                new_particle_ids, putative_weights = self.resample(resample_rng, putative_weights, self.max_evals)
                putative_particles = putative_particles[:, new_particle_ids]
                sur_LL = sur_LL[new_particle_ids]

                if p.size > 1:
                    p = p[new_particle_ids]

            # Reweight according to model
            putative_weights -= sur_LL
            new_clusters = [self.state.clusters[cluster].add(self.state.n_obs) for cluster in putative_particles[1]]
            self.compute_scores(update_rng, new_clusters)
            evals = len(jnp.unique(putative_particles[1]))
            update = jnp.array(
                [
                    self.state.score_cache[hash(new_cluster)] - self.state.score_cache[old_cluster_id]
                    for new_cluster, old_cluster_id in zip(new_clusters, putative_particles[1])
                ]
            )
            putative_weights += update

            if empty_hash in putative_particles[1]:
                single_LL = update[putative_particles[1] == empty_hash][0]
                if self.model_threshold is not None:
                    # Discard assignments below threshold
                    keep_ids = update - single_LL > self.model_threshold
                    putative_weights = putative_weights[keep_ids]
                    putative_particles = putative_particles[:, keep_ids]
                    if p.size > 1:
                        p = p[keep_ids]

        new_particle_ids = None
        if putative_weights.shape[0] > self.max_particles:
            # Resample
            rng, resample_rng = jax.random.split(rng)
            new_particle_ids, putative_weights = self.resample(resample_rng, putative_weights, self.max_particles)
            putative_particles = putative_particles[:, new_particle_ids]
            if p.size > 1:
                p = p[new_particle_ids]

        subprobs = jnp.unique(p)
        expected_resample = None
        if len(subprobs) > 1:
            # Remove subproblems that have negligible weight in particle set
            w = putative_weights - jax.scipy.special.logsumexp(putative_weights)
            subprob_weights = jnp.array([jax.scipy.special.logsumexp(w[p == s]) for s in p])
            expected_resample = subprob_weights > -jnp.log(self.max_particles)
            if not expected_resample.all():
                p = p[expected_resample]
                putative_particles = putative_particles[:, expected_resample]
                putative_weights = putative_weights[expected_resample]
                subprobs = jnp.unique(p)

        assignments = putative_particles[1]
        weights = putative_weights - jnp.max(putative_weights)

        if logger.level <= 20:
            logger.info(f"Obervation: {str(self.state.data[self.state.n_obs][0])}")
            unique_idx = jnp.unique(assignments, return_index=True)[1]
            unique_assignments = assignments[unique_idx]
            pooled_weights = jnp.array(
                [jax.scipy.special.logsumexp(weights[assignments == a]) for a in unique_assignments]
            )

            if self.max_evals > 0:
                update = update if new_particle_ids is None else update[new_particle_ids]
                update = update if expected_resample is None else update[expected_resample]
            else:
                sur_LL = sur_LL if new_particle_ids is None else sur_LL[new_particle_ids]
                sur_LL = sur_LL if expected_resample is None else sur_LL[expected_resample]

            for w, cl in sorted(
                zip(pooled_weights - jax.scipy.special.logsumexp(weights), unique_assignments),
                key=lambda c: c[0],
                reverse=True,
            ):
                if self.max_evals > 0:
                    logger.info(
                        f"-> Weight {jnp.exp(w):.2g}, score {(update[putative_particles[1] == cl][0] - single_LL):.2g}: {[str(i) for i in self.state.retrieve_cluster_data(np.int64(cl.item()))]}"
                    )
                else:
                    logger.info(
                        f"-> Weight {jnp.exp(w):.2g}, score {(sur_LL[putative_particles[1] == cl][0] - single_LL):.2g}: {[str(i) for i in self.state.retrieve_cluster_data(np.int64(cl.item()))]}"
                    )

        # Update particle set, merging subproblems if necessary
        if (putative_particles[1] == empty_hash).all() and self.split_interval is not None:
            # If new observation is in a cluster on its own on all particles, add it to a new subproblem and do not update other subproblems
            self.state.add_subproblem(self.state.n_obs)
            p = len(self.state.particles) - 1

        elif len(subprobs) == 1:
            # A single subproblem has been chosen - update particle set
            p = subprobs.reshape(())
            self.state.update_particle_set(p, putative_particles[0], weights, putative_particles[1], self.state.n_obs)

        elif len(subprobs) == 2 or len([p for p in subprobs if len(self.state.weights[p]) > 1]) <= 2:
            # More than one subproblem has been chosen
            # Compute full joint distribution for merged problem as all combinations of chosen subproblems
            # (This is not attempted if more than two subproblems have more than 1 particle)
            subprobs = sorted(subprobs, key=lambda p: len(self.state.weights[p]), reverse=True)

            n_pairs = sum(p == subprobs[0]) * len(self.state.particles[subprobs[1]]) + sum(p == subprobs[1]) * len(
                self.state.particles[subprobs[0]]
            )
            pairings = np.zeros((n_pairs, len(subprobs)), dtype=np.int64)
            others = [0] * (max(0, len(subprobs) - 2))
            assignments = np.zeros((n_pairs,), dtype=np.int64)
            pair_weights = np.zeros((n_pairs,))

            i = 0
            for w_i, (p_i, c_i) in zip(
                putative_weights[p == subprobs[0]], putative_particles[:, p == subprobs[0]].transpose()
            ):
                for p_j in range(len(self.state.particles[subprobs[1]])):
                    pairings[i] = np.array([[p_i, p_j, *others]])
                    assignments[i] = c_i
                    pair_weights[i] = w_i + self.state.weights[subprobs[1]][p_j]
                    i += 1

            for w_j, (p_j, c_j) in zip(
                putative_weights[p == subprobs[1]], putative_particles[:, p == subprobs[1]].transpose()
            ):
                for p_i in range(len(self.state.particles[subprobs[0]])):
                    pairings[i] = np.array([[p_i, p_j, *others]])
                    assignments[i] = c_j
                    pair_weights[i] = w_j + self.state.weights[subprobs[0]][p_i]
                    i += 1

            if n_pairs > self.max_particles:
                # Resample again to bring new subproblem to correct size
                rng, resample_rng = jax.random.split(rng)
                new_particle_ids, weights = self.resample(resample_rng, pair_weights, self.max_particles)
                assignments = assignments[new_particle_ids]
                self.state.update_and_merge_particle_set(
                    subprobs, pairings[new_particle_ids], weights, assignments, self.state.n_obs
                )
            else:
                weights = pair_weights
                self.state.update_and_merge_particle_set(subprobs, pairings, weights, assignments, self.state.n_obs)

            p = len(self.state.particles) - 1

            if verbose:
                if self.callback is not None:
                    self.callback(self.state, highlight=self.state.n_obs)
                else:
                    print("Merged problem:")
                    self.summary(problems=[p])

        else:
            # More than one subproblem chosen and more than two subproblems have >1 particles
            # Avoid calculating full joint distribution by independantly sampling particles from each subproblem

            # Resample assignments with replacement, allowing duplicates
            rng, merge_rng = jax.random.split(rng)
            w = jnp.exp(putative_weights - jax.scipy.special.logsumexp(putative_weights))
            resample_idx = jax.random.choice(merge_rng, jnp.arange(len(w)), (self.max_particles,), replace=True, p=w)
            p = p[resample_idx]
            putative_particles = putative_particles[:, resample_idx]

            # Sample other subproblem solutions
            particle_ids = []
            assignments = []
            counts = []
            for i in range(self.max_particles):
                particle_ids_i = []
                for j in subprobs:
                    if j == p[i]:
                        particle_ids_i.append(putative_particles[0][i].item())
                        assignment = putative_particles[1][i]
                    else:
                        rng, sample_rng = jax.random.split(rng)
                        particle_ids_i.append(
                            jax.random.choice(
                                sample_rng, len(self.state.particles[j]), p=jnp.exp(jnp.array(self.state.weights[j]))
                            ).item()
                        )

                # Merge duplicates
                if particle_ids_i in particle_ids:
                    idx = particle_ids.index(particle_ids_i)
                    counts[idx] += 1
                else:
                    particle_ids.append(particle_ids_i)
                    assignments.append(assignment)
                    counts.append(1)

            weights = jnp.log(jnp.array(counts)) - jnp.log(self.max_particles)
            self.state.update_and_merge_particle_set(subprobs, particle_ids, weights, assignments, self.state.n_obs)

            p = len(self.state.particles) - 1
            if verbose:
                if self.callback is not None:
                    self.callback(self.state, highlight=self.state.n_obs)
                else:
                    print("Merged problem:")
                    self.summary(problems=[p])

        # Split the subproblem that was just updated
        if (
            self.split_interval is not None
            and len(jnp.concatenate([self.state.clusters[cl].ids for cl in self.state.particles[p][0]]))
            > self.split_interval
        ):
            if verbose:
                old_summary = self.summary(problems=[p], print_summary=False)

            n_split = self.state.split_problem(p)

            if verbose:
                problems = [len(self.state.particles) - i - 1 for i in range(n_split)]
                if n_split > 1:
                    if self.callback is not None:
                        self.callback(self.state)
                    else:
                        print("\nProblem to split:")
                        print(old_summary)
                        print("\nSplit problem:")
                        self.summary(problems=problems)

        # Return number of model evaluations made
        return evals

    def cluster(self, rng, steps=None, callback_interval=5, verbose=False):
        """
        Cluster the dataset
        """

        if steps is None:
            steps = self.state.data.shape[0] - 1

        if self.state.n_obs == 1:
            rng, score_rng = jax.random.split(rng)
            if self.score_fn is not None and self.max_evals > 0:
                self.compute_scores(score_rng, [frozenset({0})])

        n_evals = []
        n_subprobs = []
        for _ in (pbar := tqdm(range(min(steps, self.state.data.shape[0] - self.state.n_obs)))):
            rng, update_rng = jax.random.split(rng)
            new_obs = self.state.data[self.state.n_obs]

            evals = self.update_step(update_rng, new_obs, verbose)
            n_evals.append(evals)
            n_subprobs.append(len(self.state.particles))
            self.state.n_obs += 1

            pbar.set_postfix({"Subproblems": f"{len(self.state.particles)}", "Evals": f"{evals}"})

            if self.callback and callback_interval > 0 and self.state.n_obs % callback_interval == 0:
                self.callback(self.state)
                self.summary()

        print()
        return n_evals, n_subprobs

    def summary(self, problems=None, max_print=3, min_problem_size=1, print_summary=True):
        """
        Summarise particle set
        """
        summary_text = ""
        if problems is None:
            problems = range(len(self.state.particles))
            n_points = [sum([self.state.clusters[c].size for c in self.state.particles[p][0]]) for p in problems]
            summary_text += f"{len(problems)} subproblems of sizes {sorted(n_points, reverse=True)}"

        for p in problems:
            n_points = sum([self.state.clusters[c].size for c in self.state.particles[p][0]])
            if n_points >= min_problem_size:
                summary_text += f"\nSubproblem {p}: {len(self.state.particles[p])} particles, {n_points} points"
                for i, (particle, weight) in enumerate(
                    sorted(zip(self.state.particles[p], self.state.weights[p]), key=lambda c: c[1], reverse=True)
                ):
                    if i >= max_print:
                        summary_text += f"\n\t... {len(self.state.particles[p]) - max_print} particles omitted"
                        break
                    exp_weight = jnp.exp(weight - jax.scipy.special.logsumexp(jnp.array(self.state.weights[p])))
                    clusters = sorted(particle, key=lambda c: self.state.clusters[c].size, reverse=True)
                    summary_text += f"\n\tParticle {i}, weight {exp_weight:.2g}({weight:.3g}), {len(clusters)} clusters, {[self.state.clusters[c].size for c in clusters]}"
                    if self.print_cluster_data:
                        for c in clusters:
                            summary_text += f"\n\t\tLL:{self.state.score_cache[c]:.2g}, " if self.max_evals > 0 else ""
                            summary_text += str(self.state.retrieve_cluster_data(c))
                        summary_text += "\n"
                summary_text += "\n"

        if print_summary:
            print(summary_text)

        return summary_text


def plot_particles_2D(state, subprob=None, n_plots=5, fig_scale=3, highlight=None):
    """
    Plot particles with highest weights
    """
    subprob = 0 if len(state.particles) == 1 else subprob
    if subprob is not None:
        n_plots = min(n_plots, len(state.particles[subprob]))
        ncols = min(5, n_plots)
        nrows = 1 + (n_plots - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_scale * ncols, fig_scale * nrows))
        for i, (particle, weight) in enumerate(
            sorted(zip(state.particles[subprob], state.weights[subprob]), key=lambda c: c[1], reverse=True)[:n_plots]
        ):
            subfig = axes[int(i // ncols), i % ncols] if nrows > 1 else axes[i] if n_plots > 1 else axes
            weight = jnp.exp(weight - jax.scipy.special.logsumexp(jnp.array(state.weights[subprob])))
            clusters = sorted(
                [state.retrieve_cluster_data(cluster_hash) for cluster_hash in particle],
                key=lambda c: c.shape[0],
                reverse=True,
            )

            if state.n_obs < state.data.shape[0]:
                unobserved_pts = state.data[(state.n_obs + 1) :]
                subfig.plot(unobserved_pts[:, 0], unobserved_pts[:, 1], "x", color="black", alpha=0.25, markersize=2)

            for c in clusters:
                if c.shape[0] > 1:
                    subfig.plot(c[:, 0], c[:, 1], "o", markersize=3)
                else:
                    subfig.plot(c[:, 0], c[:, 1], "ks", markersize=3, alpha=0.75)

            if highlight is not None:
                subfig.plot(
                    state.data[highlight, 0],
                    state.data[highlight, 1],
                    "o",
                    markersize=6,
                    color="black",
                    markerfacecolor="none",
                )

            subfig.set_title(f"Particle {i + 1}/{len(state.particles[subprob])}, weight {weight:.2g}")
            subfig.set_aspect("equal")
            subfig.set_xticks([])
            subfig.set_yticks([])

        for i in range(n_plots, nrows * ncols):
            subfig = axes[int(i // ncols), i % ncols] if nrows > 1 else axes[i] if n_plots > 1 else axes
            subfig.set_xticks([])
            subfig.set_yticks([])

        fig.tight_layout()
        plt.show()

        return fig

    else:
        ncols = min(5, n_plots)
        nrows = 1 + (n_plots - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_scale * ncols, fig_scale * nrows))
        markers = [
            "o",
            "v",
            "^",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
            "8",
            "s",
            "p",
            "*",
            "h",
            "H",
            "+",
            "x",
            "D",
            "d",
            "|",
            "_",
            "P",
            "X",
            0,
            2,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]
        for subprob in range(len(state.particles)):
            for i, (particle, weight) in enumerate(
                sorted(zip(state.particles[subprob], state.weights[subprob]), key=lambda c: c[1], reverse=True)[
                    :n_plots
                ]
            ):
                subfig = axes[int(i // ncols), i % ncols] if nrows > 1 else axes[i] if n_plots > 1 else axes
                weight = jnp.exp(weight - jax.scipy.special.logsumexp(jnp.array(state.weights[subprob])))
                clusters = sorted(
                    [state.retrieve_cluster_data(cluster_hash) for cluster_hash in particle],
                    key=lambda c: c.shape[0],
                    reverse=True,
                )

                for c in clusters:
                    if c.shape[0] > 1:
                        subfig.plot(c[:, 0], c[:, 1], markers[subprob % len(markers)], markersize=3)
                    else:
                        subfig.plot(
                            c[:, 0], c[:, 1], markers[subprob % len(markers)], markersize=3, alpha=0.75, color="black"
                        )

                x_min = jnp.min(jnp.concatenate(clusters)[:, 0]) - 0.2
                x_max = jnp.max(jnp.concatenate(clusters)[:, 0]) + 0.2
                y_min = jnp.min(jnp.concatenate(clusters)[:, 1]) - 0.2
                y_max = jnp.max(jnp.concatenate(clusters)[:, 1]) + 0.2
                subfig.plot(
                    [x_min, x_min, x_max, x_max, x_min],
                    [y_min, y_max, y_max, y_min, y_min],
                    color="black",
                    linestyle="dotted",
                )
                subfig.annotate(f"{weight:.2g}", (x_max, jnp.mean(jnp.concatenate(clusters)[:, 1])))

        for i in range(n_plots):
            subfig = axes[int(i // ncols), i % ncols] if nrows > 1 else axes[i] if n_plots > 1 else axes
            if state.n_obs < state.data.shape[0]:
                unobserved_pts = state.data[(state.n_obs + 1) :]
                subfig.plot(unobserved_pts[:, 0], unobserved_pts[:, 1], "x", color="black", alpha=0.25, markersize=2)

            if highlight is not None:
                subfig.plot(
                    state.data[highlight, 0],
                    state.data[highlight, 1],
                    "o",
                    markersize=6,
                    color="black",
                    markerfacecolor="none",
                )

            subfig.set_aspect("equal")
            subfig.set_xticks([])
            subfig.set_yticks([])

        for i in range(n_plots, nrows * ncols):
            subfig = axes[int(i // ncols), i % ncols] if nrows > 1 else axes[i] if n_plots > 1 else axes
            subfig.set_xticks([])
            subfig.set_yticks([])

        fig.tight_layout()
        plt.show()

        return fig


# ====================== Resamplers ======================


def resample_multinomial(rng, weights, max_particles, **kwargs):
    """
    Simple multinomial resampling scheme.
    """
    w = jnp.exp(weights - jax.scipy.special.logsumexp(weights))
    resample_idx = jax.random.choice(rng, jnp.arange(len(weights)), (max_particles,), replace=True, p=w)
    unique_idx, counts = jnp.unique(resample_idx, return_counts=True)
    return unique_idx, jnp.log(counts) - jnp.log(max_particles)


def resample_stratified(rng, weights, max_particles, **kwargs):
    """
    Stratified resampling scheme of Carpenter et al. (1999).
    """
    w = jnp.exp(weights - jax.scipy.special.logsumexp(weights))  # can the rest be done in log space?
    k = jnp.sum(w) / max_particles
    w = jnp.concatenate([w, jnp.zeros((1,))])
    u = jax.random.uniform(rng, minval=0, maxval=k)
    resample_idx = []
    i = 0
    u -= w[i]
    while i < len(weights):
        if u < 0:
            resample_idx.append(i)
            u = u + k
        else:
            i += 1
            u = u - w[i]

    unique_idx, counts = jnp.unique(jnp.array(resample_idx, dtype=jnp.int32), return_counts=True)
    return unique_idx, jnp.log(counts) - jnp.log(max_particles)


def resample_optimal(rng, weights, max_particles, **kwargs):
    """
    Optimal resampling scheme of Fearnhead and Clifford (2003) -
    automatically keeps the highest-weighted particles and uses stratified
    resampling to choose the rest.
    """
    weights -= jax.scipy.special.logsumexp(weights)
    w = jnp.exp(weights)
    bounds = [jnp.min(w), jnp.max(w)]
    while bounds[1] != bounds[0]:
        rng, k_rng = jax.random.split(rng)
        k = jax.random.uniform(k_rng, minval=bounds[0], maxval=bounds[1])
        p1 = w[w <= k]
        p2 = w[w > k]
        a = len(p2)
        b = jnp.sum(p1)
        if b / k + a <= max_particles:
            bounds[1] = jnp.max(p1)
        else:
            bounds[0] = jnp.min(p2)

    k = bounds[0]
    p1 = w[w <= k]
    p2 = w[w > k]
    a = len(p2)
    b = jnp.sum(p1)
    if b / k + a <= max_particles:
        c = (max_particles - a) / b

    elif (w > k).any():
        k = jnp.min(w[w > k])
        p1 = w[w <= k]
        p2 = w[w > k]
        a = len(p2)
        b = jnp.sum(p1)
        c = (max_particles - a) / b

    else:
        c = max_particles

    resample_idx = jnp.where(w < 1 / c)[0]
    keep_idx = jnp.where(w >= 1 / c)[0]
    l = len(weights) - len(resample_idx)
    rng, resample_rng = jax.random.split(rng)
    p, w = resample_stratified(resample_rng, weights[resample_idx], max_particles - l)

    new_particles = jnp.concatenate([resample_idx[p], keep_idx])
    new_weights = jnp.concatenate([jnp.full_like(w, jnp.log(1 / c)), weights[keep_idx]])

    return new_particles, new_weights


def resample_greedy(rng, weights, max_particles, **kwargs):
    """
    Deterministically chooses the top weighted particles.
    """
    idx = jnp.argsort(weights)
    return idx[-max_particles:], weights[idx[-max_particles:]]
