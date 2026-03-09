# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import collections
import pickle
import time

import cloudpickle
import jax
import numpy as np
import scipy.special

from smc_clustering.clustering.metrics import cluster_metrics
from smc_clustering.clustering.smc import SMCClusterer, resample_greedy
from smc_clustering.clustering.surrogate_models import GaussianCluster


parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=0)
args = parser.parse_args()
seed = args.seed

from circles_setup import alpha, generate_circles_dataset, load_model, prior, surrogate


batched_score_eval = load_model()
data, labels = generate_circles_dataset()

n = data.shape[0]
rng = jax.random.PRNGKey(seed)
shuffle_rng, rng = jax.random.split(rng)
shuffled_idx = jax.random.permutation(shuffle_rng, n)
labels = labels[shuffled_idx]
data = data[shuffled_idx]

ground_truth = [str(i) for i in labels]

#####################

ps = [1, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000]

results = collections.defaultdict(lambda: collections.defaultdict(list))

configs = ["split", "smc", "greedy"]

resample_fn = resample_greedy
t_max = 10**3

for conf in configs:
    t = 0
    for p in ps:
        if t < t_max and not (p > 1 and conf == "greedy"):
            max_particles = p
            max_evals = np.inf if conf == "greedy" else p
            split = True if conf == "split" else False
            method = f"{p}, {conf}"

            clusterer = SMCClusterer(
                data=data,
                split=split,
                score_fn=batched_score_eval,
                max_particles=max_particles,
                max_evals=max_evals,
                prior=prior,
                surrogate=surrogate,
                resample_fn=resample_fn,
                ClusterClass=GaussianCluster,
            )
            rng = jax.random.PRNGKey(seed)

            t = time.time()
            evals, subprobs = clusterer.cluster(rng, callback_interval=0)
            t = time.time() - t

            ll = sum(
                [
                    clusterer.state.score_cache[h]
                    for s in range(len(clusterer.state.particles))
                    for h in clusterer.state.particles[s][np.argmax(clusterer.state.weights[s])]
                ]
            )
            lp = ll + sum(
                [
                    np.log(alpha) + scipy.special.gammaln(clusterer.state.clusters[h].size)
                    for s in range(len(clusterer.state.particles))
                    for h in clusterer.state.particles[s][np.argmax(clusterer.state.weights[s])]
                ]
            )

            metrics = cluster_metrics(
                clusterer.state.list_cluster_labels(), ground_truth[: clusterer.state.n_obs]
            )
            metrics["LL"] = ll
            metrics["LP"] = lp
            metrics["total_evals"] = len(clusterer.state.score_cache)
            metrics["t"] = t
            print(f"{method}\n {t:.4g} {metrics['LP'], metrics['f1']}")

            for metric in metrics.keys():
                results[method][metric].append(metrics[metric])

            with open(f"data/circles_particles_s{seed}.pickle", "wb") as handle:
                cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
