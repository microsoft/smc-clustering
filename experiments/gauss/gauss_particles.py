# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Sweep particle counts for Gaussian SMC experiments.

This script benchmarks SMC variants across particle counts on the Gaussian synthetic dataset.
"""

import argparse
import collections
import pickle
from pathlib import Path
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

from gauss_setup import alpha, batched_score_eval, generate_gauss_dataset, prior, surrogate


data, labels = generate_gauss_dataset()
n = data.shape[0]

rng = jax.random.PRNGKey(seed)
shuffle_rng, rng = jax.random.split(rng)
shuffled_idx = jax.random.permutation(shuffle_rng, n)

labels = np.array(labels[shuffled_idx])
data = np.array(data[shuffled_idx])

ground_truth = [str(i) for i in labels]

#####################


ps = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 3000]

results = collections.defaultdict(lambda: collections.defaultdict(list))

configs = ["split", "smc"]

resample_fn = resample_greedy
t_max = 60 * 60

for conf in configs:
    t = 0
    for p in ps:
        if t < t_max:
            max_particles = p
            max_evals = 0
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
            metrics["evals"] = np.mean(evals)
            metrics["subprobs"] = np.array(subprobs)
            metrics["t"] = t
            print(f"{method}\n {t:.4g} {metrics['LP'], metrics['f1']}")

            for metric in metrics.keys():
                results[method][metric].append(metrics[metric])

            with Path(f"data/gauss_particles_s{seed}.pickle").open("wb") as handle:
                cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
