# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run online SMC experiments on the Gaussian dataset.

The script reveals Gaussian observations incrementally and records clustering metrics after each update block.
"""

import argparse
import collections
import pickle
from pathlib import Path

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

p = 100

results = collections.defaultdict(lambda: collections.defaultdict(list))

configs = ["split", "smc", "greedy"]

resample_fn = resample_greedy
increment = 10
for conf in configs:
    max_particles = 1 if conf == "greedy" else p
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

    for i in range(increment, data.shape[0] + increment, increment):
        ds_size = min(data.shape[0], i + 1)
        if i + 1 > data.shape[0]:
            steps = data.shape[0] - 1 - (i - increment)
        else:
            steps = increment

        rng, cl_rng = jax.random.split(rng)
        n_evals, _ = clusterer.cluster(cl_rng, steps=steps, callback_interval=0)

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
        print(f"{method}\n {i} {metrics['LP'], metrics['f1']}")

        for metric in metrics.keys():
            results[method][metric].append(metrics[metric])

        with Path(f"data/gauss_smc_online_s{seed}.pickle").open("wb") as handle:
            cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
