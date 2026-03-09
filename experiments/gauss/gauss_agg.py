# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import collections
import pickle
import time

import cloudpickle
import jax
import numpy as np

from smc_clustering.clustering.agglomerative import Clusterer
from smc_clustering.clustering.metrics import cluster_metrics


parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=0)
args = parser.parse_args()
seed = args.seed

from gauss_setup import batched_score_eval, generate_gauss_dataset, prior


data, labels = generate_gauss_dataset()
n = data.shape[0]

rng = jax.random.PRNGKey(seed)
shuffle_rng, rng = jax.random.split(rng)
shuffled_idx = jax.random.permutation(shuffle_rng, n)

labels = np.array(labels[shuffled_idx])
data = np.array(data[shuffled_idx])

ground_truth = [str(i) for i in labels]

#####################

t_max = np.inf

results = collections.defaultdict(lambda: collections.defaultdict(list))

batch_sizes = [16, 32, 64, 128, data.shape[0]]

for batch_size in batch_sizes:
    score_cache = None
    rng = jax.random.PRNGKey(seed)
    clusterer = Clusterer(data, batched_score_eval, prior=prior, cluster_batch_size=batch_size)
    t = 0
    done = False
    iters = 10
    total_iters = 0
    best = -np.inf
    iters_since_change = 0
    while t < t_max and not done and iters_since_change < 100:
        rng, cl_rng = jax.random.split(rng)

        start = time.time()
        n_evals, done = clusterer.cluster(cl_rng, max_iter=iters)
        t += time.time() - start

        metrics = cluster_metrics(clusterer.list_cluster_labels(), ground_truth)
        metrics["LL"] = sum([clusterer.score_cache[cl.hash] for cl in clusterer.clusters])
        metrics["LP"] = clusterer.objective
        metrics["t"] = t
        total_iters += iters

        if clusterer.objective > best:
            best = clusterer.objective
            iters_since_change = 0
        else:
            iters_since_change += iters

        print(f"agg {batch_size} {t:.4f} {total_iters}\n {metrics['LL'], metrics['LP'], metrics['f1']}")

        for metric in metrics.keys():
            results[f"agg {batch_size} {total_iters}"][metric].append(metrics[metric])

        with open(f"data/gauss_agg_s{seed}.pickle", "wb") as handle:
            cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"agg {batch_size} final {total_iters}\n {metrics['LL'], metrics['LP'], metrics['f1']}")

    for metric in metrics.keys():
        results[f"agg {batch_size} final"][metric].append(metrics[metric])

    with open(f"data/gauss_agg_s{seed}.pickle", "wb") as handle:
        cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
