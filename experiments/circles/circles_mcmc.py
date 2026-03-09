# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import collections
import pickle
import time

import cloudpickle
import jax
import numpy as np

from smc_clustering.clustering.mcmc import GibbsClusterer
from smc_clustering.clustering.metrics import cluster_metrics
from smc_clustering.clustering.surrogate_models import GaussianCluster


parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=0)
args = parser.parse_args()
seed = args.seed

from circles_setup import generate_circles_dataset, load_model, prior, surrogate


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


t_max = 10**4
results = collections.defaultdict(lambda: collections.defaultdict(list))

score_cache = None
rng = jax.random.PRNGKey(seed)
steps = [1, 2, 5, 10] + [10] * 2000

score_cache = None
rng = jax.random.PRNGKey(seed)
clusterer = GibbsClusterer(
    data=data,
    score_fn=batched_score_eval,
    prior=prior,
    surrogate=None,
    score_cache=score_cache,
    ClusterClass=GaussianCluster,
)
t = 0
iters_since_change = 0
best = -np.inf
for i, s in enumerate(steps):
    if t < t_max and iters_since_change < 500:
        rng, cl_rng = jax.random.split(rng)

        start = time.time()
        evals = clusterer.cluster(cl_rng, sweeps=s)
        t += time.time() - start

        metrics = cluster_metrics(clusterer.list_cluster_labels(), ground_truth)
        metrics["LL"] = sum([clusterer.score_cache[cl.hash] for cl in clusterer.best_clustering])
        metrics["LP"] = clusterer.best_logpost
        metrics["t"] = t
        metrics["evals"] = evals
        metrics["total_evals"] = len(clusterer.score_cache)

        print(
            f"mcmc gibbs {t:4f} {sum(steps[: (i + 1)])}\n {metrics['LL'], metrics['LP'], metrics['f1']}"
        )

        for metric in metrics.keys():
            results[f"mcmc gibbs {sum(steps[: (i + 1)])}"][metric].append(metrics[metric])

        with open(f"data/circles_mcmc_s{seed}.pickle", "wb") as handle:
            cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if clusterer.best_logpost > best:
            best = clusterer.best_logpost
            iters_since_change = 0
        else:
            iters_since_change += s

for metric in metrics.keys():
    results["mcmc gibbs final"][metric].append(metrics[metric])

with open(f"data/circles_mcmc_s{seed}.pickle", "wb") as handle:
    cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


score_cache = None
rng = jax.random.PRNGKey(seed)
clusterer = GibbsClusterer(
    data=data,
    score_fn=batched_score_eval,
    prior=prior,
    surrogate=surrogate,
    score_cache=score_cache,
    ClusterClass=GaussianCluster,
)
t = 0
iters_since_change = 0
best = -np.inf
for i, s in enumerate(steps):
    if t < t_max and iters_since_change < 500:
        rng, cl_rng = jax.random.split(rng)

        start = time.time()
        evals = clusterer.cluster(cl_rng, sweeps=s)
        t += time.time() - start

        metrics = cluster_metrics(clusterer.list_cluster_labels(), ground_truth)
        metrics["LL"] = sum([clusterer.score_cache[cl.hash] for cl in clusterer.best_clustering])
        metrics["LP"] = clusterer.best_logpost
        metrics["t"] = t
        metrics["evals"] = evals
        metrics["total_evals"] = len(clusterer.score_cache)

        print(f"mcmc mh {t:4f} {sum(steps[: (i + 1)])}\n {metrics['LL'], metrics['LP'], metrics['f1']}")

        for metric in metrics.keys():
            results[f"mcmc mh {sum(steps[: (i + 1)])}"][metric].append(metrics[metric])

        with open(f"data/circles_mcmc_s{seed}.pickle", "wb") as handle:
            cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if clusterer.best_logpost > best:
            best = clusterer.best_logpost
            iters_since_change = 0
        else:
            iters_since_change += s

for metric in metrics.keys():
    results["mcmc mh final"][metric].append(metrics[metric])

with open(f"data/circles_mcmc_s{seed}.pickle", "wb") as handle:
    cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
