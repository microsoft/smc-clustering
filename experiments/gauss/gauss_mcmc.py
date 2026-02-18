#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle, collections, time, argparse
import cloudpickle
import jax
import numpy as np

import diffusion_linking
from diffusion_linking.mcmc_clustering import GibbsClusterer
from diffusion_linking.metrics import cluster_metrics

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=0)
args = parser.parse_args()
seed = args.seed

from gauss_setup import generate_gauss_dataset, prior, alpha, surrogate, batched_score_eval
data, labels = generate_gauss_dataset()
n = data.shape[0]

rng = jax.random.PRNGKey(seed)
shuffle_rng, rng = jax.random.split(rng)
shuffled_idx = jax.random.permutation(shuffle_rng, n)

labels = np.array(labels[shuffled_idx])
data = np.array(data[shuffled_idx])

ground_truth = [str(i) for i in labels]
    

results = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

steps = [1, 2, 5, 10, 20, 20, 50, 50, 50, 100] + [100]*100
score_cache=None
rng = jax.random.PRNGKey(seed)

clusterer = GibbsClusterer(data=data, score_fn=batched_score_eval, prior=prior, score_cache=score_cache)
t = 0
t_max = 10**4
best = -np.inf
iters_since_change = 0
for i, s in enumerate(steps):
    if t < t_max and iters_since_change < 500:
        rng, cl_rng = jax.random.split(rng)
        
        start = time.time()      
        evals = clusterer.cluster(cl_rng, sweeps=s)
        t += time.time() - start
        
        metrics = cluster_metrics(clusterer.list_cluster_labels(), ground_truth)
        metrics["LL"] = sum([clusterer.score_cache[cl.hash] for cl in clusterer.best])
        metrics['LP'] = clusterer.best_weight
        metrics ['t'] = t
        metrics['evals'] = evals
        metrics['total_evals'] = len(clusterer.score_cache)
    
        print(f'mcmc {sum(steps[:(i+1)])}\n {metrics["LL"], metrics["LP"], metrics["f1"]}')
        
        for metric in metrics.keys():
            results[f'mcmc {sum(steps[:(i+1)])}'][metric].append(metrics[metric])        
    
        with open(f'data/gauss_mcmc_s{seed}.pickle', 'wb') as handle:
            cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        if clusterer.best_weight > best:
            best = clusterer.best_weight
            iters_since_change = 0
        else:
            iters_since_change += s
            
for metric in metrics.keys():
    results[f'mcmc final'][metric].append(metrics[metric])        

with open(f'data/gauss_mcmc_s{seed}.pickle', 'wb') as handle:
    cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
