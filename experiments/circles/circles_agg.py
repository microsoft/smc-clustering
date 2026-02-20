#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle, collections, time, argparse
import cloudpickle
import jax
import numpy as np


import diffusion_linking
from diffusion_linking.clustering import Clusterer
from diffusion_linking.metrics import cluster_metrics

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=0)
args = parser.parse_args()
seed = args.seed

from circles_setup import load_model, generate_circles_dataset, prior, surrogate, alpha 
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

results = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
batch_sizes = [16, 32, 64, 128, len(data)]

for batch_size in batch_sizes:
    score_cache=None
    rng = jax.random.PRNGKey(seed)
    clusterer = Clusterer(data, batched_score_eval, prior=prior, cluster_batch_size=batch_size)
    t = 0
    done = False
    iters = 10
    total_iters = 0
    best = - np.inf
    iters_since_change = 0
    while (t < t_max and not done and iters_since_change < 100):
        rng, cl_rng = jax.random.split(rng)
    
        start = time.time()      
        _, done = clusterer.cluster(cl_rng, max_iter=iters)
        t += time.time() - start
        
        metrics = cluster_metrics(clusterer.list_cluster_labels(), ground_truth)
        metrics['evals'] = len(clusterer.score_cache)
        metrics["LL"] = sum([clusterer.score_cache[cl.hash] for cl in clusterer.clusters])
        metrics['LP'] = clusterer.objective
        metrics ['t'] = t
        total_iters += iters
        
        if clusterer.objective > best:
            best = clusterer.objective
            iters_since_change = 0
        else:
            iters_since_change += iters
    
        print(f'agg {batch_size} {t:.4f} {total_iters}\n {metrics["LL"], metrics["LP"], metrics["f1"], metrics["evals"]}')
        
        for metric in metrics.keys():
            results[f'agg {batch_size} {total_iters}'][metric].append(metrics[metric])     
    
        with open(f'data/circles_agg_s{seed}.pickle', 'wb') as handle:
            cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(f'agg final {t:.4f} {total_iters}\n {metrics["LL"], metrics["LP"], metrics["f1"], metrics["evals"]}')
    
    for metric in metrics.keys():
        results[f'agg {batch_size} final'][metric].append(metrics[metric])     

    with open(f'data/circles_agg_s{seed}.pickle', 'wb') as handle:
        cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
