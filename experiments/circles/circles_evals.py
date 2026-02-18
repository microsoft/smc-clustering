#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, random, pickle, collections, sys, math, argparse, time

import cloudpickle
import jax
import scipy.special
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from flax.training import checkpoints


import diffusion_linking
from diffusion_linking.utils import batched_eval
from diffusion_linking.circles import generate_circles
from diffusion_linking.schedule import LinearSchedule
from diffusion_linking.diffusion import VariationalDiffusion
from diffusion_linking.surrogate_models import Gaussian, GaussianCluster
from diffusion_linking.smc_clustering import SMCClusterer, DirichletProcess, resample_optimal, resample_greedy, plot_particles_2D
from diffusion_linking.metrics import cluster_metrics

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=0)
args = parser.parse_args()
seed = args.seed

from circles_setup import *

batched_score_eval = load_model()
data, labels = generate_circles_dataset()

n = data.shape[0]
rng = jax.random.PRNGKey(seed)
shuffle_rng, rng = jax.random.split(rng)
shuffled_idx = jax.random.permutation(shuffle_rng, n)
labels = labels[shuffled_idx]
data = data[shuffled_idx]

ground_truth = [str(i) for i in labels]
gt_clusters = collections.defaultdict(set)
gt_cluster_map = {}
for i, cluster_id in enumerate(ground_truth):
    cluster = gt_clusters[cluster_id]
    cluster.add(i)
    gt_cluster_map[i] = cluster
    
##################### 


surrogate_threshold = None
model_threshold = None
evals = [0, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000]
p = 50
results = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

configs={
         'smc': (None,None,1,False,False),
         'split': (1,None,1,False,False),
        'greedy': (None,None,1,False,False),
         }

resample_fn = resample_greedy
score_cache=None
for e in evals:
    for conf in configs.keys():
        if conf=='greedy' or e >= p or e == 0:
            max_particles = 1 if conf=='greedy' else p; max_evals = e          
            split_interval, surrogate_threshold, split_quantile, entropy_condition, reweight_splits = configs[conf]
            method=f'{e}, {resample_fn}, {conf}'
            clusterer = SMCClusterer(data=data, split_interval=split_interval, entropy_condition=entropy_condition, split_quantile=split_quantile, surrogate_threshold=surrogate_threshold, model_threshold=model_threshold, score_fn=batched_score_eval, max_particles=max_particles, max_evals = max_evals, prior = prior, surrogate = surrogate, resample_fn=resample_fn, ClusterClass=GaussianCluster, callback=plot_particles_2D, score_cache=score_cache)
            rng = jax.random.PRNGKey(seed)
            
            t = time.time()
            evals, subprobs = clusterer.cluster(rng,callback_interval=0)    
            t = time.time() - t
            
            ll = sum([clusterer.state.score_cache[h] for s in range(len(clusterer.state.particles)) for h in clusterer.state.particles[s][np.argmax(clusterer.state.weights[s])] ])
            lp = ll + sum([ np.log(alpha) + scipy.special.gammaln(clusterer.state.clusters[h].size) for s in range(len(clusterer.state.particles)) for h in clusterer.state.particles[s][np.argmax(clusterer.state.weights[s])] ])
    
            
            metrics = cluster_metrics(clusterer.state.list_cluster_labels(), ground_truth[:clusterer.state.n_obs])
            metrics["LL"] = ll
            metrics["LP"] = lp
            metrics['total_evals'] = len(clusterer.state.score_cache)
            metrics['t'] = t
            print(f'{method}\n {t:.4g} {metrics["LL"], metrics["f1"]}')
            print(sum([clusterer.state.score_cache[h] for s in range(len(clusterer.state.particles)) for h in clusterer.state.particles[s][np.argmax(clusterer.state.weights[s])] ]))
            
            for metric in metrics.keys():
                results[method][metric].append(metrics[metric])
    
            with open(f'data/circles_evals_s{seed}.pickle', 'wb') as handle:
                cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
