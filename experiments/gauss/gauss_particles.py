#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle, collections, time, argparse

import cloudpickle
import jax
import scipy.special
import numpy as np

import smc_clustering
from smc_clustering.surrogate_models import GaussianCluster
from smc_clustering.smc import SMCClusterer, resample_greedy, plot_particles_2D
from smc_clustering.metrics import cluster_metrics

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

##################### 

surrogate_threshold = None
model_threshold = None

ps = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 3000]

results = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

configs={'smc': (None,None,1,False),
         'split': (1,None,1,False)
         }

resample_fn = resample_greedy
t_max = 60*60
score_cache=None

for conf in configs.keys():
    t = 0
    for p in ps:
        if t < t_max:
            max_particles = p; max_evals = 0         
            split_interval, surrogate_threshold, split_quantile, entropy_condition = configs[conf]
            method=f'{p}, {conf}'
            clusterer = SMCClusterer(data=data, split_interval=split_interval, entropy_condition=entropy_condition, split_quantile=split_quantile, surrogate_threshold=surrogate_threshold, model_threshold=model_threshold, score_fn=batched_score_eval, max_particles=max_particles, max_evals = max_evals, prior = prior, surrogate = surrogate, resample_fn=resample_fn, ClusterClass=GaussianCluster, callback=plot_particles_2D, score_cache=score_cache)
            rng = jax.random.PRNGKey(seed)
            
            t = time.time()
            evals, subprobs = clusterer.cluster(rng,callback_interval=0)    
            t = time.time() - t
            
            ll = sum([clusterer.state.score_cache[h] for s in range(len(clusterer.state.particles)) for h in clusterer.state.particles[s][np.argmax(clusterer.state.weights[s])] ])
            lp = ll + sum([ np.log(alpha) + scipy.special.gammaln(clusterer.state.clusters[h].size) for s in range(len(clusterer.state.particles)) for h in clusterer.state.particles[s][np.argmax(clusterer.state.weights[s])] ])
            
            metrics = cluster_metrics(clusterer.state.list_cluster_labels(), ground_truth[:clusterer.state.n_obs])
            metrics["LL"] = ll
            metrics['LP'] = lp
            metrics["evals"] = np.mean(evals)
            metrics['subprobs'] = np.array(subprobs)
            metrics['t'] = t
            print(f'{method}\n {t:.4g} {metrics["LP"], metrics["f1"]}')
            
            for metric in metrics.keys():
                results[method][metric].append(metrics[metric])
    
            with open(f'data/gauss_particles_s{seed}.pickle', 'wb') as handle:
                cloudpickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
