#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, pickle, collections, argparse, pathlib
wdir = os.getcwd()

import jax
import numpy as np
from pathlib import Path

import torch
import warnings
warnings.filterwarnings("ignore") # avoid torchmetrics warning
os.environ["TOKENIZERS_PARALLELISM"] = "False"  # Avoid huggingface warnings

from kebab import mskebab
from kebab.utils.io_helpers import ItemJsonlReader
from kebab.contracts.entity import Entity
from disk.model.lightning import DiSKLightning

from diffusion_linking.smc_clustering import SMCClusterer, DirichletProcess, resample_optimal, resample_greedy
from diffusion_linking.surrogate_models import CountDict
from DiSK_utils import ListWrapper, NameBigram, NameBigramCluster, estimate_lls, summary, cluster_metrics


parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-reps', type=int, default=1)
parser.add_argument('-ds_size', type=int, default=None)
parser.add_argument('-checkpoint_path', type=str, default='/data/best.ckpt')


parser.add_argument('-alpha', type=float)
parser.add_argument('-prior_scale', type=float, default=25)
parser.add_argument('-max_particles', type=int, default=300)
parser.add_argument('-max_evals', type=int, default=None)
parser.add_argument('-threshold', type=float, default=None)
parser.add_argument('-split_interval', type=int, default=None)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

checkpoint = wdir + args.checkpoint_path
checkpoint = pathlib.Path(checkpoint).expanduser()
model = DiSKLightning.load_from_checkpoint(checkpoint)

def batched_score_eval(rng, cluster_data, num_samples = 10, batch_size = 10):
    '''
    Estimates cluster log-likelihoods
    rng : unused, rng key for JAX models
    cluster_data : list[list[Entity]], list of the fragments in each cluster
    '''
    entities = [Entity.merge(cl) if len(cl)>1 else cl[0] for cl in cluster_data]
    linking_scores = estimate_lls(entities, model, num_samples, device=device, batch_size=batch_size)    
    return linking_scores


benchmark = mskebab.Benchmark(Path("config/benchmark_conf.json"))
task = benchmark.tasks_by_name["Clustering-REBEL-Small"]

if args.ds_size is not None:
    data = ListWrapper([fragment[0] for fragment in task.read_items() ][:args.ds_size])
    ground_truth = list(ItemJsonlReader[str](task.data_ground_truth_labels).read_items())[:args.ds_size]
else:
    data = ListWrapper([fragment[0] for fragment in task.read_items() ])
    ground_truth = list(ItemJsonlReader[str](task.data_ground_truth_labels).read_items())
    
 
with open('data/wikipedia_names_2gram_counts.pickle', 'rb') as handle:
    count_dict = pickle.load(handle)
prior_counts = CountDict(count_dict['<UNK>'], count_dict)

prior = DirichletProcess(args.alpha)
surrogate = NameBigram(args.prior_scale, prior_counts)

p = args.max_particles
max_evals = np.inf if args.max_evals is None else args.max_evals
    
results = collections.defaultdict(lambda: [])
score_cache = None
seeds = [i for i in range(args.seed, args.seed + args.reps)]

for seed in seeds:
    rng = jax.random.PRNGKey(seed)
    torch.manual_seed(seed)
    clusterer = SMCClusterer(data=data, split_interval=args.split_interval, surrogate_threshold=None, model_threshold=args.threshold, score_fn=batched_score_eval, max_particles=p, max_evals = max_evals, prior = prior, surrogate = surrogate, resample_fn=resample_optimal, ClusterClass=NameBigramCluster, callback=None, score_cache=score_cache)
    n_evals, _ = clusterer.cluster(rng, callback_interval=0)
    clustering = clusterer.state.list_cluster_labels()
    metrics = cluster_metrics(clustering, ground_truth)
    if n_evals[-1] > 0:
        ll = sum([ clusterer.state.score_cache[h] for s in range(len(clusterer.state.particles)) for h in clusterer.state.particles[s][np.argmax(clusterer.state.weights[s])] ])
        metrics["LL"] = ll
    metrics['evals'] = len(clusterer.state.score_cache)
    if args.reps > 1:
        print(metrics)
    for metric in metrics.keys():
        results[metric].append(metrics[metric])

print(vars(args), '\n\n', f'Final from {args.reps} reps:', '\n', [f'{metric}: {np.mean(results[metric])}' for metric in metrics.keys()])

