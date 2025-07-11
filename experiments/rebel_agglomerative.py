#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, pickle, collections, argparse, pathlib
wdir = os.getcwd()

import jax
import numpy as np

import torch
import warnings
warnings.filterwarnings("ignore") # avoid torchmetrics warning
os.environ["TOKENIZERS_PARALLELISM"] = "False"  # Avoid huggingface warnings

from kebab import mskebab
from kebab.utils.io_helpers import ItemJsonlReader
from kebab.contracts.entity import Entity
from disk.model.lightning import DiSKLightning

from diffusion_linking.clustering import Clusterer
from diffusion_linking.surrogate_models import CountDict
from DiSK_utils import ListWrapper, estimate_lls, summary, cluster_metrics


parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-reps', type=int, default=1)
parser.add_argument('-ds_size', type=int, default=None)
parser.add_argument('-checkpoint_path', type=str, default='/data/best.ckpt')


parser.add_argument('-batch_size', type=int, default=None)
parser.add_argument('-threshold', type=float, default=0)
parser.add_argument('-max_iter', type=int, default=None)

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

benchmark = mskebab.Benchmark('data/sampled_rebel_clustering.json')
task = benchmark.tasks_by_name['Clustering-Heldout-Small']

if args.ds_size is not None:
    data = ListWrapper([fragment[0] for fragment in task.read_items() ][:args.ds_size])
    ground_truth = list(ItemJsonlReader[str](task.data_ground_truth_labels).read_items())[:args.ds_size]
else:
    data = ListWrapper([fragment[0] for fragment in task.read_items() ])
    ground_truth = list(ItemJsonlReader[str](task.data_ground_truth_labels).read_items())
    
batch_size = data.shape[0] if args.batch_size is None else args.batch_size
max_iter = data.shape[0] if args.max_iter is None else args.max_iter

results = collections.defaultdict(lambda: [])
score_cache = None
seeds = [i for i in range(args.seed, args.seed + args.reps)]

for seed in seeds:
    rng = jax.random.PRNGKey(seed)
    torch.manual_seed(seed)
    clusterer = Clusterer(data, batched_score_eval, cluster_batch_size=batch_size, link_threshold=args.threshold)
    clusterer.cluster(rng, max_iter=max_iter)
    cluster_list = clusterer.list_cluster_labels()
    metrics = cluster_metrics(cluster_list, ground_truth)
    metrics["LL"] = sum([clusterer.score_cache[cl.hash] for cl in clusterer.clusters])
    metrics['evals'] = len(clusterer.score_cache)
    if args.reps > 1:
        print(metrics)
    for metric in metrics.keys():
        results[metric].append(metrics[metric])

print(vars(args), '\n\n', f'Final from {args.reps} reps:', '\n', [f'{metric}: {np.mean(results[metric])}' for metric in metrics.keys()])

