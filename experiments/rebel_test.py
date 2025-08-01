#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import collections
import json
import os
import pickle
import sys
from math import ceil

import numpy as np
import pandas as pd


wdir = os.getcwd()

import warnings

import torch


warnings.filterwarnings("ignore")  # avoid torchmetrics warning

import jax
import jax.numpy as jnp
from disk.data.mapper import map_frame
from disk.data.stats import TextStats
from disk.dataset.wikidata_names import WikidataNames
from disk.model.disk import DiSK
from kebab import mskebab

from diffusion_linking.smc_clustering import DirichletProcess, SMCClusterer, resample_optimal
from diffusion_linking.surrogate_models import Bigram, BigramCluster, CountDict
from diffusion_linking.utils import DFWrapper


parser = argparse.ArgumentParser()
parser.add_argument("-ds_size", type=int)
parser.add_argument("-n_particles", type=int, default=100)
parser.add_argument("-max_evals", type=int)

args = parser.parse_args()
n_particles = args.n_particles
if args.max_evals is None:
    max_evals = n_particles
else:
    max_evals = args.max_evals
ds_size = args.ds_size


def cluster_metrics(predictions, ground_truth) -> dict[str, float]:
    """From kebab.tasks.clustering, modified to allow evaluation for subsets of the data"""
    fragment_count = len(predictions)
    metrics = collections.defaultdict(float)

    # construct the predicted {element_idx -> set of element_idx} map
    pred_clusters = collections.defaultdict(set)
    pred_cluster_map = {}
    for i, cluster_id in enumerate(predictions):
        cluster = pred_clusters[cluster_id]
        cluster.add(i)
        pred_cluster_map[i] = cluster

    # construct the ground truth {element_idx -> set of element_idx} map
    gt_clusters = collections.defaultdict(set)
    gt_cluster_map = {}
    for i, cluster_id in enumerate(ground_truth):
        cluster = gt_clusters[cluster_id]
        cluster.add(i)
        gt_cluster_map[i] = cluster

    # compute BCubed P,R and F1
    precisions = []
    recalls = []
    f1s = []
    for i in range(fragment_count):
        tp = len(pred_cluster_map[i].intersection(gt_cluster_map[i]))
        fp = len(pred_cluster_map[i].difference(gt_cluster_map[i]))
        fn = len(gt_cluster_map[i].difference(pred_cluster_map[i]))

        precision = tp / len(pred_cluster_map[i])
        recall = tp / len(gt_cluster_map[i])
        f1 = tp / (tp + 0.5 * (fp + fn))

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics["precision"] = float(np.mean(precisions, dtype=np.float64))
    metrics["recall"] = float(np.mean(recalls, dtype=np.float64))
    metrics["f1"] = float(np.mean(f1s, dtype=np.float64))

    metrics = dict(metrics)
    return metrics


def estimate_probs(
    df: pd.DataFrame,
    model: DiSK,
    num_samples: int,
    device: str = "cpu",
    batch_size: int = 200,
    num_workers: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    df.reset_index(inplace=True)

    if len(df) <= batch_size:
        record_data = map_frame(df, model.schema, model.stats)
        loader = [record_data]
    else:
        loader = []
        for b in range(ceil(len(df) / batch_size)):
            record_data = map_frame(
                df.iloc[(b * batch_size) : ((b + 1) * batch_size)].reset_index(), model.schema, model.stats
            )
            loader.append(record_data)

    model.to(device)
    disk_model = model.model

    # get all log probs in one array
    all_log_probs = []
    with torch.no_grad():
        for data in loader:
            data.to(device)
            log_probs = disk_model.monte_carlo_log_probs(data, num_samples=num_samples)
            log_probs = log_probs.detach().cpu().numpy()
            all_log_probs.append(log_probs)
    log_probs = np.concatenate(all_log_probs, axis=0)

    return log_probs


device = "cuda:0" if torch.cuda.is_available() else "cpu"
schema = WikidataNames.schema
stats = {"names": TextStats("names", "t5-small", 32128)}

checkpoint = "/data/DiSK_Autoregressive/checkpoints/best.ckpt"
model = DiSK.load_from_checkpoint(wdir + checkpoint, schema=schema, stats=stats)
model.to(device)
model.model.use_diffusion_weights = True


def batched_score_eval(rng, cluster_data, num_samples=10, batch_size=10):
    # Converts each cluster's data from a list of lists of names
    # to a single list containing all of the names from each fragment in the cluster
    df = pd.DataFrame.from_records([{"names": sum(names, start=[])} for names in cluster_data])
    linking_scores = estimate_probs(df, model, num_samples, device, batch_size=batch_size)
    return linking_scores


benchmark = mskebab.Benchmark("data/sampled_rebel_clustering.json")
task = benchmark.task_instances["Clustering-Heldout-Small"]

if ds_size is not None:
    dataset = DFWrapper(
        pd.DataFrame.from_records(
            [{"name": fragment[0].properties["name"]} for fragment in task.read_items()][:ds_size]
        )
    )

    ground_truth = []
    with open("data/sampled_rebel_clustering_ground_truth.jsonl", "rb") as handle:
        i = 0
        for line in handle:
            ground_truth.append(json.loads(line))
            i += 1
            if i >= ds_size:
                break
else:
    dataset = DFWrapper(
        pd.DataFrame.from_records([{"name": fragment[0].properties["name"]} for fragment in task.read_items()])
    )


with open("data/wikipedia_names_2gram_counts.pickle", "rb") as handle:
    count_dict = pickle.load(handle)
prior_counts = CountDict(count_dict["<UNK>"], count_dict)

prior = DirichletProcess(10)
surrogate = Bigram(0.01, prior_counts)

score_cache = None
max_particles = n_particles
split_interval = 1
surrogate_threshold = 10
model_threshold = None
resample_fn = resample_optimal
rng = jax.random.PRNGKey(1)
clusterer = SMCClusterer(
    data=dataset,
    split_interval=split_interval,
    surrogate_threshold=surrogate_threshold,
    model_threshold=model_threshold,
    score_fn=batched_score_eval,
    max_particles=max_particles,
    max_evals=max_evals,
    prior=prior,
    surrogate=surrogate,
    resample_fn=resample_fn,
    ClusterClass=BigramCluster,
    score_cache=score_cache,
    print_cluster_data=True,
)
clusterer.cluster(rng, verbose=False)

cluster_list = clusterer.state.list_cluster_labels()
with open(
    f"data/smc_clusterer_output_{n_particles}-particles_{max_evals}-evals_{dataset.shape[0]}-datapoints.jsonl", "w"
) as f:
    f.write("\n".join(cluster_list))

if ds_size is not None:
    print(cluster_metrics(cluster_list, ground_truth))
else:
    print(task.evaluate("data/smc_clusterer_output.jsonl"))
