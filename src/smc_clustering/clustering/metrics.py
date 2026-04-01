# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Metrics for evaluating clustering assignments.

These helpers compute B-Cubed-style precision, recall, and F1 statistics from predicted and ground-truth cluster labels.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from typing import Any

import numpy as np


def cluster_precision(
    pred_cluster_map: dict[int, set[int]], gt_cluster_map: dict[int, set[int]]
) -> float:
    """Compute B-Cubed precision for a clustering."""
    fragment_count = len(gt_cluster_map)
    precision = 0
    for item_id, gt_cluster in gt_cluster_map.items():
        tp = len(pred_cluster_map[item_id].intersection(gt_cluster))
        precision += tp / len(pred_cluster_map[item_id])
    return precision / fragment_count


def cluster_recall(pred_cluster_map: dict[int, set[int]], gt_cluster_map: dict[int, set[int]]) -> float:
    """Compute B-Cubed recall for a clustering."""
    fragment_count = len(gt_cluster_map)
    recall = 0
    for item_id, gt_cluster in gt_cluster_map.items():
        tp = len(pred_cluster_map[item_id].intersection(gt_cluster))
        recall += tp / len(gt_cluster)
    return recall / fragment_count


def cluster_f1(pred_cluster_map: dict[int, set[int]], gt_cluster_map: dict[int, set[int]]) -> float:
    """Compute B-Cubed F1 for a clustering."""
    fragment_count = len(gt_cluster_map)
    f1 = 0
    for item_id, gt_cluster in gt_cluster_map.items():
        tp = len(pred_cluster_map[item_id].intersection(gt_cluster))
        fp = len(pred_cluster_map[item_id].difference(gt_cluster))
        fn = len(gt_cluster.difference(pred_cluster_map[item_id]))
        f1 += tp / (tp + 0.5 * (fp + fn))
    return f1 / fragment_count


def cluster_metrics(predictions: Sequence[Any], ground_truth: Sequence[Any]) -> dict[str, float]:
    """Evaluate an output for the clustering task instance."""
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
