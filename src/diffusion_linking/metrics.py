# Licensed under the MIT license.
import collections
import numpy as np

def cluster_precision(pred_cluster_map, gt_cluster_map):
    fragment_count = len(gt_cluster_map)
    precision = 0
    for i in gt_cluster_map.keys():
        tp = len(pred_cluster_map[i].intersection(gt_cluster_map[i]))
        precision += (tp / len(pred_cluster_map[i]))
    return precision / fragment_count

def cluster_recall(pred_cluster_map, gt_cluster_map):
    fragment_count = len(gt_cluster_map)
    recall = 0
    for i in gt_cluster_map.keys():
        tp = len(pred_cluster_map[i].intersection(gt_cluster_map[i]))
        recall += (tp / len(gt_cluster_map[i]))
    return recall / fragment_count

def cluster_f1(pred_cluster_map, gt_cluster_map):
    fragment_count = len(gt_cluster_map)
    f1 = 0
    for i in gt_cluster_map.keys():
        tp = len(pred_cluster_map[i].intersection(gt_cluster_map[i]))
        fp = len(pred_cluster_map[i].difference(gt_cluster_map[i]))
        fn = len(gt_cluster_map[i].difference(pred_cluster_map[i]))
        f1 += tp / (tp + 0.5 * (fp + fn))
    return f1 / fragment_count
    
def cluster_metrics(predictions, ground_truth) -> dict[str, float]:
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
    
