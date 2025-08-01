import collections
import gc

import jax
import numpy as np
import torch
from disk.data.loader import RecordLoader
from disk.data.mapper import map_entities
from disk.model.lightning import DiSKLightning
from disk.utils.transform import CleanPyGGraph
from jax import numpy as jnp
from kebab.contracts.entity import Entity
from torch_geometric.profile.utils import get_data_size
from tqdm import tqdm

from diffusion_linking.clustering import Cluster, Clusterer
from diffusion_linking.smc_clustering import SMCClusterer
from diffusion_linking.surrogate_models import Bigram, get_ngram_counts


def gpu_tensors():
    """Prints the sizes of all GPU tensors in the current process."""
    data = []
    for o in gc.get_objects():
        if torch.is_tensor(o) and o.is_cuda:
            data.append((o.numel() * o.element_size(), o.shape, o.dtype, o.requires_grad))

    for s, sh, dt, rg in sorted(data, reverse=True)[:20]:
        if s < 1024**2 * 8:
            continue

        print(f"{s / 1024**2:7.1f} MB  shape={sh} dtype={dt} grad={rg}")


def estimate_lls(
    entities: list[Entity],
    model: DiSKLightning,
    num_samples: int,
    apply_arity_weights: bool = True,
    device: str = "cpu",
    batch_size: int = 200,
    num_workers: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Estimates entity log probabilities. From disk.linking.score.estimate_linking_scores

    Args:
        entities (list[Entity]): The entities.
        model (DiSKLightning): The DiSK model to use.
        num_samples (int): Number of samples to estimate log probabilities.
        apply_arity_weights: (bool, optional):  Whether to apply the arity weights to the log probabilities.
        Defaults to False.
        device (str, optional): The device to use. Defaults to "cpu".
        batch_size (int, optional): Batch size for data loading. Defaults to 200.
        num_workers (int, optional): Number of workers for loading data. Defaults to 0.
        verbose (bool, optional): Display progress bar if true. Defaults to False.

    Returns:
        np.ndarray: The linking scores as a 1-d numpy array.
    """
    # map to RecordData
    record_data = map_entities(entities, model.schema, model.stats)

    # Build data loader if necessary
    if record_data.num_records <= batch_size:
        loader = [record_data]
    else:
        loader = RecordLoader(
            data=record_data,
            input_nodes=np.arange(record_data.num_records),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            transform=CleanPyGGraph(),
        )

    model.to(device)
    model.eval()

    # get all log probs in one array
    all_log_probs = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Computing Linking Scores", disable=not verbose):
            data.to(device)

            if verbose:
                mb = get_data_size(data)  # bytes
                print(f"Batch takes {mb / 1024**2:.2f} MB, {data.num_nodes} nodes, {data.num_edges} edges")
                gpu_tensors()

            log_probs, std = model.model.monte_carlo_log_probs(
                data=data,
                num_samples=num_samples,
                apply_arity_weights=apply_arity_weights,
                return_std=True,
            )
            log_probs = log_probs.detach().cpu().numpy()
            all_log_probs.append(log_probs)
    log_probs = np.concatenate(all_log_probs, axis=0)

    return log_probs


class ListWrapper:
    # Allows easier retrieval of cluster data from lists
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        return (len(self.data),)

    def __getitem__(self, row_ids):
        if type(row_ids) is int:
            return [self.data[row_ids]]
        if type(row_ids) is slice:
            return self.data[row_ids]
        else:
            return [self.data[idx] for idx in row_ids]


class NameBigram(Bigram):
    """
    Retrieves name property for use in the bigram model
    """

    def __init__(self, prior_scale, prior_counts):
        super().__init__(prior_scale, prior_counts)

    def post_predictive(self, obs, n, summary):
        if type(obs) is list:
            name = obs[0].properties["name"]
        else:
            name = obs.properties["name"]

        return super().post_predictive(name, n, summary)


class NameBigramCluster(Cluster):
    """
    Cluster subclass with summary statistics for an n-gram model, looks at name property for counts
    """

    def __init__(self, data_ids, n=2, counts=None, data=None):
        super().__init__(data_ids)
        self.n = n
        if counts is not None:
            self.counts = counts
        elif data is not None:
            self.counts = get_ngram_counts([entity.properties["name"] for entity in data], self.n)
        else:
            self.counts = collections.Counter()

    @property
    def summary(self):
        return self.counts

    def merge_point(self, data_id, data):
        new_counts = self.counts + get_ngram_counts([entity.properties["name"] for entity in data], self.n)
        return NameBigramCluster(self.data.union({data_id}), self.n, counts=new_counts)


def summary(clusterer, problems=None, max_print=3, min_problem_size=1, print_summary=True):
    """
    Summarise clustering, prints out entity names
    """
    if isinstance(clusterer, SMCClusterer):
        summary_text = ""
        if problems is None:
            problems = range(len(clusterer.state.particles))
            n_points = [
                sum([clusterer.state.clusters[c].size for c in clusterer.state.particles[p][0]]) for p in problems
            ]
            summary_text += f"{len(problems)} subproblems of sizes {sorted(n_points, reverse=True)}"

        for p in problems:
            n_points = sum([clusterer.state.clusters[c].size for c in clusterer.state.particles[p][0]])
            if n_points >= min_problem_size:
                summary_text += f"\nSubproblem {p}: {len(clusterer.state.particles[p])} particles, {n_points} points"
                for i, (particle, weight) in enumerate(
                    sorted(
                        zip(clusterer.state.particles[p], clusterer.state.weights[p]), key=lambda c: c[1], reverse=True
                    )
                ):
                    if i >= max_print:
                        summary_text += f"\n\t... {len(clusterer.state.particles[p]) - max_print} particles omitted"
                        break
                    exp_weight = jnp.exp(weight - jax.scipy.special.logsumexp(jnp.array(clusterer.state.weights[p])))
                    clusters = sorted(particle, key=lambda c: clusterer.state.clusters[c].size, reverse=True)
                    summary_text += f"\n\tParticle {i}, weight {exp_weight:.2g}({weight:.3g}), {len(clusters)} clusters, {[clusterer.state.clusters[c].size for c in clusters]}"
                    summary_text += "\n"

                    if clusterer.print_cluster_data:
                        for c in clusters:
                            summary_text += (
                                f"\n\t\tLL:{clusterer.state.score_cache[c]:.2g}, " if clusterer.max_evals > 0 else ""
                            )
                            summary_text += str(
                                [entity.properties["name"] for entity in clusterer.state.retrieve_cluster_data(c)]
                            )
                            summary_text += "\n"

                        summary_text += "\n"
                summary_text += "\n"

        print(summary_text)

    elif isinstance(clusterer, Clusterer):
        clusters = sorted(clusterer.clusters, key=lambda c: c.size, reverse=True)
        print(f"{len(clusters)} clusters, {clusterer.data.shape[0]} points, {[c.size for c in clusters]}")
        for c in clusters:
            print(f"LL:{clusterer.score_cache[c.hash]:.2g}, ")
            print([entity.properties["name"] for entity in clusterer.data[c.ids]])
        print()


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
