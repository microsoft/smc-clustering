# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import scipy


class Cluster:
    """Represents a cluster as a frozen set of datapoint ids."""

    def __init__(self, data_ids: Iterable[int], **kwargs: Any):
        self.data = frozenset(data_ids)
        self.size = len(data_ids)

    @property
    def ids(self) -> np.ndarray:
        # Convert to numpy array for easier retrieval of datapoints
        return np.fromiter(self.data, dtype=np.int64)

    @property
    def hash(self) -> int:
        return hash(self.data)

    @property
    def summary(self) -> list[Any]:
        return []

    def add(self, data_id: int) -> frozenset[int]:
        return self.data.union({data_id})

    def merge_point(self, data_id: int, data: np.ndarray) -> Cluster:
        return Cluster(self.data.union({data_id}))

    def merge(self, other: Cluster) -> Cluster:
        return Cluster(self.data.union(other.data))


class Uniform:
    """Uniform prior on cluster sizes"""

    def __call__(self, cluster_sizes: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.zeros((1,))

    def marginal(self, n_obs: int, cluster_size: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.zeros(cluster_size.shape[0])


class DirichletProcess:
    """Dirichlet process prior on cluster sizes"""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, cluster_sizes: np.ndarray, **kwargs: Any) -> float:
        # Prior probability of a clustering, based on cluster sizes and hyperparameter alpha
        return (
            len(cluster_sizes) * np.log(self.alpha) + np.sum(scipy.special.gammaln(cluster_sizes))
        ).item()

    def marginal(self, n_obs: int, cluster_size: np.ndarray, **kwargs: Any) -> np.ndarray:
        # Prior probability of a single assignment, based on cluster size and hyperparameter alpha
        return np.where(cluster_size > 0, np.log(cluster_size), np.log(self.alpha))
