# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Core cluster containers and cluster-size priors.

The classes in this module represent immutable clusters and provide prior terms used by the clustering algorithms.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import scipy


class Cluster:
    """Represents a cluster as a frozen set of datapoint ids."""

    def __init__(self, data_ids: Iterable[int], **kwargs: Any):
        """Initialize Cluster with the given data identifiers."""
        self.data = frozenset(data_ids)
        self.size = len(data_ids)

    @property
    def ids(self) -> np.ndarray:
        # Convert to numpy array for easier retrieval of datapoints
        """Return the data-point identifiers as a NumPy array."""
        return np.fromiter(self.data, dtype=np.int64)

    @property
    def hash(self) -> int:
        """Return a stable hash for the cluster contents."""
        return hash(self.data)

    @property
    def summary(self) -> list[Any]:
        """Print a summary of the current clustering state."""
        return []

    def add(self, data_id: int) -> frozenset[int]:
        """Return the cluster contents with one additional data point."""
        return self.data.union({data_id})

    def merge_point(self, data_id: int, data: np.ndarray) -> Cluster:
        """Return a new cluster after adding the given data point."""
        return Cluster(self.data.union({data_id}))

    def merge(self, other: Cluster) -> Cluster:
        """Return a new cluster formed by merging both clusters."""
        return Cluster(self.data.union(other.data))


class Uniform:
    """Uniform prior on cluster sizes."""

    def __call__(self, cluster_sizes: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return a constant prior score for the provided cluster sizes."""
        return np.zeros((1,))

    def marginal(self, n_obs: int, cluster_size: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return the marginal prior term for candidate cluster sizes."""
        return np.zeros(cluster_size.shape[0])


class DirichletProcess:
    """Dirichlet process prior on cluster sizes."""

    def __init__(self, alpha: float):
        """Initialize DirichletProcess with the given concentration parameter."""
        self.alpha = alpha

    def __call__(self, cluster_sizes: np.ndarray, **kwargs: Any) -> float:
        # Prior probability of a clustering, based on cluster sizes and hyperparameter alpha
        """Return the prior log probability of a clustering."""
        return (
            len(cluster_sizes) * np.log(self.alpha) + np.sum(scipy.special.gammaln(cluster_sizes))
        ).item()

    def marginal(self, n_obs: int, cluster_size: np.ndarray, **kwargs: Any) -> np.ndarray:
        # Prior probability of a single assignment, based on cluster size and hyperparameter alpha
        """Return the marginal prior term for candidate cluster sizes."""
        return np.where(cluster_size > 0, np.log(cluster_size), np.log(self.alpha))
