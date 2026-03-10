"""Shared typing protocols for clustering components."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class PriorLike(Protocol):
    """Protocol for cluster-size priors used by the clustering algorithms."""

    def __call__(self, cluster_sizes: np.ndarray) -> float | np.ndarray:
        """Return the prior score for a batch of cluster sizes."""

    def marginal(self, _n_obs: int, cluster_size: int | np.ndarray) -> np.ndarray:
        """Return marginal prior terms for candidate assignments."""


class SurrogateLike(Protocol):
    """Protocol for surrogate likelihood models."""

    def post_predictive(self, x: object, n: np.ndarray, summary: object) -> np.ndarray:
        """Return surrogate scores for candidate cluster assignments."""
