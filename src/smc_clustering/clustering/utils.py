# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utility helpers for batched clustering computations.

The functions and wrappers in this module adapt batched scorers and tabular data to the clustering interfaces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np


class _SupportsILoc(Protocol):
    """Minimal dataframe-like interface used by DFWrapper."""

    shape: tuple[int, ...]
    iloc: Any


def batched_eval(
    f: Callable, batch_size: int, batched_argnums: Sequence[int], *inputs: Any
) -> jax.Array:
    """Evaluate a function on padded mini-batches.

    Padding the final batch keeps every JIT invocation on the same shape, which avoids recompiling on short tails.
    """
    n = inputs[batched_argnums[0]].shape[0]
    n_batches = ceil(n / batch_size)
    pad_by = batch_size * n_batches - n

    batched_input: list[Any] = []
    for i in range(len(inputs)):
        if i in batched_argnums:
            batched = [
                inputs[i][(batch_size * j) : min(batch_size * (j + 1), n)] for j in range(n_batches)
            ]
            batched[-1] = np.concatenate(
                [batched[-1], np.zeros((pad_by, *inputs[i].shape[1:]), dtype=inputs[i].dtype)], axis=0
            )
            batched_input.append(np.array(batched))
        else:
            batched_input.append(None)

    batched_output = [
        f(*[batched_input[i][b] if i in batched_argnums else inputs[i] for i in range(len(inputs))])
        for b in range(n_batches)
    ]  # Each batch result has leading shape [batch_size, ...].
    if pad_by > 0:
        return jnp.concatenate(batched_output, axis=0)[:-pad_by]  # Drop padded rows -> [n, ...].
    return jnp.concatenate(batched_output, axis=0)


def generate_batched_score_func(
    score_func: Callable, batch_shape: tuple[int, int] = (16, 16)
) -> Callable:
    """Wrap a score function with padded batching logic."""

    def batched_score_func(rng: jax.Array, compute_clusters: list[np.ndarray]) -> list[jax.Array]:
        """Pad cluster batches so every scorer call sees a fixed batch/sequence shape."""
        data_dim = compute_clusters[0].shape[1]
        n_batches = ceil(len(compute_clusters) / batch_shape[0])
        cluster_batches = [
            compute_clusters[(batch_shape[0] * i) : min(batch_shape[0] * (i + 1), len(compute_clusters))]
            for i in range(n_batches)
        ]

        scores = []
        for cluster_batch in cluster_batches:
            max_size = (
                batch_shape[1]
                + max([0, ceil((max([len(c) for c in cluster_batch]) - batch_shape[1]) / 8)]) * 8
            )
            data, masks = [], []
            for cluster in cluster_batch:
                size = len(cluster)

                mask = jnp.concat([jnp.ones((size,)), jnp.zeros((max_size - size,))])  # [max_size]
                masks.append(mask)

                padding = jnp.full((max_size - size, cluster.shape[1]), float("nan"))
                data.append(jnp.concat([cluster, padding], axis=0))  # [max_size, data_dim]

            if len(cluster_batch) < batch_shape[0]:
                # Pad the batch itself so the scorer always receives [batch_shape[0], max_size, data_dim].
                data += [jnp.full((max_size, data_dim), float("nan"))] * (
                    batch_shape[0] - len(cluster_batches[-1])
                )
                masks += [jnp.zeros((max_size,))] * (batch_shape[0] - len(cluster_batches[-1]))

            if len(data) > 0:
                scores.append(score_func(rng, jnp.stack(data), jnp.stack(masks))[: len(cluster_batch)])

        return list(jnp.concatenate(scores))

    return batched_score_func


class DFWrapper:
    """Wrapper that exposes dataframe rows through the clustering API."""

    def __init__(self, df: _SupportsILoc):
        """Initialize DFWrapper with the provided dataframe."""
        self.df = df

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the wrapped data."""
        return self.df.shape

    def __getitem__(self, row_ids: int | np.ndarray) -> list[Any]:
        """Return items selected from the wrapped dataframe."""
        if type(row_ids) is int:
            return [self.df.iloc[row_ids]["name"]]
        return list(self.df.iloc[row_ids]["name"])
