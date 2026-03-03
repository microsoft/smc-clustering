# Licensed under the MIT license.
from math import ceil

import jax.numpy as jnp
import numpy as np


def batched_eval(f, batch_size, batched_argnums, *inputs):
    # Split arguments in batched_argnums into batches and pad last batch
    # (avoids recompilations of jitted functions)
    n = inputs[batched_argnums[0]].shape[0]
    n_batches = ceil(n / batch_size)
    pad_by = batch_size * n_batches - n

    batched_input = []
    for i in range(len(inputs)):
        if i in batched_argnums:
            batched = [inputs[i][(batch_size * j) : min(batch_size * (j + 1), n)] for j in range(n_batches)]
            batched[-1] = np.concatenate(
                [batched[-1], np.zeros((pad_by, *inputs[i].shape[1:]), dtype=inputs[i].dtype)], axis=0
            )
            batched_input.append(np.array(batched))
        else:
            batched_input.append(None)

    batched_output = [
        f(*[batched_input[i][b] if i in batched_argnums else inputs[i] for i in range(len(inputs))])
        for b in range(n_batches)
    ]
    if pad_by > 0:
        return jnp.concatenate(batched_output, axis=0)[:-pad_by]
    else:
        return jnp.concatenate(batched_output, axis=0)


def generate_batched_score_func(score_func, batch_shape=(16, 16)):
    def batched_score_func(rng, compute_clusters):
        # split clusters into batches padded to have same cluster size,
        # and evaluate scoring function
        data_dim = compute_clusters[0].shape[1]
        n_batches = ceil(len(compute_clusters) / batch_shape[0])
        compute_clusters = [
            compute_clusters[(batch_shape[0] * i) : min(batch_shape[0] * (i + 1), len(compute_clusters))]
            for i in range(n_batches)
        ]

        scores = []
        for cluster_batch in compute_clusters:
            max_size = batch_shape[1] + max([0, ceil((max([len(c) for c in cluster_batch]) - batch_shape[1]) / 8)]) * 8
            # prepare the data and masks
            data, masks = [], []
            for cluster in cluster_batch:
                size = len(cluster)

                # prepare masks
                mask = jnp.concat([jnp.ones((size,)), jnp.zeros((max_size - size,))])
                masks.append(mask)

                # append nans onto data to make it the same size
                padding = jnp.full((max_size - size, cluster.shape[1]), float("nan"))
                data.append(jnp.concat([cluster, padding], axis=0))

            if len(cluster_batch) < batch_shape[0]:
                data += [jnp.full((max_size, data_dim), float("nan"))] * (batch_shape[0] - len(compute_clusters[-1]))
                masks += [jnp.zeros((max_size,))] * (batch_shape[0] - len(compute_clusters[-1]))

            if len(data) > 0:
                scores.append(score_func(rng, jnp.stack(data), jnp.stack(masks))[: len(cluster_batch)])

        return list(jnp.concatenate(scores))

    return batched_score_func


class DFWrapper:
    # Allows easier retrieval of cluster data from dataframes
    def __init__(self, df):
        self.df = df

    @property
    def shape(self):
        return self.df.shape

    def __getitem__(self, row_ids):
        if type(row_ids) is int:
            return [self.df.iloc[row_ids]["name"]]
        else:
            return list(self.df.iloc[row_ids]["name"])
