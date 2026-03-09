# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared setup helpers for circles experiments.

The module loads the diffusion scorer, configures priors and surrogates, and synthesizes the circles benchmark dataset.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from smc_clustering import clustering
from smc_clustering.clustering.cluster import DirichletProcess
from smc_clustering.clustering.surrogate_models import Gaussian, GaussianCluster
from smc_clustering.diffusion.diffusion import VariationalDiffusion
from smc_clustering.diffusion.schedule import LinearSchedule


theta = np.array([1.4e3, 1.7e3, 2.7e2, 3e2, -9.6e-2, -3.2e-2, 2.1e-2, -2e-2])
a = theta[0:2]
b = theta[2:4]
mu = theta[4:6]
lam = jnp.abs(theta[6:8])
alpha = 1
surrogate = Gaussian(a, b, mu, lam)
prior = DirichletProcess(alpha)
ClusterClass = GaussianCluster


def load_model(checkpoint_path: str = "checkpoints") -> Callable:
    """Load the diffusion scorer from a checkpoint directory."""
    rng = jax.random.PRNGKey(1)
    schedule = LinearSchedule()
    model = model = VariationalDiffusion(rng, dim=2, depth=6, schedule=schedule)

    raw_restored = checkpoints.restore_checkpoint(
        ckpt_dir=os.path.join(os.getcwd(), checkpoint_path), target=None
    )
    model.params = {"params": raw_restored["model"]["params"]}
    model.compile_net()

    num_time_steps = 100

    def score_func(rng: jax.Array, data: jax.Array, masks: jax.Array) -> jax.Array:
        scores, _ = model.log_prob_ode(rng, data, masks, num_time_steps=num_time_steps)
        return scores

    batched_score_eval = clustering.utils.generate_batched_score_func(score_func, batch_shape=(8, 8))

    return batched_score_eval


def generate_circles(
    rng: jax.Array,
    num_circles: int,
    min_radius: float = 0.1,
    max_radius: float = 1.5,
    min_points: int = 3,
    max_points: int = 20,
    min_x: float = -5.0,
    max_x: float = 5.0,
    min_y: float = -5.0,
    max_y: float = 5.0,
    num_points: int = None,
) -> tuple[list, list]:
    """Generate synthetic circles and their masks."""
    r_rng, x_rng, y_rng, n_rng, theta_rng = jax.random.split(rng, 5)

    radii = jax.random.uniform(r_rng, (num_circles,)) * (max_radius - min_radius) + min_radius
    x = jax.random.uniform(x_rng, (num_circles,)) * (max_x - min_x) + min_x
    y = jax.random.uniform(y_rng, (num_circles,)) * (max_y - min_y) + min_y
    if num_points is None:
        num_points = jax.random.randint(n_rng, (num_circles,), min_points, max_points)

    theta = jax.random.uniform(theta_rng, (jnp.sum(num_points),)) * 2 * 3.14159
    i = jnp.arange(0, jnp.max(num_points))
    masks = jax.vmap(lambda n: jnp.where(i < n, True, False))(num_points)
    masks = jnp.array(masks, dtype=jnp.bool)

    angles = np.zeros((jnp.max(num_points) * num_circles,))
    angles[jnp.concat(masks, axis=-1)] = theta

    def gen_circle(
        x: jax.Array, y: jax.Array, radius: jax.Array, angles: jax.Array, mask: jax.Array
    ) -> jax.Array:
        x = mask * (radius * jnp.cos(angles) + x)
        y = mask * (radius * jnp.sin(angles) + y)
        circle = jnp.stack([x, y], axis=-1)
        return circle

    circles = jax.vmap(gen_circle, in_axes=(0, 0, 0, 0, 0))(
        x, y, radii, jnp.array(jnp.split(angles, num_circles)), masks
    )

    return list(np.array(circles)), list(np.array(masks))


def generate_circles_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate the flattened circles benchmark dataset."""
    rng = jax.random.PRNGKey(23)
    circles_valid, masks_valid = generate_circles(
        rng, 15, min_points=10, max_points=30, min_radius=0.6, max_radius=0.6
    )

    cluster_data = []
    labels = []
    for k, (c, m) in enumerate(zip(circles_valid, masks_valid)):
        for ci, mi in zip(c, m):
            if mi:
                cluster_data.append(ci[None, :])
                labels.append(k)

    labels = np.array(labels)
    data = np.array(cluster_data).squeeze(1)

    return data, labels
