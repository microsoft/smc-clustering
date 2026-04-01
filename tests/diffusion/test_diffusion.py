# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Regression tests for the diffusion research code path."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from smc_clustering.diffusion.diffusion import VariationalDiffusion
from smc_clustering.diffusion.schedule import LinearSchedule


class _BaseTestDiffusion(VariationalDiffusion):
    """Minimal diffusion stub for deterministic regression tests."""

    def __init__(self):
        self.dim = 1
        self.schedule = LinearSchedule()
        self.trained_net = lambda x, masks: jnp.zeros_like(x[..., : self.dim])


class _PropagatingDiffusion(_BaseTestDiffusion):
    """Diffusion stub whose SMC step increments the latent state."""

    def smc_step(
        self,
        rng: jax.Array,
        s: jax.Array,
        t: jax.Array,
        z_s: jax.Array,
        log_weights: jax.Array,
        log_prob: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del rng, s, t
        return z_s + 1.0, log_weights, log_prob


class _ScheduledGenerationDiffusion(_BaseTestDiffusion):
    """Diffusion stub that encodes the reverse-time schedule into the sample path."""

    def moments_p_st(
        self, z: jax.Array, s: jax.Array, t: jax.Array, masks: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        del masks
        return z + (10.0 * s + t), jnp.zeros_like(z)


def test_logp_smc_propagates_particles_across_time():
    """SMC should feed the updated particle state into the next time step."""
    model = _PropagatingDiffusion()
    x = jnp.zeros((1, 1, 1))

    _, z, log_weights = model.logp_smc(
        rng=jax.random.PRNGKey(0),
        x=x,
        num_particles=3,
        num_time_steps=4,
        resample_thresh=0.0,
    )

    assert jnp.array_equal(z, jnp.full((3, 1, 1), 4.0))
    assert jnp.array_equal(log_weights, jnp.full((3,), jnp.log(1.0 / 3.0)))


def test_generate_uses_reverse_time_grid_without_skipping_first_step():
    """Reverse diffusion should start at t=1 and end at s=0 without overshooting."""
    model = _ScheduledGenerationDiffusion()
    rng = jax.random.PRNGKey(7)
    num_samples = 2
    seq_len = 3
    num_time_steps = 4

    _, init_rng = jax.random.split(rng)
    initial_z = jax.random.normal(init_rng, (num_samples, seq_len, model.dim))

    samples = model.generate(
        rng=rng,
        num_samples=num_samples,
        seq_len=seq_len,
        num_time_steps=num_time_steps,
    )

    expected_increment = 17.5
    assert jnp.allclose(samples, initial_z + expected_increment)
