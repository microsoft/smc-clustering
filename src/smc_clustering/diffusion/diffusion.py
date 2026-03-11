# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Variational diffusion models for set-valued data.

This module defines the diffusion objective, sampling procedures, and log-probability estimators used in the research code path.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp

from smc_clustering.diffusion.model import SetFormer
from smc_clustering.diffusion.schedule import LinearSchedule


class VariationalDiffusion:
    """Variational diffusion model for sets.

    The model follows https://arxiv.org/pdf/2107.00630.pdf and uses a SetFormer to predict the noise added to a set-valued input. A monotone schedule controls the variance injected at each diffusion time step.
    """

    def __init__(self, rng: jax.Array, dim: int, depth: int, schedule: LinearSchedule | None = None):
        """Initialize VariationalDiffusion with the given network configuration."""
        self.dim = dim
        if schedule is None:
            schedule = LinearSchedule()
        self.schedule = schedule
        self.net = SetFormer(dim, depth)
        self.params: Any = self.net.init(
            rng, jnp.zeros((1, 1, self.dim + 2)), jnp.ones((1, 1), dtype=jnp.bool), train=False
        )

        self.trained_net: Callable[[jax.Array, jax.Array], jax.Array] | None = None

    def compile_net(self):
        """JIT-compile the network for inference."""
        self.trained_net = jax.jit(
            lambda x, masks: cast(jax.Array, self.net.apply(self.params, x, masks, train=False))
        )

    def loss(self, rng: jax.Array, params: Any, x: jax.Array, masks: jax.Array) -> jax.Array:  # noqa: ANN401
        """Compute the diffusion training loss for a batch."""
        batch_size, seq_len, _ = x.shape  # x: [B, L, D]

        # sample time points across batch
        # ti = mod(u0 + i/batch_size, 1)
        rng, t_rng = jax.random.split(rng)
        u0 = jax.random.uniform(t_rng, (1,), dtype=x.dtype)
        t = jnp.arange(batch_size, dtype=x.dtype) / batch_size + u0
        t %= 1
        t = t[:, None, None]  # [B, 1, 1], broadcast over set elements and features

        # set up noise schedule
        sigma2 = self.schedule(t)
        sigma = jnp.sqrt(sigma2)
        alpha = jnp.sqrt(1 - sigma2)

        # sample noise and add to data
        rng, eps_rng = jax.random.split(rng)
        eps = jax.random.normal(eps_rng, x.shape)
        z = alpha * x + sigma * eps

        # pass z, set_size and t through the model, concatenated
        set_size = masks.sum(axis=1)[:, None, None]
        model_input = jnp.concat(
            [z, jnp.tile(set_size, (1, seq_len, 1)), jnp.tile(t, (1, seq_len, 1))], axis=-1
        )  # [B, L, D + 2]
        rng, dropout_rng = jax.random.split(rng)
        eps_hat = cast(
            jax.Array,
            self.net.apply(params, model_input, masks, train=True, rngs={"dropout": dropout_rng}),
        )

        # compute the L_inf loss
        gamma_grad = self.schedule.gamma_grad(t)
        loss = 0.5 * jnp.sum(gamma_grad * masks[:, :, None] * (eps_hat - eps) ** 2, axis=[1, 2])  # [B]
        loss = loss.mean()  # average over batch

        return loss

    def smc_step(
        self,
        rng: jax.Array,
        s: jax.Array,
        t: jax.Array,
        z_s: jax.Array,
        log_weights: jax.Array,
        log_prob: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # compute mean and variance of q(z_t | z_s)
        """Advance one SMC step through diffusion time."""
        mu_q_ts, sigma2_q_ts = self.moments_q_ts(z_s, t, s)

        # sample z_t | z_s
        eps = jax.random.normal(rng, z_s.shape)
        z_t = mu_q_ts + jnp.sqrt(sigma2_q_ts) * eps

        # compute mean and variance of p(z_s | z_t) (includes network call)
        mu_p_st, sigma2_p_st = self.moments_p_st(z_t, s, t)

        # compute log( p(z_s | z_t) / q(z_t | z_s) )
        log_p = -0.5 * jnp.sum(jnp.log(sigma2_p_st) + (z_s - mu_p_st) ** 2 / sigma2_p_st, axis=[1, 2])
        log_q = -0.5 * jnp.sum(jnp.log(sigma2_q_ts) + (z_t - mu_q_ts) ** 2 / sigma2_q_ts, axis=[1, 2])
        log_ratios = log_p - log_q

        # update weights, normalizer
        log_weights += log_ratios
        log_normalizer = jax.scipy.special.logsumexp(log_weights, axis=0)
        log_weights -= log_normalizer
        log_prob += log_normalizer

        return z_t, log_weights, log_prob

    def logp_smc(
        self,
        rng: jax.Array,
        x: jax.Array,
        num_particles: int,
        num_time_steps: int,
        resample_thresh: float = 0.25,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Run an SMC estimate of the data log probability.

        The routine propagates particles forward in time and reweights them by the ratio between reverse and forward transitions.
        """
        z_s = jnp.tile(x, (num_particles, 1, 1))

        log_prob = 0.0
        log_weights = jnp.ones((num_particles,)) * jnp.log(1.0 / num_particles)
        for i in jnp.arange(0, num_time_steps):
            # t is the "next" time step, s is the "current" time step
            s = i / num_time_steps
            t = (i + 1) / num_time_steps

            rng, step_rng = jax.random.split(rng)
            z, log_weights, log_prob = self.smc_step(step_rng, s, t, z_s, log_weights, log_prob)

            # resample particles if degenerate
            n_eff = 1.0 / jnp.sum(jnp.square(jnp.exp(log_weights)))
            if n_eff < resample_thresh * num_particles:
                rng, resample_rng = jax.random.split(rng)
                z = jax.random.choice(
                    resample_rng, z, shape=(num_particles,), p=jnp.exp(log_weights), replace=True, axis=0
                )

                log_weights = jnp.ones((num_particles,)) * jnp.log(1.0 / num_particles)

        return log_prob, z, log_weights

    def moments_q_ts(self, z_s: jax.Array, t: jax.Array, s: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute the mean and variance of q(z_t | z_s).

        See eqs 20-22 of the paper (https://arxiv.org/pdf/2107.00630.pdf).

        note s < t
        """
        sigma2_t = self.schedule(t)
        alpha2_t = 1 - sigma2_t
        sigma2_s = self.schedule(s)
        alpha2_s = 1 - sigma2_s
        alpha2_ts = alpha2_t / alpha2_s
        mu = z_s * jnp.sqrt(alpha2_ts)
        sigma2_ts = sigma2_t - alpha2_ts * sigma2_s

        return mu, sigma2_ts

    def moments_p_st(
        self, z: jax.Array, s: jax.Array, t: jax.Array, masks: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the moments of p(z_s | z_t).

        This follows equation 34 of https://arxiv.org/pdf/2107.00630.pdf with s < t.
        """
        if masks is None:
            masks = jnp.ones_like(z[:, :, 0], dtype=jnp.bool)

        batch_size, seq_len, _ = z.shape
        set_size = masks.sum(axis=1)[:, None, None]  # [B, 1, 1]
        model_input = jnp.concat(
            [z, jnp.tile(set_size, (1, seq_len, 1)), t * jnp.ones((batch_size, seq_len, 1))], axis=-1
        )  # [B, L, D + 2]
        eps_hat = self.trained_net(model_input, masks)

        gamma_s = self.schedule.gamma(s)
        gamma_t = self.schedule.gamma(t)
        sigma2_t = self.schedule(t)
        sigma_t = jnp.sqrt(sigma2_t)
        alpha2_t = 1 - sigma2_t
        sigma2_s = self.schedule(s)
        alpha2_s = 1 - sigma2_s
        c = -jnp.expm1(gamma_s - gamma_t)
        mu = jnp.sqrt(alpha2_s / alpha2_t) * (z - sigma_t * c * eps_hat)
        sigma2 = sigma2_s * c
        return mu, sigma2

    def generate(
        self,
        rng: jax.Array,
        num_samples: int,
        seq_len: int,
        num_time_steps: int,
        masks: jax.Array | None = None,
    ) -> jax.Array:
        """Generate samples from the diffusion model."""
        if masks is None:
            masks = jnp.ones((num_samples, seq_len), dtype=jnp.bool)

        rng, init_rng = jax.random.split(rng)
        z = jax.random.normal(init_rng, (num_samples, seq_len, self.dim))

        def update_step(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], rng: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
            z, s, t, masks = carry
            t -= 1 / num_time_steps
            s -= 1 / num_time_steps

            mu, sigma2 = self.moments_p_st(z, s, t, masks)
            z = mu + jnp.sqrt(sigma2) * jax.random.normal(rng, z.shape)

            return (z, s, t, masks), None

        step_rng = jax.random.split(rng, num_time_steps)
        t = 1.0
        s = 1 - 1 / num_time_steps
        (z, s, t, masks), _ = jax.lax.scan(update_step, (z, s, t, masks), step_rng)

        return z

    def score_fn(self, z: jax.Array, masks: jax.Array, t: jax.Array) -> jax.Array:
        """Compute the score function implied by the trained network.

        See equations 29-31 of https://arxiv.org/pdf/2107.00630.pdf.
        """
        batch_size, seq_len, _ = z.shape
        # concatenate z, set_size and t
        set_size = masks.sum(axis=1)[:, None, None]
        model_input = jnp.concat(
            [z, jnp.tile(set_size, (1, seq_len, 1)), t * jnp.ones((batch_size, seq_len, 1))], axis=-1
        )  # [B, L, D + 2]
        eta_hat = self.trained_net(model_input, masks)
        sigma2 = self.schedule(t)
        sigma = jnp.sqrt(sigma2)
        return -eta_hat / sigma

    def prob_flow_grad_fn(self, z: jax.Array, masks: jax.Array, t: jax.Array) -> jax.Array:
        # taken from https://arxiv.org/pdf/2210.05475.pdf
        # Eq 18 and surrounding notes
        """Compute the probability-flow ODE drift."""
        s = self.score_fn(z, masks, t)
        beta = self.schedule(t) * self.schedule.gamma_grad(t)
        return -0.5 * beta * (z + s)

    def generate_ode(
        self,
        num_samples: int,
        seq_len: int,
        num_time_steps: int,
        masks: jax.Array | None = None,
        z: jax.Array | None = None,
        rng: jax.Array | None = None,
    ) -> jax.Array:
        """Generate samples with the probability-flow ODE.

        The ode is:

            dz = -beta * (z + score(z)) dt

        Note that we run the ode _backwards_ in time, from t=1 to t=0.
        This just uses a simple Euler expansion for now.
        """
        if masks is None:
            masks = jnp.ones((num_samples, seq_len), dtype=jnp.bool)
        if z is None:
            z = jax.random.normal(rng, (num_samples, seq_len, self.dim))

        def update_step(
            carry: tuple[jax.Array, jax.Array], t: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            z, masks = carry
            z -= 1 / num_time_steps * self.prob_flow_grad_fn(z, masks, t)

            return (z, masks), None

        ts = jnp.linspace(1, 1 / num_time_steps, num_time_steps)
        (z, masks), _ = jax.lax.scan(update_step, (z, masks), ts)
        return z

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def log_prob_ode(
        self, rng: jax.Array, x: jax.Array, masks: jax.Array, num_time_steps: int = 100
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the log probability of x under the model, using the probability flow ODE."""

        def update_step(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], rng: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
            # use vector-Jacobian product and Skilling-Hutchinson estimator
            # see https://openreview.net/pdf?id=PxTIG12RRHS eq 39 - 40

            x, masks, t, log_prob = carry
            eps = jax.random.normal(rng, x.shape)

            f = functools.partial(self.prob_flow_grad_fn, masks=masks, t=t)
            grad, f_vjp = jax.vjp(f, x)
            vjp = f_vjp(eps)[0]

            x += 1 / num_time_steps * grad
            log_prob += jnp.sum(eps * vjp * masks[:, :, None], axis=[1, 2]) * 1 / num_time_steps

            t += 1 / num_time_steps

            return (x, masks, t, log_prob), None

        t = 0.0
        log_prob = jnp.zeros((x.shape[0],))

        step_rng = jax.random.split(rng, num_time_steps)
        (x, masks, t, log_prob), _ = jax.lax.scan(update_step, (x, masks, t, log_prob), step_rng)

        log_p_xT = -0.5 * (x**2 + 2 * jnp.log(2 * 3.141592653589793))
        log_prob += jnp.sum(jnp.where(masks, log_p_xT.sum(axis=-1), 0), axis=-1)
        return log_prob, x
