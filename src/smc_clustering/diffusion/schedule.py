# Copyright (c) Lancaster University.
# Licensed under the MIT license.


"""Noise schedules for the diffusion models.

These helpers parameterize the variance schedule and its derivatives across diffusion time.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class LinearSchedule:
    """Linear schedule in logit space for diffusion noise.

    The schedule uses gamma(t) = a + (b - a) * t so that gamma(0) = a and gamma(1) = b, yielding a simple monotone variance schedule without tunable curvature.
    """

    gamma_0: float = -10.0
    gamma_1: float = 10.0

    def gamma(self, t: jax.Array) -> jax.Array:
        """Return the schedule logits at time t."""
        return self.gamma_0 + (self.gamma_1 - self.gamma_0) * t

    def gamma_grad(self, t: jax.Array) -> jax.Array:
        """Return the derivative of the schedule logits."""
        return (self.gamma_1 - self.gamma_0) * jnp.ones_like(t)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Return the diffusion variance at time t."""
        return nn.sigmoid(self.gamma(t))
