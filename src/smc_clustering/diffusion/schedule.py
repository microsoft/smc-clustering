# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class LinearSchedule:
    """We use gamma(x) = a + (b-a) * x

    so that gamma(0) = a and gamma(1) = b

    a should be a negative number, b should be positive. The resulting diffusion model is only
    "correct" as a -> -inf and b -> inf, but there are no parameters to tune.
    """

    gamma_0: float = -10.0
    gamma_1: float = 10.0

    def gamma(self, t: jax.Array) -> jax.Array:
        return self.gamma_0 + (self.gamma_1 - self.gamma_0) * t

    def gamma_grad(self, t: jax.Array) -> jax.Array:
        return (self.gamma_1 - self.gamma_0) * jnp.ones_like(t)

    def __call__(self, t: jax.Array) -> jax.Array:
        return nn.sigmoid(self.gamma(t))
