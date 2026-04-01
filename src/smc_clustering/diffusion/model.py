# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Neural network building blocks for diffusion models.

The module implements a lightweight SetFormer architecture with self-attention layers for permutation-invariant set processing.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class SelfAttention(nn.Module):
    """Self-attention block used inside the SetFormer."""

    dim: int

    @nn.compact
    def __call__(self, x: jax.Array, masks: jax.Array, train: bool) -> jax.Array:
        """Apply self-attention to the masked inputs."""
        q = nn.Dense(self.dim)(x)
        k = nn.Dense(self.dim)(x)
        v = nn.Dense(self.dim)(x)

        z = q @ k.mT
        z = nn.Dropout(0.1, deterministic=not train)(z * self.dim ** (-0.5))
        z = jnp.where(masks, z, -1e9)
        z = nn.softmax(z)

        z = z @ v
        return nn.Dense(self.dim)(z)


class Layer(nn.Module):
    """Transformer-style residual block for the SetFormer."""

    dim: int

    @nn.compact
    def __call__(self, x: jax.Array, masks: jax.Array, train: bool) -> jax.Array:
        """Apply one residual SetFormer layer."""
        x = x + SelfAttention(self.dim)(nn.RMSNorm()(x), masks, train)

        mlp = nn.Dense(4 * self.dim)(nn.RMSNorm()(x))
        mlp = nn.gelu(mlp, approximate=False)
        mlp = nn.Dense(self.dim)(mlp)
        mlp = nn.Dropout(0.1, deterministic=not train)(mlp)

        return x + mlp


class SetFormer(nn.Module):
    """Permutation-invariant network for set diffusion models."""

    dim: int
    depth: int

    @nn.compact
    def __call__(self, x: jax.Array, masks: jax.Array, train: bool) -> jax.Array:
        """Apply the SetFormer network to the inputs."""
        seq_len = x.shape[1]
        x = jnp.where(masks[:, :, None], x, 0.0)
        masks = jnp.logical_and(
            jnp.tile(masks[:, :, None], (1, 1, seq_len)), jnp.tile(masks[:, None, :], (1, seq_len, 1))
        )
        for _ in range(self.depth):
            x = Layer(self.dim + 2)(x, masks, train)

        x = nn.RMSNorm()(x)
        return nn.Dense(self.dim)(x)
