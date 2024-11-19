# Licensed under the MIT license.
import jax
import jax.numpy as jnp
import flax.linen as nn

class SelfAttention(nn.Module):
    dim: int
    
    @nn.compact
    def __call__(self, x, masks, train):
        q = nn.Dense(self.dim)(x)
        k = nn.Dense(self.dim)(x)
        v = nn.Dense(self.dim)(x)
        
        z = q @ k.mT
        z = nn.Dropout(0.1, deterministic = not train)(z * self.dim**(-0.5))
        z = jnp.where(masks, z, -1e9)
        z = nn.softmax(z)
        
        z = z @ v
        return nn.Dense(self.dim)(z)


class Layer(nn.Module):
    dim: int
    
    @nn.compact
    def __call__(self, x, masks, train):
        x = x + SelfAttention(self.dim)(nn.RMSNorm()(x), masks, train)
        
        mlp = nn.Dense(4*self.dim)(nn.RMSNorm()(x))
        mlp = nn.gelu(mlp, approximate=False)
        mlp = nn.Dense(self.dim)(mlp)
        mlp = nn.Dropout(0.1, deterministic = not train)(mlp)
        
        return x + mlp


class SetFormer(nn.Module):
    dim: int
    depth: int
    
    @nn.compact
    def __call__(self, x, masks, train):
        seq_len = x.shape[1]
        x = jnp.where(masks[:,:,None], x, 0.)
        masks = jnp.logical_and(jnp.tile(masks[:, :, None],(1, 1, seq_len)), jnp.tile(masks[:, None, :],(1, seq_len, 1)))
        for _ in range(self.depth):
            x = Layer(self.dim + 2)(x, masks, train)
        
        x = nn.RMSNorm()(x)
        return nn.Dense(self.dim)(x)
        
