# Licensed under the MIT license.

import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.out = torch.nn.Linear(dim, dim)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.softmax_scale = self.dim ** (-0.5)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, masks):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        z = q @ k.mT
        z = self.dropout(z * self.softmax_scale)
        z = torch.where(masks, z, -1e9)
        z = self.softmax(z)

        z = z @ v
        return self.out(z)


class RMSNorm(torch.nn.Module):
    eps = 1e-6

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)


class Layer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = RMSNorm()
        self.attn = SelfAttention(dim)
        self.norm2 = RMSNorm()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.GELU(), torch.nn.Linear(dim * 4, dim), torch.nn.Dropout(0.1)
        )

    def forward(self, x, masks=None):
        x = x + self.attn(self.norm1(x), masks)
        x = x + self.mlp(self.norm2(x))
        return x


class SetFormer(torch.nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

        # since the NN needs z, set_size and t as inputs, we add two to the dimension
        for _ in range(depth):
            self.layers.append(Layer(dim + 2))
        self.norm = RMSNorm()
        self.out = torch.nn.Linear(dim + 2, dim)

    def forward(self, x, masks=None):
        if masks is not None:
            # put zeros in the masked bits of x
            x = torch.where(masks[:, :, None], x, 0.0)
            # turn the maks from a linear to square shape
            seq_len = x.shape[1]
            masks = torch.logical_and(masks[:, :, None].repeat(1, 1, seq_len), masks[:, None, :].repeat(1, seq_len, 1))
        else:
            batch_size, seq_len, _ = x.shape
            masks = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool, device=x.device)

        for layer in self.layers:
            x = layer(x, masks)
        x = self.norm(x)
        return self.out(x)
