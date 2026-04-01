# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""A Transformer model for next-token prediction.

The model uses token + positional embeddings, pre-LN blocks with causal self-attention, and a 2-layer GELU MLP.
It returns logits of shape [B, T, V] for input IDs [B, T].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the Transformer decoder."""

    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    use_bias: bool = False  # Linear layer bias
    pos_encoding: str = "rope"  # "rope" | "learned"
    rope_theta: float = 10000.0
    norm_type: str = "rms"  # "rms" | "layernorm"
    ffn_activation: str = "swiglu"  # "swiglu" | "gelu"


def _rope_apply(
    q: torch.Tensor, k: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to query and key tensors.

    Args:
        q: Query tensor [B, nH, T, H] where H must be even.
        k: Key tensor [B, nH, T, H] where H must be even.
        rope_cache: Precomputed RoPE cache [T, H/2, 2].

    Returns:
        Rotated (q, k) tensors with same shapes as input.
    """
    cos = rope_cache[..., 0]  # [T, H/2]
    sin = rope_cache[..., 1]  # [T, H/2]

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        # Expand cos/sin to match x dimensions [B, nH, T, H]
        # cos/sin are [T, H/2], need to become [B, nH, T, H/2] then expand to [B, nH, T, H]
        cos_expanded = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H/2]
        sin_expanded = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H/2]
        # Repeat to match the full head dimension pattern
        cos_full = torch.repeat_interleave(cos_expanded, 2, dim=-1)  # [1, 1, T, H]
        sin_full = torch.repeat_interleave(sin_expanded, 2, dim=-1)  # [1, 1, T, H]
        return x * cos_full + _rotate(x) * sin_full

    return _apply(q), _apply(k)


def _build_rope_cache(
    T: int, H: int, theta: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build RoPE positional encoding cache.

    Args:
        T: Maximum sequence length.
        H: Head dimension (must be even).
        theta: RoPE theta parameter.
        device: Target device for the cache.
        dtype: Target dtype for the cache.

    Returns:
        RoPE cache tensor [T, H/2, 2] where cache[..., 0] is cos and cache[..., 1] is sin.
    """
    assert H % 2 == 0, "Head dimension must be even for RoPE"
    inv_freq = 1.0 / (theta ** (torch.arange(0, H, 2, device=device, dtype=dtype) / H))
    t = torch.arange(T, device=device, dtype=dtype)
    freqs = torch.einsum("t,f->tf", t, inv_freq)  # [T, H/2]
    cos = torch.cos(freqs)  # [T, H/2]
    sin = torch.sin(freqs)  # [T, H/2]
    # Stack cos and sin: [T, H/2, 2]
    return torch.stack((cos, sin), dim=-1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        """Initialize RMSNorm with learnable scale parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to input tensor x [..., D]."""
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class FeedForward(nn.Module):
    """Two-layer MLP with configurable activation (GELU or SwiGLU) and dropout."""

    def __init__(
        self, d_model: int, d_ff: int, dropout: float, use_bias: bool, activation: str = "swiglu"
    ) -> None:
        """Initialize FFN with configurable activation."""
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            self.w1 = nn.Linear(d_model, d_ff, bias=use_bias)
            self.w3 = nn.Linear(d_model, d_ff, bias=use_bias)  # gate
            self.w2 = nn.Linear(d_ff, d_model, bias=use_bias)
        else:
            self.fc1 = nn.Linear(d_model, d_ff, bias=use_bias)
            self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN with specified activation function."""
        if self.activation == "swiglu":
            a = F.silu(self.w1(x)) * self.w3(x)
            return self.w2(self.dropout(a))
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using scaled dot-product attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        use_bias: bool,
        rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 512,
    ) -> None:
        """Initialize multi-head causal self-attention."""
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.out = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = rope
        if rope:
            self.register_buffer(
                "rope_cache",
                _build_rope_cache(
                    max_seq_len,
                    self.head_dim,
                    rope_theta,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.rope_theta = rope_theta

    def _shape_qkv(
        self, qkv: torch.Tensor, B: int, T: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape fused QKV [B,T,3D] into (Q,K,V) each [B, nH, T, H]."""
        q, k, v = qkv.chunk(3, dim=-1)
        # [B, T, nH, H] -> [B, nH, T, H]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal self-attention over x [B, T, D]."""
        B, T, D = x.shape
        assert self.d_model == D
        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = self._shape_qkv(qkv, B, T)  # [B, nH, T, H] each

        if self.use_rope:
            # Ensure cache on correct device/dtype and length
            if (
                self.rope_cache.device != x.device
                or self.rope_cache.dtype != x.dtype
                or self.rope_cache.size(0) < T
            ):
                self.rope_cache = _build_rope_cache(T, self.head_dim, self.rope_theta, x.device, x.dtype)
            q, k = _rope_apply(q, k, self.rope_cache[:T, :])

        # PyTorch's scaled_dot_product_attention with causal masking.
        # Returns [B, nH, T, H].
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads: [B, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out(y)
        return y


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN -> Self-Attn -> residual -> LN -> MLP -> residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_bias: bool,
        eps: float,
        rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 512,
        norm_type: str = "rms",
        ffn_activation: str = "swiglu",
    ) -> None:
        """Initialize Transformer block with configurable normalization and activation."""
        super().__init__()
        if norm_type == "rms":
            self.ln1 = RMSNorm(d_model, eps=eps)
            self.ln2 = RMSNorm(d_model, eps=eps)
        else:
            self.ln1 = nn.LayerNorm(d_model, eps=eps)
            self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.attn = CausalSelfAttention(
            d_model, n_heads, dropout, use_bias, rope, rope_theta, max_seq_len
        )
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = FeedForward(d_model, d_ff, dropout, use_bias, activation=ffn_activation)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one Transformer block to x [B, T, D]."""
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        return x


class TransformerLM(nn.Module):
    """A GPT-style decoder-only LM that returns logits over the vocabulary."""

    def __init__(self, cfg: TransformerConfig) -> None:
        """Initialize the Transformer LM from config."""
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = None if cfg.pos_encoding == "rope" else nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    cfg.dropout,
                    cfg.use_bias,
                    cfg.layer_norm_eps,
                    rope=(cfg.pos_encoding == "rope"),
                    rope_theta=cfg.rope_theta,
                    max_seq_len=cfg.max_seq_len,
                    norm_type=cfg.norm_type,
                    ffn_activation=cfg.ffn_activation,
                )
                for _ in range(cfg.n_layers)
            ],
        )
        if cfg.norm_type == "rms":
            self.ln_f = RMSNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        else:
            self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=cfg.use_bias)

        # Optional tying of token embedding and head.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters with small std for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits [B, T, V] from input token IDs [B, T]."""
        assert input_ids.dim() == 2 and input_ids.dtype == torch.long, "input_ids must be [B, T] long"
        _B, T = input_ids.shape

        device = input_ids.device
        x = self.tok_emb(input_ids)  # [B, T, D]
        if self.pos_emb is not None:  # learned positional embeddings
            assert self.cfg.max_seq_len >= T, (
                f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
            pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)  # [1, T]
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits
