"""
A tiny GPT-style Transformer decoder for next-token prediction.

The model uses token + positional embeddings, pre-LN blocks with causal self-attention (via scaled dot-product
attention), and a 2-layer GELU MLP. It returns logits of shape [B, T, V] for input IDs [B, T]. Weight tying with
the token embedding is supported for parameter efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the tiny Transformer decoder."""

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


class FeedForward(nn.Module):
    """Two-layer MLP with GELU and dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, use_bias: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN: x -> GELU(Linear) -> Dropout -> Linear."""
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using scaled dot-product attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, use_bias: bool) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.out = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def _shape_qkv(self, qkv: torch.Tensor, B: int, T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, use_bias: bool, eps: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, use_bias)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.mlp = FeedForward(d_model, d_ff, dropout, use_bias)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one Transformer block to x [B, T, D]."""
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        return x


class TinyTransformerLM(nn.Module):
    """A tiny GPT-style decoder-only LM that returns logits over the vocabulary."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.use_bias, cfg.layer_norm_eps)
                for _ in range(cfg.n_layers)
            ],
        )
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits [B, T, V] from input token IDs [B, T]."""
        assert input_ids.dim() == 2 and input_ids.dtype == torch.long, "input_ids must be [B, T] long"
        B, T = input_ids.shape
        assert self.cfg.max_seq_len >= T, f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        device = input_ids.device
        pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)  # [1, T]
        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # [B, T, D]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits
