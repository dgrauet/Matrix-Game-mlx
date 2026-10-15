"""Shared Wan building blocks: embeddings, RoPE, norms and T2V cross-attention."""
from typing import List, Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .attention import attention

__all__ = [
    'sinusoidal_embedding_1d',
    'rope_params',
    'rope_apply',
    'WanRMSNorm',
    'WanLayerNorm',
    'WanSelfAttention',
    'WanT2VCrossAttention',
    'WAN_CROSSATTENTION_CLASSES',
]


def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    """Upstream computes in float64; float64 is CPU-only in MLX, so build in numpy float64."""
    assert dim % 2 == 0
    half = dim // 2
    pos = np.array(position.astype(mx.float32), dtype=np.float64)
    sinusoid = np.outer(pos, np.power(10000.0, -np.arange(half, dtype=np.float64) / half))
    return mx.array(np.concatenate([np.cos(sinusoid), np.sin(sinusoid)], axis=1).astype(np.float32))


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> Tuple[mx.array, mx.array]:
    """cos/sin tables built in float64 (as torch.polar on float64), stored float32."""
    assert dim % 2 == 0
    freqs = np.outer(np.arange(max_seq_len, dtype=np.float64),
                     1.0 / np.power(theta, np.arange(0, dim, 2, dtype=np.float64) / dim))
    return mx.array(np.cos(freqs).astype(np.float32)), mx.array(np.sin(freqs).astype(np.float32))


def _apply_rope_rotation(
    x_pairs: mx.array, cos_freqs: mx.array, sin_freqs: mx.array
) -> mx.array:
    """Apply rotary embedding using real-valued cos/sin arithmetic.

    Args:
        x_pairs: Input reshaped to ``[..., dim//2, 2]``.
        cos_freqs: Cosine frequencies broadcastable to ``[..., dim//2]``.
        sin_freqs: Sine frequencies broadcastable to ``[..., dim//2]``.

    Returns:
        Rotated tensor with same leading dims, last dim ``dim``.
    """
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    out_real = x_real * cos_freqs - x_imag * sin_freqs
    out_imag = x_real * sin_freqs + x_imag * cos_freqs
    return mx.stack([out_real, out_imag], axis=-1).reshape(
        *x_pairs.shape[:-2], x_pairs.shape[-2] * 2
    )


def rope_apply(
    x: mx.array,
    grid_sizes: List[List[int]],
    freqs: Tuple[mx.array, mx.array],
) -> mx.array:
    """Apply 3-D RoPE (temporal + height + width) to *x*.

    Args:
        x: Input tensor ``[B, S, N, C]`` (batch, sequence, heads, channels).
        grid_sizes: Per-batch ``[[f, h, w], ...]`` spatial grid sizes.
        freqs: ``(cos_freqs, sin_freqs)`` each ``[max_seq_len, C // 2]``.

    Returns:
        Tensor of same shape and dtype as *x* (rotation computed in float32).
    """
    n = x.shape[2]
    c = x.shape[3] // 2  # half-dim for cos/sin pairs

    # Split frequency tables into temporal / height / width components
    c_t = c - 2 * (c // 3)
    c_h = c // 3
    c_w = c // 3

    cos_all, sin_all = freqs
    cos_splits = [cos_all[:, :c_t], cos_all[:, c_t:c_t + c_h], cos_all[:, c_t + c_h:]]
    sin_splits = [sin_all[:, :c_t], sin_all[:, c_t:c_t + c_h], sin_all[:, c_t + c_h:]]

    output = []
    for i, (f, h, w) in enumerate(grid_sizes):
        seq_len = f * h * w

        # Reshape to pair up dims: [seq_len, n, dim//2, 2]
        x_i = x[i, :seq_len].astype(mx.float32).reshape(seq_len, n, -1, 2)

        # Build per-axis cos/sin and broadcast over the 3-D grid
        # Temporal: [f, 1, 1, c_t] -> [f, h, w, c_t]
        cos_t = mx.broadcast_to(cos_splits[0][:f].reshape(f, 1, 1, -1), (f, h, w, c_t))
        sin_t = mx.broadcast_to(sin_splits[0][:f].reshape(f, 1, 1, -1), (f, h, w, c_t))
        # Height: [1, h, 1, c_h] -> [f, h, w, c_h]
        cos_h = mx.broadcast_to(cos_splits[1][:h].reshape(1, h, 1, -1), (f, h, w, c_h))
        sin_h = mx.broadcast_to(sin_splits[1][:h].reshape(1, h, 1, -1), (f, h, w, c_h))
        # Width: [1, 1, w, c_w] -> [f, h, w, c_w]
        cos_w = mx.broadcast_to(cos_splits[2][:w].reshape(1, 1, w, -1), (f, h, w, c_w))
        sin_w = mx.broadcast_to(sin_splits[2][:w].reshape(1, 1, w, -1), (f, h, w, c_w))

        cos_i = mx.concatenate([cos_t, cos_h, cos_w], axis=-1).reshape(seq_len, 1, -1)
        sin_i = mx.concatenate([sin_t, sin_h, sin_w], axis=-1).reshape(seq_len, 1, -1)

        x_i = _apply_rope_rotation(x_i, cos_i, sin_i)

        # Append any padding beyond seq_len unchanged
        x_i = mx.concatenate([x_i.astype(x.dtype), x[i, seq_len:]], axis=0)
        output.append(x_i)

    return mx.stack(output)


class WanRMSNorm(nn.Module):
    """RMS normalization layer (statistics in float32).

    Args:
        dim: Feature dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization to ``[B, L, C]``."""
        return self._norm(x.astype(mx.float32)).astype(x.dtype) * self.weight

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)


class WanLayerNorm(nn.Module):
    """Layer normalization with optional elementwise affine.

    Args:
        dim: Feature dimension.
        eps: Epsilon for numerical stability.
        elementwise_affine: Whether to include learnable weight and bias.
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones(dim)
            self.bias = mx.zeros(dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply layer normalization.

        Args:
            x: Input tensor ``[B, L, C]``.

        Returns:
            Normalized tensor of same shape.
        """
        x_float = x.astype(mx.float32)
        mean = mx.mean(x_float, axis=-1, keepdims=True)
        var = mx.var(x_float, axis=-1, keepdims=True)
        x_norm = (x_float - mean) * mx.rsqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.weight.astype(mx.float32) + self.bias.astype(mx.float32)
        return x_norm.astype(x.dtype)


class WanSelfAttention(nn.Module):
    """Projections shared by the Wan attention variants.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        window_size: Window size for local attention (unused, kept for compat).
        qk_norm: Whether to apply RMS normalization to Q and K.
        eps: Epsilon for normalization.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()


class WanT2VCrossAttention(WanSelfAttention):
    """Text cross-attention with an optional per-block K/V cache."""

    def __call__(self, x: mx.array, context: mx.array, context_lens, crossattn_cache: Optional[dict] = None) -> mx.array:
        b, n, d = x.shape[0], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
                v = self.v(context).reshape(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
            v = self.v(context).reshape(b, -1, n, d)
        x = attention(q, k, v, k_lens=context_lens)
        return self.o(x.reshape(b, -1, n * d))


WAN_CROSSATTENTION_CLASSES = {"t2v_cross_attn": WanT2VCrossAttention}
