"""DiT core components: norms, embeddings, and RoPE for MLX.

Ports sinusoidal_embedding_1d, rope_params, rope_apply, rope_apply_with_indices,
WanRMSNorm, and WanLayerNorm from the PyTorch reference, converting complex-number
RoPE to real-valued cos/sin arithmetic.
"""
from typing import List, Optional, Tuple, Union

import numpy as np

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    'sinusoidal_embedding_1d',
    'rope_params',
    'rope_apply',
    'rope_apply_with_indices',
    'WanRMSNorm',
    'WanLayerNorm',
]


def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    """Compute 1-D sinusoidal positional embedding.

    Args:
        dim: Embedding dimension (must be even).
        position: 1-D array of positions.

    Returns:
        Embedding array of shape ``[len(position), dim]``.
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(mx.float32)

    # Use float32 (MLX GPU does not support float64). Compute inverse
    # frequencies in high-precision Python floats, then convert.
    inv_freq = np.power(10000.0, -np.arange(half, dtype=np.float64) / half)
    inv_freq_mx = mx.array(inv_freq.astype(np.float32))

    sinusoid = mx.expand_dims(position, 1) * mx.expand_dims(inv_freq_mx, 0)
    x = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1)
    return x


def rope_params(
    max_seq_len: int, dim: int, theta: float = 10000
) -> Tuple[mx.array, mx.array]:
    """Precompute RoPE frequency cos/sin tables.

    Unlike the PyTorch reference which returns complex exponentials via
    ``torch.polar``, this returns a ``(cos, sin)`` tuple of real arrays.

    Args:
        max_seq_len: Maximum sequence length.
        dim: Head dimension (must be even).
        theta: RoPE base frequency.

    Returns:
        Tuple ``(cos_freqs, sin_freqs)`` each of shape
        ``[max_seq_len, dim // 2]``.
    """
    assert dim % 2 == 0
    # Compute inverse frequencies in numpy float64 for precision, then
    # convert to float32 for MLX (GPU does not support float64).
    inv_freq = 1.0 / np.power(theta, np.arange(0, dim, 2, dtype=np.float64) / dim)
    inv_freq_mx = mx.array(inv_freq.astype(np.float32))

    freqs = mx.expand_dims(mx.arange(max_seq_len).astype(mx.float32), 1) * mx.expand_dims(
        inv_freq_mx, 0
    )
    cos_freqs = mx.cos(freqs)
    sin_freqs = mx.sin(freqs)
    return (cos_freqs, sin_freqs)


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
        Tensor of same shape as *x*, dtype float32.
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
        x_i = mx.concatenate([x_i, x[i, seq_len:]], axis=0)
        output.append(x_i)

    return mx.stack(output).astype(mx.float32)


def rope_apply_with_indices(
    x: mx.array,
    grid_sizes: List[List[int]],
    freqs: Tuple[mx.array, mx.array],
    t_indices: Optional[Union[mx.array, List[int]]] = None,
) -> mx.array:
    """Apply 3-D RoPE with optional custom temporal indices.

    Supports both 2-D freqs ``[max_seq_len, C//2]`` and per-head freqs
    ``[n_heads, max_seq_len, C//2]``.

    Args:
        x: Input tensor ``[B, S, N, C]``.
        grid_sizes: Per-batch ``[[f, h, w], ...]``.
        freqs: ``(cos_freqs, sin_freqs)`` each ``[max_seq_len, C//2]`` or
            ``[n_heads, max_seq_len, C//2]``.
        t_indices: Custom temporal indices. Can be a 1-D list/array (shared
            across batch) or 2-D ``[B, f]``.

    Returns:
        Tensor of same shape as *x*, dtype float32.
    """
    n = x.shape[2]
    c = x.shape[3] // 2

    c_t = c - 2 * (c // 3)
    c_h = c // 3
    c_w = c // 3

    cos_all, sin_all = freqs
    is_per_head = cos_all.ndim == 3  # [n_heads, max_seq_len, C//2]

    if is_per_head:
        cos_splits = [cos_all[:, :, :c_t], cos_all[:, :, c_t:c_t + c_h], cos_all[:, :, c_t + c_h:]]
        sin_splits = [sin_all[:, :, :c_t], sin_all[:, :, c_t:c_t + c_h], sin_all[:, :, c_t + c_h:]]
    else:
        cos_splits = [cos_all[:, :c_t], cos_all[:, c_t:c_t + c_h], cos_all[:, c_t + c_h:]]
        sin_splits = [sin_all[:, :c_t], sin_all[:, c_t:c_t + c_h], sin_all[:, c_t + c_h:]]

    output = []
    for i, (f, h, w) in enumerate(grid_sizes):
        seq_len = f * h * w
        x_i = x[i, :seq_len].astype(mx.float32).reshape(seq_len, n, -1, 2)

        # Resolve temporal indices
        if t_indices is None:
            t_idx = list(range(f))
        else:
            t_idx = t_indices
            if isinstance(t_idx, mx.array) and t_idx.ndim > 1:
                t_idx = t_idx[i]
            if isinstance(t_idx, mx.array):
                t_idx = t_idx.tolist()
            elif not isinstance(t_idx, list):
                t_idx = list(t_idx)

        if is_per_head:
            # Per-head freqs: [n_heads, max_seq_len, c_*]
            # Index temporal: [n_heads, f, c_t]
            cos_t = cos_splits[0][:, t_idx, :]
            sin_t = sin_splits[0][:, t_idx, :]
            cos_h_vals = cos_splits[1][:, :h, :]
            sin_h_vals = sin_splits[1][:, :h, :]
            cos_w_vals = cos_splits[2][:, :w, :]
            sin_w_vals = sin_splits[2][:, :w, :]

            # Permute to [f, n_heads, c_t] -> broadcast over grid
            # [f, 1, 1, n, c_t] -> [f, h, w, n, c_t]
            cos_t = mx.broadcast_to(
                cos_t.transpose(1, 0, 2).reshape(f, 1, 1, n, -1), (f, h, w, n, c_t)
            )
            sin_t = mx.broadcast_to(
                sin_t.transpose(1, 0, 2).reshape(f, 1, 1, n, -1), (f, h, w, n, c_t)
            )
            cos_h_bc = mx.broadcast_to(
                cos_h_vals.transpose(1, 0, 2).reshape(1, h, 1, n, -1), (f, h, w, n, c_h)
            )
            sin_h_bc = mx.broadcast_to(
                sin_h_vals.transpose(1, 0, 2).reshape(1, h, 1, n, -1), (f, h, w, n, c_h)
            )
            cos_w_bc = mx.broadcast_to(
                cos_w_vals.transpose(1, 0, 2).reshape(1, 1, w, n, -1), (f, h, w, n, c_w)
            )
            sin_w_bc = mx.broadcast_to(
                sin_w_vals.transpose(1, 0, 2).reshape(1, 1, w, n, -1), (f, h, w, n, c_w)
            )

            cos_i = mx.concatenate([cos_t, cos_h_bc, cos_w_bc], axis=-1).reshape(seq_len, n, -1)
            sin_i = mx.concatenate([sin_t, sin_h_bc, sin_w_bc], axis=-1).reshape(seq_len, n, -1)
        else:
            # Standard 2-D freqs
            cos_t = mx.broadcast_to(
                cos_splits[0][t_idx].reshape(f, 1, 1, -1), (f, h, w, c_t)
            )
            sin_t = mx.broadcast_to(
                sin_splits[0][t_idx].reshape(f, 1, 1, -1), (f, h, w, c_t)
            )
            cos_h_bc = mx.broadcast_to(
                cos_splits[1][:h].reshape(1, h, 1, -1), (f, h, w, c_h)
            )
            sin_h_bc = mx.broadcast_to(
                sin_splits[1][:h].reshape(1, h, 1, -1), (f, h, w, c_h)
            )
            cos_w_bc = mx.broadcast_to(
                cos_splits[2][:w].reshape(1, 1, w, -1), (f, h, w, c_w)
            )
            sin_w_bc = mx.broadcast_to(
                sin_splits[2][:w].reshape(1, 1, w, -1), (f, h, w, c_w)
            )

            cos_i = mx.concatenate([cos_t, cos_h_bc, cos_w_bc], axis=-1).reshape(seq_len, 1, -1)
            sin_i = mx.concatenate([sin_t, sin_h_bc, sin_w_bc], axis=-1).reshape(seq_len, 1, -1)

        x_i = _apply_rope_rotation(x_i, cos_i, sin_i)
        x_i = mx.concatenate([x_i, x[i, seq_len:]], axis=0)
        output.append(x_i)

    return mx.stack(output).astype(mx.float32)


class WanRMSNorm(nn.Module):
    """RMS normalization layer.

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
        """Apply RMS normalization.

        Args:
            x: Input tensor ``[B, L, C]``.

        Returns:
            Normalized tensor of same shape.
        """
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
