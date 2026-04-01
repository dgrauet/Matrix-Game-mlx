"""DiT backbone and core components for MLX.

Ports the full WanModel DiT backbone from the PyTorch reference, including
WanSelfAttention, WanCrossAttention, WanAttentionBlock, Head, and WanModel,
along with the previously ported sinusoidal_embedding_1d, rope_params,
rope_apply, rope_apply_with_indices, WanRMSNorm, and WanLayerNorm.
"""
import math
from typing import List, Optional, Tuple, Union

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .attention import attention

__all__ = [
    'sinusoidal_embedding_1d',
    'rope_params',
    'rope_apply',
    'rope_apply_with_indices',
    'WanRMSNorm',
    'WanLayerNorm',
    'WanSelfAttention',
    'WanCrossAttention',
    'WanAttentionBlock',
    'Head',
    'WanModel',
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


class WanSelfAttention(nn.Module):
    """Self-attention with RoPE for the DiT backbone.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        window_size: Window size for local attention (unused in MLX, kept for compat).
        qk_norm: Whether to apply RMS normalization to Q and K.
        eps: Epsilon for normalization.
        use_memory: Whether to use memory-based RoPE indexing.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        use_memory: bool = False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_memory = use_memory

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else None

    def __call__(
        self,
        x: mx.array,
        seq_lens: mx.array,
        grid_sizes: Union[mx.array, List[List[int]]],
        freqs: Tuple[mx.array, mx.array],
        memory_length: int = 0,
        memory_latent_idx: Optional[List[int]] = None,
        predict_latent_idx: Optional[Union[Tuple[int, int], List[int]]] = None,
        **kwargs,
    ) -> mx.array:
        """Self-attention forward pass.

        Args:
            x: Input tensor [B, L, C].
            seq_lens: Sequence lengths [B].
            grid_sizes: Spatial grid sizes [B, 3] as (F, H, W).
            freqs: RoPE frequencies (cos, sin) tuple.
            memory_length: Number of memory frames.
            memory_latent_idx: Indices for memory frames.
            predict_latent_idx: Indices for prediction frames.

        Returns:
            Output tensor [B, L, C].
        """
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        # Convert grid_sizes to list if needed
        if isinstance(grid_sizes, mx.array):
            grid_sizes_list = grid_sizes.tolist()
        else:
            grid_sizes_list = grid_sizes

        q = self.q(x)
        if self.norm_q is not None:
            q = self.norm_q(q)
        q = q.reshape(b, s, n, d)

        k = self.k(x)
        if self.norm_k is not None:
            k = self.norm_k(k)
        k = k.reshape(b, s, n, d)

        v = self.v(x).reshape(b, s, n, d)

        if self.use_memory:
            if memory_length > 0:
                hw = grid_sizes_list[0][1] * grid_sizes_list[0][2]
                q_pred = q[:, memory_length * hw:]
                k_pred = k[:, memory_length * hw:]

                grid_sizes_pred = [list(gs) for gs in grid_sizes_list]
                grid_sizes_pred[0][0] = grid_sizes_pred[0][0] - memory_length

                if predict_latent_idx is not None:
                    if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
                        start_idx, end_idx = predict_latent_idx
                        pred_indices = list(range(start_idx, end_idx))
                    else:
                        pred_indices = list(predict_latent_idx)
                else:
                    pred_indices = list(range(grid_sizes_pred[0][0]))

                q_pred = rope_apply_with_indices(q_pred, grid_sizes_pred, freqs, pred_indices)
                k_pred = rope_apply_with_indices(k_pred, grid_sizes_pred, freqs, pred_indices)

                q_memory = q[:, :memory_length * hw]
                k_memory = k[:, :memory_length * hw]
                grid_sizes_mem = [list(gs) for gs in grid_sizes_list]
                grid_sizes_mem[0][0] = memory_length

                if memory_latent_idx is not None:
                    mem_indices = list(memory_latent_idx)
                else:
                    mem_indices = list(range(memory_length))

                q_memory = rope_apply_with_indices(q_memory, grid_sizes_mem, freqs, mem_indices)
                k_memory = rope_apply_with_indices(k_memory, grid_sizes_mem, freqs, mem_indices)

                q = mx.concatenate([q_memory, q_pred], axis=1)
                k = mx.concatenate([k_memory, k_pred], axis=1)
            else:
                if predict_latent_idx is not None:
                    if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
                        start_idx, end_idx = predict_latent_idx
                        pred_indices = list(range(start_idx, end_idx))
                    else:
                        pred_indices = list(predict_latent_idx)
                else:
                    pred_indices = list(range(grid_sizes_list[0][0]))

                q = rope_apply_with_indices(q, grid_sizes_list, freqs, pred_indices)
                k = rope_apply_with_indices(k, grid_sizes_list, freqs, pred_indices)
        else:
            q = rope_apply(q, grid_sizes_list, freqs)
            k = rope_apply(k, grid_sizes_list, freqs)

        x = attention(q=q, k=k, v=v, k_lens=seq_lens)
        x = x.reshape(b, s, -1)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    """Cross-attention: queries from x, keys/values from context."""

    def __call__(  # type: ignore[override]
        self,
        x: mx.array,
        context: mx.array,
        context_lens: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """Cross-attention forward pass.

        Args:
            x: Query tensor [B, L1, C].
            context: Key/value tensor [B, L2, C].
            context_lens: Context sequence lengths [B].

        Returns:
            Output tensor [B, L1, C].
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        q = self.q(x)
        if self.norm_q is not None:
            q = self.norm_q(q)
        q = q.reshape(b, -1, n, d)

        k = self.k(context)
        if self.norm_k is not None:
            k = self.norm_k(k)
        k = k.reshape(b, -1, n, d)

        v = self.v(context).reshape(b, -1, n, d)

        x = attention(q, k, v, k_lens=context_lens)
        x = x.reshape(b, -1, n * d)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    """Single transformer block with self-attention, cross-attention, and FFN.

    Args:
        dim: Hidden dimension.
        ffn_dim: FFN intermediate dimension.
        num_heads: Number of attention heads.
        window_size: Window size for local attention.
        qk_norm: Whether to apply QK normalization.
        cross_attn_norm: Whether to use learnable norm for cross-attention.
        eps: Epsilon for normalization.
        action_config: Configuration dict for action module.
        block_idx: Block index (for action module block selection).
        use_memory: Whether to use memory mode.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        action_config: dict = {},
        block_idx: int = 0,
        use_memory: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_memory = use_memory

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps, use_memory=use_memory
        )

        if len(action_config) != 0 and block_idx in action_config['blocks']:
            from .action_module import ActionModule
            self.action_model = ActionModule(**action_config)
        else:
            self.action_model = None

        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else None
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)

        # FFN: Linear -> GELU(tanh) -> Linear
        self.ffn_linear1 = nn.Linear(dim, ffn_dim)
        self.ffn_linear2 = nn.Linear(ffn_dim, dim)

        self.modulation = mx.random.normal((1, 6, dim)) / (dim ** 0.5)

        if use_memory:
            self.cam_injector_layer1 = nn.Linear(dim, dim)
            self.cam_injector_layer2 = nn.Linear(dim, dim)
            self.cam_scale_layer = nn.Linear(dim, dim)
            self.cam_shift_layer = nn.Linear(dim, dim)

    def _ffn(self, x: mx.array) -> mx.array:
        """Apply FFN: Linear -> GELU(tanh) -> Linear."""
        x = self.ffn_linear1(x)
        x = nn.gelu_approx(x)
        x = self.ffn_linear2(x)
        return x

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        seq_lens: mx.array,
        grid_sizes: Union[mx.array, List[List[int]]],
        freqs: Tuple[mx.array, mx.array],
        context: mx.array,
        context_lens: Optional[mx.array] = None,
        mouse_cond: Optional[mx.array] = None,
        keyboard_cond: Optional[mx.array] = None,
        plucker_emb: Optional[mx.array] = None,
        mouse_cond_memory: Optional[mx.array] = None,
        keyboard_cond_memory: Optional[mx.array] = None,
        memory_length: int = 0,
        memory_latent_idx: Optional[List[int]] = None,
        predict_latent_idx: Optional[Union[Tuple[int, int], List[int]]] = None,
        **kwargs,
    ) -> mx.array:
        """Transformer block forward pass.

        Args:
            x: Hidden states [B, L, C].
            e: Time embedding [B, L1, 6, C].
            seq_lens: Sequence lengths [B].
            grid_sizes: Grid sizes [B, 3].
            freqs: RoPE frequencies.
            context: Text embeddings [B, L2, C].
            context_lens: Context lengths [B].
            mouse_cond: Mouse condition.
            keyboard_cond: Keyboard condition.
            plucker_emb: Plucker ray embeddings.
            mouse_cond_memory: Mouse memory conditioning.
            keyboard_cond_memory: Keyboard memory conditioning.
            memory_length: Number of memory frames.
            memory_latent_idx: Memory frame indices.
            predict_latent_idx: Prediction frame indices.

        Returns:
            Updated hidden states [B, L, C].
        """
        # Convert grid_sizes to list for indexing
        if isinstance(grid_sizes, mx.array):
            grid_sizes_list = grid_sizes.tolist()
        else:
            grid_sizes_list = grid_sizes

        # Modulation: compute 6 modulation vectors
        e_mod = self.modulation[None] + e  # [B, L1, 6, C]
        # Split along dim 2 into 6 chunks
        e0 = e_mod[:, :, 0:1, :]  # shift for norm1
        e1 = e_mod[:, :, 1:2, :]  # scale for norm1
        e2 = e_mod[:, :, 2:3, :]  # gate for self_attn
        e3 = e_mod[:, :, 3:4, :]  # shift for norm2
        e4 = e_mod[:, :, 4:5, :]  # scale for norm2
        e5 = e_mod[:, :, 5:6, :]  # gate for ffn

        # Self-attention with modulation
        norm_x = self.norm1(x).astype(mx.float32)
        mod_x = (norm_x * (1 + e1.squeeze(2)) + e0.squeeze(2)).astype(x.dtype)

        if self.use_memory:
            y = self.self_attn(
                mod_x, seq_lens, grid_sizes, freqs, memory_length,
                memory_latent_idx=memory_latent_idx,
                predict_latent_idx=predict_latent_idx,
            )
        else:
            y = self.self_attn(mod_x, seq_lens, grid_sizes, freqs)

        x = x + y * e2.squeeze(2)

        # Camera injection (plucker embeddings)
        if plucker_emb is not None:
            c2ws_hidden = self.cam_injector_layer2(
                nn.silu(self.cam_injector_layer1(plucker_emb))
            )
            c2ws_hidden = c2ws_hidden + plucker_emb
            cam_scale = self.cam_scale_layer(c2ws_hidden)
            cam_shift = self.cam_shift_layer(c2ws_hidden)
            x = (1.0 + cam_scale) * x + cam_shift

        # Cross-attention
        if mouse_cond is not None or self.use_memory:
            x_normed = self.norm3(x) if self.norm3 is not None else x
            x = x_normed + self.cross_attn(x_normed, context, context_lens)
        else:
            x_normed = self.norm3(x) if self.norm3 is not None else x
            x = x + self.cross_attn(x_normed, context, context_lens)

        # Action model
        if self.action_model is not None:
            valid_len = 1
            for gs in grid_sizes_list[0]:
                valid_len *= gs

            if self.use_memory:
                x_valid = x[:, :valid_len, :]
                x_valid = self.action_model(
                    x_valid.astype(self.ffn_linear1.weight.dtype),
                    grid_sizes_list[0][0], grid_sizes_list[0][1], grid_sizes_list[0][2],
                    mouse_cond, keyboard_cond,
                    mouse_cond_memory, keyboard_cond_memory,
                )
                if x_valid.shape[1] < x.shape[1]:
                    x = mx.concatenate([x_valid, x[:, valid_len:, :]], axis=1)
                else:
                    x = x_valid
            else:
                x_valid = x[:, :valid_len, :]
                x_valid = self.action_model(
                    x_valid.astype(self.ffn_linear1.weight.dtype),
                    grid_sizes_list[0][0], grid_sizes_list[0][1], grid_sizes_list[0][2],
                    mouse_cond, keyboard_cond,
                )
                if x_valid.shape[1] < x.shape[1]:
                    x = mx.concatenate([x_valid, x[:, valid_len:, :]], axis=1)
                else:
                    x = x_valid

        # FFN with modulation
        norm_x2 = self.norm2(x).astype(mx.float32)
        ffn_input = (norm_x2 * (1 + e4.squeeze(2)) + e3.squeeze(2)).astype(
            self.ffn_linear1.weight.dtype
        )
        y = self._ffn(ffn_input)
        x = x + y * e5.squeeze(2)

        return x


class Head(nn.Module):
    """Output head: normalization + linear projection with modulation.

    Args:
        dim: Hidden dimension.
        out_dim: Output channels.
        patch_size: 3D patch dimensions (t, h, w).
        eps: Epsilon for normalization.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        linear_out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, linear_out_dim)
        self.modulation = mx.random.normal((1, 2, dim)) / (dim ** 0.5)

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        """Head forward pass.

        Args:
            x: Hidden states [B, L1, C].
            e: Time embedding [B, L1, C].

        Returns:
            Output [B, L1, out_dim * prod(patch_size)].
        """
        # e: [B, L1, C] -> [B, L1, 1, C] + modulation [1, 2, C] -> [B, L1, 2, C]
        e_mod = self.modulation[None] + e[:, :, None, :]
        e0 = e_mod[:, :, 0, :]  # shift
        e1 = e_mod[:, :, 1, :]  # scale

        x = self.head(
            self.norm(x).astype(mx.float32) * (1 + e1) + e0
        )
        return x


class WanModel(nn.Module):
    """Wan diffusion backbone supporting text-to-video and image-to-video.

    Args:
        model_type: Model variant ('t2v', 'i2v', 'ti2v', 's2v').
        patch_size: 3D patch dimensions (t, h, w).
        text_len: Fixed length for text embeddings.
        in_dim: Input video channels.
        dim: Hidden dimension.
        ffn_dim: FFN intermediate dimension.
        freq_dim: Sinusoidal time embedding dimension.
        text_dim: Input text embedding dimension.
        out_dim: Output video channels.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        window_size: Window size for local attention.
        qk_norm: Whether to apply QK normalization.
        cross_attn_norm: Whether to use cross-attention normalization.
        eps: Epsilon for normalization layers.
        action_config: Action module configuration.
        use_memory: Whether to use memory mode.
        sigma_theta: Per-head RoPE frequency perturbation.
        use_text_crossattn: Whether to use text cross-attention.
    """

    def __init__(
        self,
        model_type: str = 't2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        action_config: dict = {},
        use_memory: bool = True,
        sigma_theta: float = 0.0,
        use_text_crossattn: bool = True,
    ):
        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_memory = use_memory
        self.sigma_theta = sigma_theta

        # Patch embedding: reshape + linear (replaces Conv3d patchify)
        patch_dim = in_dim * math.prod(patch_size)
        if model_type == 'i2v':
            patch_dim = in_dim * 2 * math.prod(patch_size)
        self.patch_embedding = nn.Linear(patch_dim, dim)

        if use_memory:
            self.patch_embedding_wancamctrl = nn.Linear(
                6 * 256 * math.prod(patch_size), dim
            )
            self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
            self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        # Text embedding
        self.text_embedding_linear1 = nn.Linear(text_dim, dim)
        self.text_embedding_linear2 = nn.Linear(dim, dim)

        # Time embedding
        self.time_embedding_linear1 = nn.Linear(freq_dim, dim)
        self.time_embedding_linear2 = nn.Linear(dim, dim)
        self.time_projection_linear1 = nn.Linear(dim, dim * 6)

        # Blocks
        self.blocks = [
            WanAttentionBlock(
                dim, ffn_dim, num_heads, window_size, qk_norm,
                cross_attn_norm, eps, action_config, i, use_memory
            )
            for i in range(num_layers)
        ]

        self.head = Head(dim, out_dim, patch_size, eps)

        # Build RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        if use_memory:
            max_seq_len = 2048
            if sigma_theta > 0:
                c = d // 2
                c_t = c - 2 * (c // 3)
                c_h = c // 3
                c_w = c // 3
                rope_epsilon = np.linspace(-1, 1, num_heads, dtype=np.float64)
                theta_base = 10000.0
                theta_hat = theta_base * (1 + sigma_theta * rope_epsilon)

                def build_freqs_component(
                    seq_len: int, c_part: int
                ) -> Tuple[np.ndarray, np.ndarray]:
                    exp = np.arange(c_part, dtype=np.float64) / c_part
                    # omega: [num_heads, c_part]
                    omega = 1.0 / np.power(theta_hat[:, None], exp[None, :])
                    pos = np.arange(seq_len, dtype=np.float64)
                    # angles: [num_heads, seq_len, c_part]
                    angles = pos[None, :, None] * omega[:, None, :]
                    return np.cos(angles), np.sin(angles)

                cos_t, sin_t = build_freqs_component(max_seq_len, c_t)
                cos_h, sin_h = build_freqs_component(max_seq_len, c_h)
                cos_w, sin_w = build_freqs_component(max_seq_len, c_w)

                # Concatenate: [num_heads, max_seq_len, c]
                cos_all = np.concatenate([cos_t, cos_h, cos_w], axis=2).astype(np.float32)
                sin_all = np.concatenate([sin_t, sin_h, sin_w], axis=2).astype(np.float32)
                self.freqs = (mx.array(cos_all), mx.array(sin_all))
            else:
                f1 = rope_params(max_seq_len, d - 4 * (d // 6))
                f2 = rope_params(max_seq_len, 2 * (d // 6))
                f3 = rope_params(max_seq_len, 2 * (d // 6))
                self.freqs = (
                    mx.concatenate([f1[0], f2[0], f3[0]], axis=1),
                    mx.concatenate([f1[1], f2[1], f3[1]], axis=1),
                )
        else:
            f1 = rope_params(1024, d - 4 * (d // 6))
            f2 = rope_params(1024, 2 * (d // 6))
            f3 = rope_params(1024, 2 * (d // 6))
            self.freqs = (
                mx.concatenate([f1[0], f2[0], f3[0]], axis=1),
                mx.concatenate([f1[1], f2[1], f3[1]], axis=1),
            )

    def _text_embedding(self, x: mx.array) -> mx.array:
        """Apply text embedding: Linear -> GELU(tanh) -> Linear."""
        x = self.text_embedding_linear1(x)
        x = nn.gelu_approx(x)
        x = self.text_embedding_linear2(x)
        return x

    def _time_embedding(self, x: mx.array) -> mx.array:
        """Apply time embedding: Linear -> SiLU -> Linear."""
        x = self.time_embedding_linear1(x)
        x = nn.silu(x)
        x = self.time_embedding_linear2(x)
        return x

    def _time_projection(self, x: mx.array) -> mx.array:
        """Apply time projection: SiLU -> Linear."""
        x = nn.silu(x)
        x = self.time_projection_linear1(x)
        return x

    def unpatchify(
        self, x: mx.array, grid_sizes: List[List[int]]
    ) -> List[mx.array]:
        """Reconstruct video tensors from patch embeddings.

        Args:
            x: Patchified features [B, L, C_out * prod(patch_size)].
            grid_sizes: Grid dimensions [[F_patches, H_patches, W_patches], ...].

        Returns:
            List of video tensors [F, H, W, C_out] (channels-last).
        """
        c = self.out_dim
        pt, ph, pw = self.patch_size
        out = []
        for i, v in enumerate(grid_sizes):
            f_p, h_p, w_p = v[0], v[1], v[2]
            total = f_p * h_p * w_p
            u = x[i, :total]
            # Head output is flattened as (c, pt, ph, pw) per patch — matching Conv3d layout
            # Reshape to [f_p, h_p, w_p, c, pt, ph, pw]
            u = u.reshape(f_p, h_p, w_p, c, pt, ph, pw)
            # Transpose to interleave: [f_p, pt, h_p, ph, w_p, pw, c]
            u = u.transpose(0, 4, 1, 5, 2, 6, 3)
            u = u.reshape(f_p * pt, h_p * ph, w_p * pw, c)
            out.append(u)
        return out

    def __call__(
        self,
        x: List[mx.array],
        t: mx.array,
        context: List[mx.array],
        seq_len: int,
        y: Optional[List[mx.array]] = None,
        mouse_cond: Optional[mx.array] = None,
        keyboard_cond: Optional[mx.array] = None,
        x_memory: Optional[mx.array] = None,
        timestep_memory: Optional[mx.array] = None,
        mouse_cond_memory: Optional[mx.array] = None,
        keyboard_cond_memory: Optional[mx.array] = None,
        plucker_emb: Optional[mx.array] = None,
        memory_latent_idx: Optional[List[int]] = None,
        predict_latent_idx: Optional[Union[Tuple[int, int], List[int]]] = None,
        return_memory: bool = False,
        **kwargs,
    ) -> Union[List[mx.array], Tuple[List[mx.array], List[mx.array]]]:
        """Forward pass through the diffusion model.

        Args:
            x: List of input video tensors [F, H, W, C_in] (channels-last).
            t: Diffusion timesteps [B] or [B, seq_len].
            context: List of text embeddings [L, C].
            seq_len: Maximum sequence length for positional encoding.
            y: Conditional video inputs for i2v mode.
            mouse_cond: Mouse conditioning.
            keyboard_cond: Keyboard conditioning.
            x_memory: Memory video tensor.
            timestep_memory: Memory timesteps.
            mouse_cond_memory: Mouse memory conditioning.
            keyboard_cond_memory: Keyboard memory conditioning.
            plucker_emb: Plucker ray embeddings.
            memory_latent_idx: Memory frame indices.
            predict_latent_idx: Prediction frame indices.
            return_memory: Whether to return memory and prediction separately.

        Returns:
            List of output video tensors [F, H, W, C_out] (channels-last).
        """
        pt, ph, pw = self.patch_size

        memory_length = 0
        if x_memory is not None:
            memory_length = x_memory.shape[1]
            x = [mx.concatenate([x_memory[i], xi], axis=0) for i, xi in enumerate(x)]
            t = mx.concatenate([timestep_memory, t], axis=1)

        if self.model_type == 'i2v':
            assert y is not None

        if y is not None:
            x = [mx.concatenate([u, v], axis=-1) for u, v in zip(x, y)]

        # Patchify and collect grid sizes
        x_list = []
        grid_sizes_list = []
        for u in x:
            f, h, w, c = u.shape
            f_p, h_p, w_p = f // pt, h // ph, w // pw
            grid_sizes_list.append([f_p, h_p, w_p])
            u = u.reshape(f_p, pt, h_p, ph, w_p, pw, c)
            # Order: (f_p, h_p, w_p, C, pt, ph, pw) — C before spatial patch dims
            # to match Conv3d weight layout (out, in_C, kF, kH, kW)
            u = u.transpose(0, 2, 4, 6, 1, 3, 5)
            u = u.reshape(1, f_p * h_p * w_p, c * pt * ph * pw)
            u = self.patch_embedding(u)
            x_list.append(u)

        # Text embedding with padding
        context_padded = []
        for u in context:
            if u.shape[0] < self.text_len:
                pad = mx.zeros((self.text_len - u.shape[0], u.shape[1]))
                u = mx.concatenate([u, pad], axis=0)
            context_padded.append(u)
        context_emb = self._text_embedding(mx.stack(context_padded))

        # Sequence lengths and padding
        seq_lens_list = [u.shape[1] for u in x_list]
        seq_lens = mx.array(seq_lens_list, dtype=mx.int32)
        assert max(seq_lens_list) <= seq_len

        x_padded = []
        for u in x_list:
            if u.shape[1] < seq_len:
                pad = mx.zeros((1, seq_len - u.shape[1], u.shape[2]))
                u = mx.concatenate([u, pad], axis=1)
            x_padded.append(u)
        x_tensor = mx.concatenate(x_padded, axis=0)

        # Time embedding
        if t.ndim == 1:
            t = mx.broadcast_to(t[:, None], (t.shape[0], seq_len))
        elif t.ndim == 2 and t.shape[1] < seq_len:
            t = mx.concatenate([
                t, mx.zeros((t.shape[0], seq_len - t.shape[1]))
            ], axis=1)

        bt = t.shape[0]
        t_flat = t.reshape(-1)
        e = self._time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_flat)
            .reshape(bt, seq_len, -1)
            .astype(mx.float32)
        )
        e0 = self._time_projection(e).reshape(bt, seq_len, 6, self.dim)
        e = e.astype(mx.float32)
        e0 = e0.astype(mx.float32)

        context_lens = None

        # Plucker embeddings
        if plucker_emb is not None:
            if isinstance(plucker_emb, list):
                plucker_items = plucker_emb
            else:
                plucker_items = [plucker_emb[i:i+1] for i in range(plucker_emb.shape[0])]

            plucker_patches = []
            for item in plucker_items:
                if item.ndim == 4:
                    item = item[None]
                b_p, f_tot, h_tot, w_tot, c_p = item.shape
                f_p = f_tot // pt
                h_p_pl = h_tot // ph
                w_p_pl = w_tot // pw
                item = item.reshape(b_p, f_p, pt, h_p_pl, ph, w_p_pl, pw, c_p)
                item = item.transpose(0, 1, 3, 5, 7, 2, 4, 6)
                item = item.reshape(b_p, f_p * h_p_pl * w_p_pl, c_p * pt * ph * pw)
                plucker_patches.append(item)

            plucker_emb = mx.concatenate(plucker_patches, axis=1)
            if plucker_emb.shape[1] < seq_len:
                pad = mx.zeros((
                    plucker_emb.shape[0],
                    seq_len - plucker_emb.shape[1],
                    plucker_emb.shape[2],
                ))
                plucker_emb = mx.concatenate([plucker_emb, pad], axis=1)

            plucker_emb = self.patch_embedding_wancamctrl(plucker_emb)
            plucker_hidden = self.c2ws_hidden_states_layer2(
                nn.silu(self.c2ws_hidden_states_layer1(plucker_emb))
            )
            plucker_emb = plucker_emb + plucker_hidden

        # Build block kwargs
        if self.use_memory:
            block_kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes_list,
                freqs=self.freqs,
                context=context_emb,
                context_lens=context_lens,
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
                plucker_emb=plucker_emb,
                mouse_cond_memory=mouse_cond_memory,
                keyboard_cond_memory=keyboard_cond_memory,
                memory_length=memory_length,
                memory_latent_idx=memory_latent_idx,
                predict_latent_idx=predict_latent_idx,
            )
        else:
            block_kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes_list,
                freqs=self.freqs,
                context=context_emb,
                context_lens=context_lens,
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
            )

        for block in self.blocks:
            x_tensor = block(x_tensor, **block_kwargs)

        x_tensor = self.head(x_tensor, e)
        out = self.unpatchify(x_tensor, grid_sizes_list)

        if self.use_memory:
            if return_memory:
                mem_out = []
                pred_out = []
                for u in out:
                    mem_out.append(u[:memory_length * pt])
                    pred_out.append(u[memory_length * pt:])
                return mem_out, pred_out
            return [u[memory_length * pt:] for u in out]
        return [u.astype(mx.float32) for u in out]
