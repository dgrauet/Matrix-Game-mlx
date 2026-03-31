"""Attention module using MLX scaled dot-product attention.

Replaces PyTorch Flash Attention with mx.fast.scaled_dot_product_attention.
"""
import math
from typing import Optional, Tuple, Union

import mlx.core as mx

__all__ = ['attention']


def attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_lens: Optional[mx.array] = None,
    k_lens: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: Optional[mx.Dtype] = None,
    version: Optional[str] = None,
) -> mx.array:
    """Attention using mx.fast.scaled_dot_product_attention.

    Args:
        q: Query tensor [B, Lq, Nq, C1].
        k: Key tensor [B, Lk, Nk, C1].
        v: Value tensor [B, Lk, Nk, C2]. Nq must be divisible by Nk.
        q_lens: Query sequence lengths per batch [B] (unused, kept for API compat).
        k_lens: Key sequence lengths per batch [B] (used for masking).
        dropout_p: Dropout probability (unused in inference).
        softmax_scale: Scaling factor for QK^T. Defaults to 1/sqrt(C1).
        q_scale: Pre-scaling factor applied to queries before attention.
        causal: Whether to apply causal attention mask.
        window_size: Sliding window (unused, kept for API compat).
        deterministic: Unused, kept for API compat.
        dtype: Unused, kept for API compat.
        version: Unused, kept for API compat.

    Returns:
        Output tensor [B, Lq, Nq, C2].
    """
    # Apply query pre-scaling
    if q_scale is not None:
        q = q * q_scale

    # Compute softmax scale
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])

    # Transpose from [B, L, N, D] to [B, N, L, D] for MLX SDPA
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # Build attention mask
    mask = None
    if causal and k_lens is not None:
        # Combine causal mask with k_lens padding mask
        b, _, lq, _ = q.shape
        lk = k.shape[2]
        # Causal mask: query position i can attend to key positions <= i
        # For cross-attention with different lengths, use full causal
        q_pos = mx.arange(lq).reshape(1, 1, lq, 1)
        k_pos = mx.arange(lk).reshape(1, 1, 1, lk)
        causal_mask = q_pos >= k_pos  # [1, 1, Lq, Lk]
        # k_lens mask: mask out positions >= k_lens[b]
        k_lens_arr = mx.array(k_lens) if not isinstance(k_lens, mx.array) else k_lens
        len_mask = k_pos < k_lens_arr.reshape(b, 1, 1, 1)  # [B, 1, 1, Lk]
        mask = causal_mask & len_mask  # [B, 1, Lq, Lk]
    elif causal:
        mask = "causal"
    elif k_lens is not None:
        b = q.shape[0]
        lk = k.shape[2]
        k_pos = mx.arange(lk).reshape(1, 1, 1, lk)
        k_lens_arr = mx.array(k_lens) if not isinstance(k_lens, mx.array) else k_lens
        mask = k_pos < k_lens_arr.reshape(b, 1, 1, 1)  # [B, 1, 1, Lk]

    # Call MLX SDPA
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    # Transpose back from [B, N, L, D] to [B, L, N, D]
    out = out.transpose(0, 2, 1, 3)

    return out
