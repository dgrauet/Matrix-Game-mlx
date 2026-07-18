"""Validation of dtype discipline in the DiT forward pass.

The PyTorch reference runs under torch.amp autocast: the residual stream and
time-embedding modulation are float32, but every Linear / attention input is
automatically cast to bfloat16 at the matmul boundary (see the explicit casts
at ``Matrix-Game-3/wan/modules/model.py:500-502`` and the ``half()`` cast in
the reference ``flash_attention`` wrapper, ``attention.py:65-87``).

MLX has no autocast, so the port must cast manually. Without the casts a
float32 residual stream silently upcasts the bfloat16 weights at every gemm
(2x memory traffic, fp32 kernels throughout).
"""

import mlx.core as mx
import pytest

from wan.modules.model import Head, WanCrossAttention, WanSelfAttention

DIM = 96
NUM_HEADS = 4
SEQ = 24


def _bf16(module):
    module.set_dtype(mx.bfloat16)
    return module


class TestMatmulBoundaryCasts:
    """float32 activations must not upcast bfloat16 weights at matmuls."""

    def test_self_attention_output_is_weight_dtype(self):
        attn = _bf16(WanSelfAttention(DIM, NUM_HEADS))
        x = mx.random.normal((1, SEQ, DIM)).astype(mx.float32)
        seq_lens = mx.array([SEQ])
        grid_sizes = [[6, 2, 2]]
        half = DIM // NUM_HEADS // 2
        freqs = (
            mx.ones((SEQ, half), dtype=mx.float32),
            mx.zeros((SEQ, half), dtype=mx.float32),
        )

        out = attn(x, seq_lens, grid_sizes, freqs)

        assert out.dtype == mx.bfloat16, f"got {out.dtype}"

    def test_cross_attention_output_is_weight_dtype(self):
        attn = _bf16(WanCrossAttention(DIM, NUM_HEADS))
        x = mx.random.normal((1, SEQ, DIM)).astype(mx.float32)
        context = mx.random.normal((1, 8, DIM)).astype(mx.float32)

        out = attn(x, context, None)

        assert out.dtype == mx.bfloat16, f"got {out.dtype}"

    def test_head_output_is_weight_dtype(self):
        head = _bf16(Head(DIM, out_dim=8, patch_size=(1, 2, 2)))
        x = mx.random.normal((1, SEQ, DIM)).astype(mx.float32)
        e = mx.random.normal((1, SEQ, DIM)).astype(mx.float32)

        out = head(x, e)

        assert out.dtype == mx.bfloat16, f"got {out.dtype}"
