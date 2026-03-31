"""Validation tests for attention module: MLX vs PyTorch SDPA."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import assert_close from conftest (loaded via importlib to avoid pytest auto-import issues)
import importlib.util
_conftest_spec = importlib.util.spec_from_file_location(
    "conftest", os.path.join(os.path.dirname(__file__), "conftest.py"))
_conftest = importlib.util.module_from_spec(_conftest_spec)
_conftest_spec.loader.exec_module(_conftest)
assert_close = _conftest.assert_close


def _run_torch_sdpa(q_np, k_np, v_np, softmax_scale=None, q_scale=None,
                    causal=False, k_lens=None):
    """Run PyTorch scaled_dot_product_attention as reference."""
    import torch
    import torch.nn.functional as F

    q = torch.from_numpy(q_np).float()
    k = torch.from_numpy(k_np).float()
    v = torch.from_numpy(v_np).float()

    if q_scale is not None:
        q = q * q_scale

    # Transpose [B, L, N, D] -> [B, N, L, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Build mask
    attn_mask = None
    if k_lens is not None:
        b = k.shape[0]
        lk = k.shape[2]
        # [B, 1, 1, Lk] boolean mask: True where valid
        k_pos = torch.arange(lk).view(1, 1, 1, -1)
        k_lens_t = torch.tensor(k_lens).view(b, 1, 1, 1)
        valid = k_pos < k_lens_t
        # Convert to additive mask: 0 for valid, -inf for masked
        attn_mask = torch.where(valid, 0.0, float('-inf'))

    scale = softmax_scale if softmax_scale is not None else 1.0 / (q.shape[-1] ** 0.5)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, scale=scale)

    # Transpose back [B, N, L, D] -> [B, L, N, D]
    out = out.transpose(1, 2)
    return out


def _run_mlx_attention(q_np, k_np, v_np, softmax_scale=None, q_scale=None,
                       causal=False, k_lens=None):
    """Run MLX attention."""
    import mlx.core as mx
    from wan.modules.attention import attention

    q = mx.array(q_np)
    k = mx.array(k_np)
    v = mx.array(v_np)

    k_lens_mx = mx.array(k_lens) if k_lens is not None else None

    out = attention(
        q, k, v,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        k_lens=k_lens_mx,
    )
    mx.eval(out)  # noqa: S307 - mx.eval is MLX's standard graph evaluation, not Python eval()
    return out


def test_attention_basic():
    """B=1, L=16, N=8, D=64, no masking."""
    rng = np.random.RandomState(42)
    B, L, N, D = 1, 16, 8, 64
    q = rng.randn(B, L, N, D).astype(np.float32)
    k = rng.randn(B, L, N, D).astype(np.float32)
    v = rng.randn(B, L, N, D).astype(np.float32)

    torch_out = _run_torch_sdpa(q, k, v)
    mlx_out = _run_mlx_attention(q, k, v)

    assert_close(mlx_out, torch_out, atol=1e-4, msg="basic attention mismatch")


def test_attention_with_k_lens():
    """B=2, Lq=16, Lk=32, N=4, D=32, with k_lens=[20, 32]."""
    rng = np.random.RandomState(42)
    B, Lq, Lk, N, D = 2, 16, 32, 4, 32
    q = rng.randn(B, Lq, N, D).astype(np.float32)
    k = rng.randn(B, Lk, N, D).astype(np.float32)
    v = rng.randn(B, Lk, N, D).astype(np.float32)
    k_lens = [20, 32]

    torch_out = _run_torch_sdpa(q, k, v, k_lens=k_lens)
    mlx_out = _run_mlx_attention(q, k, v, k_lens=k_lens)

    assert_close(mlx_out, torch_out, atol=1e-4, msg="k_lens attention mismatch")


def test_attention_with_q_scale():
    """Verify q_scale pre-multiplication works correctly."""
    rng = np.random.RandomState(42)
    B, L, N, D = 1, 16, 8, 64
    q = rng.randn(B, L, N, D).astype(np.float32)
    k = rng.randn(B, L, N, D).astype(np.float32)
    v = rng.randn(B, L, N, D).astype(np.float32)
    q_scale = 0.5

    torch_out = _run_torch_sdpa(q, k, v, q_scale=q_scale)
    mlx_out = _run_mlx_attention(q, k, v, q_scale=q_scale)

    assert_close(mlx_out, torch_out, atol=1e-4, msg="q_scale attention mismatch")


if __name__ == '__main__':
    test_attention_basic()
    print("PASS: test_attention_basic")
    test_attention_with_k_lens()
    print("PASS: test_attention_with_k_lens")
    test_attention_with_q_scale()
    print("PASS: test_attention_with_q_scale")
    print("All attention tests passed!")
