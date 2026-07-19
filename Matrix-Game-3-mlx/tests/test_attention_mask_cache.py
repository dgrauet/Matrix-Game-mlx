"""k_lens mask caching in attention().

The PyTorch reference uses flash-attention varlen (cu_seqlens packing) and
never materializes a mask. The MLX port, lacking a varlen SDPA, builds a
[B, 1, 1, Lk] boolean mask inside attention() at EVERY call — ~30 identical
builds per denoise step, since seq_lens is constant across layers. Caching
the mask is a port-internal optimization: the math is unchanged.

The combined causal+k_lens path ([B, 1, Lq, Lk], potentially huge) is
deliberately NOT cached.
"""

import mlx.core as mx
import numpy as np
import pytest

from wan.modules.attention import _MASK_CACHE, _MASK_CACHE_MAX, _k_lens_mask, attention


@pytest.fixture(autouse=True)
def clear_cache():
    _MASK_CACHE.clear()
    yield
    _MASK_CACHE.clear()


def test_same_lengths_reuse_the_same_mask_object():
    k_lens = mx.array([7, 5], dtype=mx.int32)
    m1 = _k_lens_mask(2, 10, k_lens)
    m2 = _k_lens_mask(2, 10, mx.array([7, 5], dtype=mx.int32))
    assert m1 is m2, "identical (b, lk, k_lens) must hit the cache"


def test_different_lengths_get_different_masks():
    m1 = _k_lens_mask(1, 10, mx.array([7], dtype=mx.int32))
    m2 = _k_lens_mask(1, 10, mx.array([5], dtype=mx.int32))
    assert m1 is not m2
    assert not np.array_equal(np.array(m1), np.array(m2))


def test_mask_values_are_correct():
    m = _k_lens_mask(2, 6, mx.array([4, 6], dtype=mx.int32))
    expected = np.zeros((2, 1, 1, 6), dtype=bool)
    expected[0, ..., :4] = True
    expected[1, ..., :6] = True
    assert np.array_equal(np.array(m), expected)


def test_cache_is_bounded():
    for i in range(_MASK_CACHE_MAX + 5):
        _k_lens_mask(1, 100 + i, mx.array([10], dtype=mx.int32))
    assert len(_MASK_CACHE) <= _MASK_CACHE_MAX


def test_attention_masked_equals_truncated_keys():
    """Ground truth: masking keys >= k_lens == attending over only k_lens keys."""
    mx.random.seed(0)
    b, lq, lk, n, d = 1, 4, 8, 2, 16
    q = mx.random.normal((b, lq, n, d))
    k = mx.random.normal((b, lk, n, d))
    v = mx.random.normal((b, lk, n, d))

    out_masked = attention(q, k, v, k_lens=mx.array([5], dtype=mx.int32))
    out_trunc = attention(q, k[:, :5], v[:, :5])

    assert np.allclose(np.array(out_masked), np.array(out_trunc), atol=1e-5)


def test_attention_twice_is_deterministic_through_cache():
    mx.random.seed(1)
    q = mx.random.normal((2, 4, 2, 16))
    k = mx.random.normal((2, 6, 2, 16))
    v = mx.random.normal((2, 6, 2, 16))
    k_lens = mx.array([6, 3], dtype=mx.int32)

    out1 = attention(q, k, v, k_lens=k_lens)
    out2 = attention(q, k, v, k_lens=k_lens)  # second call hits the cache

    assert np.array_equal(np.array(out1), np.array(out2))
    assert len(_MASK_CACHE) == 1
