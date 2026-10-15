import numpy as np
import pytest
import torch
import mlx.core as mx
from tests.conftest import import_upstream, assert_close

from matrix_game_mlx.inference.wan.modules.model import rope_freqs
from matrix_game_mlx.inference.wan.modules.causal_model import causal_rope_apply, rope_apply_frames, _check_rope_len

up_c = import_upstream("matrix_game.inference.wan.modules.causal_model")
up_m = import_upstream("matrix_game.inference.wan.modules.model")


def _freqs(n, d):
    mlx_f = rope_freqs(n, d)
    pt_f = torch.cat([up_m.rope_params(n, d - 4 * (d // 6)), up_m.rope_params(n, 2 * (d // 6)),
                      up_m.rope_params(n, 2 * (d // 6))], dim=1)
    return mlx_f, pt_f


@pytest.mark.parametrize("start", [0, 1, 21, 4000])
def test_causal_rope_apply(start):
    d, heads, grid = 128, 2, [[1, 3, 5]]
    x = np.random.default_rng(start).standard_normal((1, 15, heads, d)).astype(np.float32)
    fm, fp = _freqs(8192, d)
    out = causal_rope_apply(mx.array(x), grid, fm, start_frame=start)
    ref = up_c.causal_rope_apply(torch.from_numpy(x), torch.tensor(grid), fp, start_frame=start)
    np.testing.assert_array_equal(np.array(out), ref.numpy())                     # float64 product as upstream


@pytest.mark.parametrize("grid,pos", [([[2, 3, 5]], [3, 100]), ([[1, 3, 5]], [4000])])
def test_rope_apply_frames_non_contiguous_and_single(grid, pos):
    f, h, w = grid[0]
    x = np.random.default_rng(5).standard_normal((1, f * h * w, 2, 128)).astype(np.float32)
    fm, fp = _freqs(8192, 128)
    out = rope_apply_frames(mx.array(x), grid, fm, pos)
    ref = up_c.rope_apply_frames(torch.from_numpy(x), torch.tensor(grid), fp, torch.tensor(pos))
    np.testing.assert_array_equal(np.array(out), ref.numpy())


def test_rope_apply_frames_bad_frame_pos():
    fm, _ = _freqs(64, 128)
    x = mx.zeros((1, 30, 2, 128))
    with pytest.raises(ValueError):
        rope_apply_frames(x, [[2, 3, 5]], fm, [1, 2, 3])
    with pytest.raises(ValueError):
        rope_apply_frames(x, [[2, 3, 5]], fm, [])


def test_causal_rope_apply_bf16_batch_padded():
    d, grid = 128, [[1, 3, 5], [1, 3, 5]]
    x = np.random.default_rng(9).standard_normal((2, 20, 2, d)).astype(np.float32)  # 15 rope tokens + 5 padding
    fm, fp = _freqs(64, d)
    xb = mx.array(x).astype(mx.bfloat16)
    out = causal_rope_apply(xb, grid, fm, start_frame=5)
    assert out.dtype == mx.bfloat16
    ref = up_c.causal_rope_apply(torch.from_numpy(x).bfloat16(), torch.tensor(grid), fp, start_frame=5)
    assert_close(out.astype(mx.float32), ref.float(), atol=3e-2, rtol=0)


def test_rope_apply_frames_matches_causal_when_contiguous():
    d, grid = 128, [[2, 3, 5]]
    x = np.random.default_rng(3).standard_normal((1, 30, 2, d)).astype(np.float32)
    fm, fp = _freqs(1024, d)
    out = rope_apply_frames(mx.array(x), grid, fm, [7, 8])
    ref = up_c.rope_apply_frames(torch.from_numpy(x), torch.tensor(grid), fp, torch.tensor([7, 8]))
    assert_close(out, ref, atol=2e-5, rtol=0)


def test_rope_len_check():
    fm, _ = _freqs(16, 128)
    with pytest.raises(ValueError):
        _check_rope_len(fm, 17)


@pytest.mark.parametrize("start", [1, 3, 25])
def test_rope_bf16_bit_exact_vs_upstream(start):
    """bf16 q/k at the real geometry (405 tokens, 24 heads x 128): the float64 RoPE stage is bit-exact vs torch."""
    d, grid = 128, [[1, 15, 27]]
    x = torch.from_numpy(np.random.default_rng(start).standard_normal((1, 405, 24, d)).astype(np.float32) * 3).bfloat16()
    xm = mx.array(x.float().numpy()).astype(mx.bfloat16)
    fm, fp = _freqs(8192, d)
    out = causal_rope_apply(xm, grid, fm, start_frame=start)
    ref = up_c.causal_rope_apply(x, torch.tensor(grid), fp, start_frame=start)
    assert out.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.array(out.astype(mx.float32)), ref.float().numpy())
    out = rope_apply_frames(xm, grid, fm, [start + 4])                          # fixed-sink re-RoPE path
    ref = up_c.rope_apply_frames(x, torch.tensor(grid), fp, torch.tensor([start + 4]))
    np.testing.assert_array_equal(np.array(out.astype(mx.float32)), ref.float().numpy())
