import numpy as np
import pytest
import torch
import mlx.core as mx
from tests.conftest import import_upstream, assert_close

from matrix_game_mlx.inference.wan.modules.model import rope_params
from matrix_game_mlx.inference.wan.modules.causal_model import causal_rope_apply, rope_apply_frames, _check_rope_len

up_c = import_upstream("matrix_game.inference.wan.modules.causal_model")
up_m = import_upstream("matrix_game.inference.wan.modules.model")


def _freqs(n, d):
    mlx_f = tuple(mx.concatenate(p, axis=1) for p in zip(rope_params(n, d - 4 * (d // 6)),
                                                         rope_params(n, 2 * (d // 6)), rope_params(n, 2 * (d // 6))))
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
    assert_close(out, ref, atol=2e-5, rtol=0)


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
