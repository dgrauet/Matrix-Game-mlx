"""Sliding-window grouping in ActionModule via a single gather.

The reference groups mouse/keyboard conditions per temporal block with a
Python loop of strided slices + mx.stack — 2 x N_feats small fp32 copies per
DiT block per step (~99k dispatches on a standard run, smeltr sessions
mg3-klens-cache / mg3-localize-copies). The grouping is a plain sliding
window (stride = vae_time_compression_ratio, length = ratio * windows_size),
i.e. a gather with a precomputable index table: one mx.take per call, the
exact same elements in the same order, no arithmetic — bit-identical.
"""

import mlx.core as mx
import numpy as np

from wan.modules.action_module import _group_windows, _window_index_table


def _reference_grouping(cond, n_feats, ratio, windows_size):
    """The original loop+stack implementation, kept as ground truth."""
    pad_t = ratio * windows_size
    groups = []
    for i in range(n_feats):
        start_idx = ratio * (i - windows_size) + pad_t
        end_idx = i * ratio + pad_t
        groups.append(cond[:, start_idx:end_idx, :])
    return mx.stack(groups, axis=1)


def test_index_table_matches_loop_bounds():
    ratio, windows = 4, 3
    table = _window_index_table(5, ratio, windows)
    assert table.shape == (5, ratio * windows)
    pad_t = ratio * windows
    for i in range(5):
        start = ratio * (i - windows) + pad_t
        assert table[i].tolist() == list(range(start, start + ratio * windows))


def test_gather_is_bit_identical_to_loop():
    mx.random.seed(0)
    for n_feats, ratio, windows in [(5, 4, 3), (13, 4, 3), (1, 4, 3), (7, 2, 5)]:
        pad_t = ratio * windows
        cond = mx.random.normal((2, pad_t + n_feats * ratio + 3, 6)).astype(mx.float32)
        got = _group_windows(cond, n_feats, ratio, windows)
        want = _reference_grouping(cond, n_feats, ratio, windows)
        assert got.shape == want.shape
        assert np.array_equal(np.array(got), np.array(want)), (n_feats, ratio, windows)


def test_index_table_is_cached():
    t1 = _window_index_table(5, 4, 3)
    t2 = _window_index_table(5, 4, 3)
    assert t1 is t2


def test_rotary_table_is_cached_and_correct():
    """get_rotary_pos_embed is a pure table (dims, head_dim, patch, theta):
    rebuilt at EVERY block call it was ~7.4k fp32 copies per step (session
    mg3-localize-2); cached by value it must return the same objects."""
    from wan.modules.action_module import _ROPE_TABLE_CACHE, ActionModule

    mod = ActionModule.__new__(ActionModule)
    mod.patch_size = [1, 2, 2]
    mod.rope_theta = 256

    _ROPE_TABLE_CACHE.clear()
    c1, s1 = mod.get_rotary_pos_embed(4, 8, 8, 48)
    c2, s2 = mod.get_rotary_pos_embed(4, 8, 8, 48)
    assert c1 is c2 and s1 is s2, "same args must hit the cache"

    c3, s3 = mod.get_rotary_pos_embed(6, 8, 8, 48)
    assert c3 is not c1
    assert c3.shape[0] != c1.shape[0]

    # Different instance, same config: same table (cache is config-keyed).
    mod2 = ActionModule.__new__(ActionModule)
    mod2.patch_size = [1, 2, 2]
    mod2.rope_theta = 256
    c4, _ = mod2.get_rotary_pos_embed(4, 8, 8, 48)
    assert c4 is c1
    _ROPE_TABLE_CACHE.clear()


def test_spatial_expand_matches_broadcast_transpose_chain():
    """Mouse path: [B,T,W,D] -> [B*S,T,W*D] via broadcast(S last)+transpose
    must equal the contiguous expand_dims(1)+broadcast+reshape rewrite."""
    from wan.modules.action_module import _expand_spatial

    mx.random.seed(3)
    B, T, W, D, S = 2, 5, 12, 6, 7
    g = mx.random.normal((B, T, W, D)).astype(mx.float32)

    ref = mx.broadcast_to(mx.expand_dims(g, -1), (B, T, W, D, S))
    ref = ref.transpose(0, 4, 1, 2, 3).reshape(B * S, T, -1)

    got = _expand_spatial(g, S)
    assert got.shape == ref.shape
    assert np.array_equal(np.array(got), np.array(ref))


def test_tile_batch_matches_concat_list():
    """Keyboard path: [B,L,H,D] -> [S*B,L,H,D] via concatenate([k]*S) must
    equal the broadcast rewrite (same s-major row order)."""
    from wan.modules.action_module import _tile_batch

    mx.random.seed(4)
    B, L, H, D, S = 2, 5, 3, 4, 6
    k = mx.random.normal((B, L, H, D)).astype(mx.float32)

    ref = mx.concatenate([k] * S, axis=0)
    got = _tile_batch(k, S)
    assert got.shape == ref.shape
    assert np.array_equal(np.array(got), np.array(ref))
