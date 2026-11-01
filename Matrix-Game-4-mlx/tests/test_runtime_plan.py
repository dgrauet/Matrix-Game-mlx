"""Ring KV planner and linear-order adapter vs upstream torch `plan_cache` (the oracle)."""
import mlx.core as mx
import numpy as np
import pytest
import torch

from matrix_game_mlx.inference.runtime import from_linear, plan_cache, to_linear
from tests.conftest import import_upstream

upstream = import_upstream("matrix_game.inference.runtime")

CASES = [(3, 2), (4, 2), (21, 2), (21, 405)]
HEADS, DIM = 1, 2


def _new_pair(window, tokens):
    shape = (1, window * tokens, HEADS, DIM)
    t = {"k": torch.zeros(shape), "v": torch.zeros(shape),
         "global_end_index": torch.zeros(1, dtype=torch.long), "local_end_index": torch.zeros(1, dtype=torch.long)}
    m = {"k": mx.zeros(shape), "v": mx.zeros(shape), "global_end_index": 0, "local_end_index": 0}
    return t, m


def _write(t, m, frame, start, end, tokens):
    t["k"][:, start:end] = float(frame)
    t["v"][:, start:end] = float(-frame)
    m["k"][:, start:end] = float(frame)
    m["v"][:, start:end] = float(-frame)


def _assert_same(t, m):
    for name in ("k", "v"):
        np.testing.assert_array_equal(np.array(m[name]), t[name].numpy())
    assert m["global_end_index"] == int(t["global_end_index"])
    assert m["local_end_index"] == int(t["local_end_index"])


def _drive(window, tokens, ring):
    """Yield (torch cache, mlx cache, frame) after each planned write with the real call pattern."""
    t, m = _new_pair(window, tokens)
    for frame in range(3 * window):
        start = frame * tokens
        for _ in range(2):  # denoise write, then clean rewrite at the same position
            ts, te = upstream.plan_cache(t, start, tokens, window, ring=ring)
            ms, me = plan_cache(m, start, tokens, window, ring=ring)
            assert (ms, me) == (ts, te)
            _write(t, m, frame, ts, te, tokens)
            yield t, m, frame


@pytest.mark.parametrize("ring", [True, False], ids=["ring", "linear"])
@pytest.mark.parametrize("window,tokens", CASES)
def test_plan_cache_matches_upstream(window, tokens, ring):
    for t, m, _ in _drive(window, tokens, ring):
        _assert_same(t, m)


@pytest.mark.parametrize("ring", [True, False], ids=["ring", "linear"])
def test_plan_cache_errors_like_upstream(ring):
    window, tokens = 4, 2
    t, m = _new_pair(window, tokens)
    for start in (0, 2):
        upstream.plan_cache(t, start, tokens, window, ring=ring)
        plan_cache(m, start, tokens, window, ring=ring)
    for bad in (3 * tokens + 2 * tokens, 0, 1):  # skipped frame, old position, unaligned
        with pytest.raises(ValueError) as e_t:
            upstream.plan_cache(t, bad, tokens, window, ring=ring)
        with pytest.raises(ValueError) as e_m:
            plan_cache(m, bad, tokens, window, ring=ring)
        assert str(e_m.value) == str(e_t.value)
    # inconsistent indices: a rewrite whose local end is below one frame, then one above the window
    for glob, loc in ((4, 0), (4, 100)):
        t2, m2 = _new_pair(window, tokens)
        t2["global_end_index"].fill_(glob)
        t2["local_end_index"].fill_(loc)
        m2["global_end_index"], m2["local_end_index"] = glob, loc
        with pytest.raises(ValueError) as e_t:
            upstream.plan_cache(t2, glob - tokens, tokens, window, ring=ring)
        with pytest.raises(ValueError) as e_m:
            plan_cache(m2, glob - tokens, tokens, window, ring=ring)
        assert str(e_m.value) == str(e_t.value) == "Invalid bounded KV range"


@pytest.mark.parametrize("window,tokens", CASES)
def test_to_linear_inverts_ring(window, tokens):
    lin_t, lin_m = _new_pair(window, tokens)
    ring_t, ring_m = _new_pair(window, tokens)
    for frame in range(3 * window):
        start = frame * tokens
        for _ in range(2):
            s, e = upstream.plan_cache(lin_t, start, tokens, window, ring=False)
            _write(lin_t, lin_m, frame, s, e, tokens)
            s, e = plan_cache(ring_m, start, tokens, window, ring=True)
            ring_m["k"][:, s:e] = float(frame)
            ring_m["v"][:, s:e] = float(-frame)
            lin = to_linear(ring_m, tokens, window)
            for name in ("k", "v"):
                np.testing.assert_array_equal(np.array(lin[name]), lin_t[name].numpy())
            assert lin["global_end_index"] == int(lin_t["global_end_index"])
            assert lin["local_end_index"] == int(lin_t["local_end_index"])
            back = from_linear(lin, tokens, window)
            for name in ("k", "v"):
                np.testing.assert_array_equal(np.array(back[name]), np.array(ring_m[name]))


def test_to_linear_copies_other_keys():
    _, m = _new_pair(3, 2)
    m["k_sink_raw"] = mx.ones((1, 2, HEADS, DIM))
    plan_cache(m, 0, 2, 3, ring=True)
    out = to_linear(m, 2, 3)
    assert out is not m and "k_sink_raw" in out
    np.testing.assert_array_equal(np.array(out["k_sink_raw"]), np.array(m["k_sink_raw"]))


def test_linear_matches_native_attention_layout():
    """plan_cache(linear) + sequential writes reproduces CausalWanSelfAttention's eviction layout."""
    from matrix_game_mlx.inference.wan.modules.causal_model import CausalWanSelfAttention as A
    import inspect
    assert "sink_tokens + num_evicted_tokens" in inspect.getsource(A.__call__)  # layout under test
    window, tokens, sink = 4, 2, 2
    m = {"k": mx.zeros((1, window * tokens, 1, 1)), "v": mx.zeros((1, window * tokens, 1, 1)),
         "global_end_index": 0, "local_end_index": 0}
    n = {"k": mx.zeros((1, window * tokens, 1, 1)), "global_end_index": 0, "local_end_index": 0}
    for frame in range(3 * window):
        cs = frame * tokens
        for _ in range(2):
            s, e = plan_cache(m, cs, tokens, window, ring=False)
            m["k"][:, s:e] = float(frame)
            # native logic (causal_model.py lines 142-157)
            ce = cs + tokens
            size = window * tokens
            if ce > n["global_end_index"] and tokens + n["local_end_index"] > size:
                ev = tokens + n["local_end_index"] - size
                rolled = n["local_end_index"] - ev - sink
                n["k"][:, sink:sink + rolled] = n["k"][:, sink + ev:sink + ev + rolled]
                le = n["local_end_index"] + ce - n["global_end_index"] - ev
            else:
                le = n["local_end_index"] + ce - n["global_end_index"]
            n["k"][:, le - tokens:le] = float(frame)
            n["global_end_index"], n["local_end_index"] = ce, le
            np.testing.assert_array_equal(np.array(m["k"]), np.array(n["k"]))
