"""Castle runtimes (`CastleRuntime`) vs the native MLX block loop on the tiny model.

`ring-fused` runs the tiny model in bf16; `apple` and `apple-fast` run it in float16 (the bf16 weights cast, as the
loader builds the DiT), against the native loop in the same dtype; `apple-fast` also with its 11-latent window.

The runtime may differ from native only through the attention key order (ring KV -> SDPA summation order); RoPE,
norms, modulation, cross-attention and FFN round exactly as native. Teacher-forced checks reset the runtime's cache to
native's (ring order via `from_linear`) before every call, so rows the call does not write must match bitwise and only
the written slot and the output carry the summation-order deviation. Free-running checks bound the drift over a rollout.
"""
import gc
import json
import re
import sys
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from matrix_game_mlx.inference.rollout import set_attention_window
from matrix_game_mlx.inference.runtime import (RUNTIME_SPECS, CastleRuntime, RuntimeSpec, dit_dtype, from_linear,
                                               install_runtime, to_linear)
from matrix_game_mlx.inference.wan.modules.causal_model import CausalWanModel
from matrix_game_mlx.inference.wan.modules.model import WanT2VCrossAttention, rope_freqs
from tests.conftest import rel_l2
from tests.test_causal_model import CFG, build_pair

FSL, WINDOW, LAYERS, HEADS, HEAD_DIM = 2, CFG["local_attn_size"], CFG["num_layers"], CFG["num_heads"], 64
TOL = 1e-2
CHUNKS = 60
MEASURED = {}  # test name -> max rel L2 (output, written slot); printed with -s for the ledger


def _bf16_model():
    model, _ = build_pair()
    model.update(tree_map(lambda a: a.astype(mx.bfloat16), model.parameters()))
    model.set_rope_max_len(8192)
    return model


@pytest.fixture(scope="module")
def weights():
    return tree_flatten(_bf16_model().parameters())


def _runtime_weights(weights, runtime):
    """The tiny model's bf16 weights cast to `runtime`'s DiT dtype (float16 for apple / apple-fast)."""
    dtype = RUNTIME_SPECS[runtime].dtype
    return [(name, a.astype(dtype)) for name, a in weights]


def _model(weights, runtime=False, mode="ring-fused", window=None):
    """Tiny DiT from `weights`; `runtime` installs `mode`; `window` sets the attention window (one sink frame)."""
    model = CausalWanModel(**CFG)
    model.load_weights(weights, strict=True)
    model.set_rope_max_len(8192)
    if window is not None:
        set_attention_window(types.SimpleNamespace(model=model), window, 1)
    if runtime:
        install_runtime(types.SimpleNamespace(model=model), mode)
        installed, run = model.castle_runtime, model.castle_runtime.run
        installed.calls = 0

        def counted(*args, **kwargs):  # proves the block loop really went through the runtime
            installed.calls += 1
            return run(*args, **kwargs)
        installed.run = counted
    return model


def _caches(batch=1, window=WINDOW, dtype=mx.bfloat16):
    kv = [{"k": mx.zeros((batch, window * FSL, HEADS, HEAD_DIM), dtype),
           "v": mx.zeros((batch, window * FSL, HEADS, HEAD_DIM), dtype),
           "global_end_index": 0, "local_end_index": 0,
           "k_sink_raw": mx.zeros((batch, FSL, HEADS, HEAD_DIM), dtype)} for _ in range(LAYERS)]
    return kv, [{"is_init": False} for _ in range(LAYERS)]


def _clone(cache):
    return {k: mx.array(v) if isinstance(v, mx.array) else v for k, v in cache.items()}


def _call(model, x, t, ctx, kv, xc, position, mode, frames=1):
    out = model([mx.array(u) for u in x], t=mx.full((len(x), frames), t), context=mx.array(ctx), seq_len=frames * FSL,
                kv_cache=kv, crossattn_cache=xc, current_start=position * FSL, sink_rope_mode=mode)
    mx.eval(out, *[c[k] for c in kv for k in ("k", "v", "k_sink_raw")],
            *[c[k] for c in xc if c.get("is_init") for k in ("k", "v")])
    return out


def _schedule(chunks, clean_first=False):
    """(position, t) calls: denoise t=937.5 then clean t=0 per chunk; optionally a lone clean pass at position 0."""
    calls = [(0, 0.0)] if clean_first else []
    for i in range(chunks):
        p = i + int(clean_first)
        calls += [(p, 937.5), (p, 0.0)]
    return calls


def _inputs(rng, n):
    return [rng.standard_normal((1, 1, 2, 4, 12)).astype(np.float32) for _ in range(n)]


def _teacher_forced(weights, mode, *, calls, switch_at=None, compiled=True, dit_runtime="ring-fused", window=None,
                    context_dtype=None):
    """Run native and runtime per call from the same (native) cache state; assert per-call equivalence.

    Both models run in `dit_runtime`'s dtype; `window` (default: the tiny model's) is set on both and sizes the
    caches; `context_dtype` casts the text context before the call (bf16: what T5 produces)."""
    rng = np.random.default_rng(0)
    contexts = [rng.standard_normal((1, 8, 32)).astype(np.float32) for _ in range(2)]
    if context_dtype is not None:
        contexts = [mx.array(c).astype(context_dtype) for c in contexts]
    weights, dtype = _runtime_weights(weights, dit_runtime), RUNTIME_SPECS[dit_runtime].dtype
    native = _model(weights, window=window)
    runtime = _model(weights, runtime=True, mode=dit_runtime, window=window)
    assert window is None or window == runtime.castle_runtime.window   # the runtime's window, as CastleStream sets
    window = window or WINDOW
    runtime.castle_runtime.use_graph = compiled
    kv_n, xc_n = _caches(window=window, dtype=dtype)
    _, xc_r = _caches(window=window, dtype=dtype)
    worst = [0.0, 0.0]
    for index, ((position, t), x) in enumerate(zip(calls, _inputs(rng, len(calls)))):
        ctx = contexts[int(switch_at is not None and position >= switch_at)]
        if switch_at is not None and position == switch_at and t != 0.0:
            for cross in xc_n + xc_r:           # as CastleStream.set_prompt
                cross["is_init"] = False
        kv_r = [from_linear(_clone(c), FSL, window) for c in kv_n]
        out_r = _call(runtime, x, t, ctx, kv_r, xc_r, position, mode)
        out_n = _call(native, x, t, ctx, kv_n, xc_n, position, mode)
        msg = f"call {index} position {position} t={t}"
        err = rel_l2(out_r, out_n)
        assert err < TOL, msg
        worst[0] = max(worst[0], err)
        for layer, (cr, cn) in enumerate(zip(kv_r, kv_n)):
            lin = to_linear(cr, FSL, window)
            assert (lin["global_end_index"], lin["local_end_index"]) == (cn["global_end_index"], cn["local_end_index"])
            end = cn["local_end_index"]
            for name in ("k", "v"):
                a, b = lin[name], cn[name]
                assert mx.array_equal(a[:, :end - FSL], b[:, :end - FSL]), f"{msg} layer {layer} {name} history"
                assert mx.array_equal(a[:, end:], b[:, end:]), f"{msg} layer {layer} {name} tail"
                slot = rel_l2(a[:, end - FSL:end], b[:, end - FSL:end])
                assert slot < TOL, f"{msg} layer {layer} {name} slot"
                worst[1] = max(worst[1], slot)
            if position == 0:
                assert rel_l2(cr["k_sink_raw"], cn["k_sink_raw"]) < TOL, f"{msg} layer {layer} k_sink_raw"
            else:
                assert mx.array_equal(cr["k_sink_raw"], cn["k_sink_raw"]), f"{msg} layer {layer} k_sink_raw"
        for layer, (cr, cn) in enumerate(zip(xc_r, xc_n)):
            assert cr["is_init"] and cn["is_init"]
            for name in ("k", "v"):
                assert mx.array_equal(cr[name], cn[name]), f"{msg} layer {layer} cross {name}"
    assert runtime.castle_runtime.calls == len(calls) and getattr(native, "castle_runtime", None) is None
    assert out_r.dtype == out_n.dtype == dtype
    steady = sum(position > window for position, _ in calls)
    assert runtime.castle_runtime.replays == (steady if compiled else 0)
    return worst


RUNTIMES = ["ring-fused", "apple"]


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("mode", ["absolute", "fixed", "future"])
def test_runtime_matches_native(weights, mode, runtime):
    MEASURED[f"teacher_forced[{runtime},{mode}]"] = _teacher_forced(weights, mode, calls=_schedule(CHUNKS),
                                                                    dit_runtime=runtime)
    print("max rel L2 (output, written slot):", MEASURED)


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("mode", ["fixed", "future"])
def test_runtime_prompt_switch_matches_native(weights, runtime, mode):
    """Dynamic captions: bf16 prompt embeddings (T5 stays bf16) switched mid-stream, into a bf16 or float16 DiT."""
    MEASURED[f"prompt_switch[{runtime},{mode}]"] = _teacher_forced(weights, mode, calls=_schedule(CHUNKS), switch_at=10,
                                                            dit_runtime=runtime, context_dtype=mx.bfloat16)
    print("max rel L2 (output, written slot):", MEASURED)


@pytest.mark.parametrize("mode", ["absolute", "fixed", "future"])
def test_apple_fast_window_matches_native(weights, mode):
    """apple-fast's 11-latent window (set from the runtime, as CastleStream does) vs native float16 with the same
    window, teacher-forced through position 15: planning, ring eviction (position 11) and the compiled steady graph
    (positions 12-15) within the per-call tolerance. The tolerance does not resolve a fixed-sink re-RoPE error on
    the tiny model; the bitwise `test_runtime_linear_layout_is_bitwise_native[...-apple-fast]` pins that."""
    window = RUNTIME_SPECS["apple-fast"].window
    assert window == 11
    MEASURED[f"apple_fast_window[{mode}]"] = _teacher_forced(weights, mode, calls=_schedule(16),
                                                             dit_runtime="apple-fast", window=window)
    print("max rel L2 (output, written slot):", MEASURED)


@pytest.mark.parametrize("mode", ["absolute", "fixed", "future"])
def test_runtime_clean_first_call_empty_cache(weights, mode):
    MEASURED[f"clean_first[{mode}]"] = _teacher_forced(weights, mode, calls=_schedule(CHUNKS, clean_first=True))
    print("max rel L2 (output, written slot):", MEASURED)


@pytest.mark.parametrize("mode", ["absolute", "fixed", "future"])
def test_runtime_free_running_matches_native(weights, mode):
    """Each model keeps its own cache for the whole rollout (the runtime's ring state evolves on its own)."""
    rng = np.random.default_rng(1)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    native, runtime = _model(weights), _model(weights, runtime=True)
    (kv_n, xc_n), (kv_r, xc_r) = _caches(), _caches()
    worst = [0.0, 0.0]
    calls = _schedule(CHUNKS)
    for index, ((position, t), x) in enumerate(zip(calls, _inputs(rng, len(calls)))):
        out_r = _call(runtime, x, t, ctx, kv_r, xc_r, position, mode)
        out_n = _call(native, x, t, ctx, kv_n, xc_n, position, mode)
        err = rel_l2(out_r, out_n)
        assert err < TOL, f"call {index}"
        worst[0] = max(worst[0], err)
        for layer, (cr, cn) in enumerate(zip(kv_r, kv_n)):
            lin = to_linear(cr, FSL, WINDOW)
            for name in ("k", "v", "k_sink_raw"):
                if layer == 0:  # layer-0 K/V depend on the input only: bitwise in any slot order
                    assert mx.array_equal(lin[name], cn[name]), f"call {index} layer 0 {name}"
                else:
                    e = rel_l2(lin[name], cn[name])
                    assert e < TOL, f"call {index} layer {layer} {name}"
                    worst[1] = max(worst[1], e)
    assert runtime.castle_runtime.calls == len(calls)
    MEASURED[f"free_running[{mode}]"] = worst
    print("max rel L2 (output, cache):", MEASURED)


@pytest.mark.parametrize("runtime", RUNTIMES + ["apple-fast"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
@pytest.mark.parametrize("mode", ["absolute", "fixed", "future"])
def test_runtime_linear_layout_is_bitwise_native(weights, mode, fused, runtime):
    """With the ring disabled (native linear slot order) the runtime is bitwise native in its dtype: GPU RoPE, the
    composed block pieces (or the fused kernels, whose row reductions follow MLX's order at these widths),
    cross-attention and the fixed-sink key set round exactly as native; only the ring order differs. apple-fast
    runs with its 11-latent window on both models (set from the runtime, as CastleStream does), so the 60 chunks
    pass its eviction and pin its fixed-sink re-RoPE (sink at position - 11) bitwise."""
    rng = np.random.default_rng(4)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    weights, dtype = _runtime_weights(weights, runtime), RUNTIME_SPECS[runtime].dtype
    window = RUNTIME_SPECS[runtime].window if runtime == "apple-fast" else None
    native = _model(weights, window=window)
    castle = _model(weights, runtime=True, mode=runtime, window=window)
    window = castle.blocks[0].self_attn.local_attn_size
    castle.castle_runtime.ring = False
    castle.castle_runtime.fused = fused
    (kv_n, xc_n), (kv_r, xc_r) = _caches(window=window, dtype=dtype), _caches(window=window, dtype=dtype)
    calls = _schedule(CHUNKS, clean_first=True)
    for index, ((position, t), x) in enumerate(zip(calls, _inputs(rng, len(calls)))):
        out = _call(castle, x, t, ctx, kv_r, xc_r, position, mode)
        assert out.dtype == dtype and mx.array_equal(out, _call(native, x, t, ctx, kv_n, xc_n, position, mode)), \
            f"call {index}"
        for cr, cn in zip(kv_r, kv_n):
            for name in ("k", "v", "k_sink_raw"):
                assert mx.array_equal(cr[name], cn[name]), f"call {index} {name}"
    assert castle.castle_runtime.calls == len(calls)


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("mode", ["absolute", "fixed", "future"])
def test_runtime_fused_is_bitwise_unfused(weights, mode, runtime, monkeypatch):
    """The runtime runs the fused kernels by default, and they round exactly as the composed native pieces in its
    dtype: a 60-chunk free-running rollout (clean-first, ring layout) is bitwise equal with fused on and off."""
    from matrix_game_mlx.inference import kernels
    counts = {}
    for name in ("ln_modulate", "gate_ln_affine", "gate_residual", "rms_norm", "gelu_tanh"):
        def counted(*args, _f=getattr(kernels, name), _n=name, **kwargs):
            counts[_n] = counts.get(_n, 0) + 1
            return _f(*args, **kwargs)
        monkeypatch.setattr(kernels, name, counted)
    rng = np.random.default_rng(5)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    weights, dtype = _runtime_weights(weights, runtime), RUNTIME_SPECS[runtime].dtype
    fused, unfused = _model(weights, runtime=True, mode=runtime), _model(weights, runtime=True, mode=runtime)
    assert fused.castle_runtime.fused
    unfused.castle_runtime.fused = False
    for model in (fused, unfused):  # eager: the Python call counters below see every call (compiled: the trace only)
        model.castle_runtime.use_graph = False
    (kv_f, xc_f), (kv_u, xc_u) = _caches(dtype=dtype), _caches(dtype=dtype)
    calls = _schedule(CHUNKS, clean_first=True)
    for index, ((position, t), x) in enumerate(zip(calls, _inputs(rng, len(calls)))):
        assert mx.array_equal(_call(fused, x, t, ctx, kv_f, xc_f, position, mode),
                              _call(unfused, x, t, ctx, kv_u, xc_u, position, mode)), f"call {index}"
        for cf, cu in zip(kv_f, kv_u):
            for name in ("k", "v", "k_sink_raw"):
                assert mx.array_equal(cf[name], cu[name]), f"call {index} {name}"
    n = len(calls) * LAYERS
    # Kernels ran only on the fused model: per layer 2 ln_modulate, 1 gate_ln_affine, 1 gate_residual, 3 rms_norm.
    assert counts == {"ln_modulate": 2 * n, "gate_ln_affine": n, "gate_residual": n, "rms_norm": 3 * n,
                      "gelu_tanh": n}


def _rollout(model, mode, calls, inputs, ctx, contexts_at=None, dtype=mx.bfloat16):
    """Free-running rollout on fresh caches; returns per-call outputs and the caches after every call."""
    kv, xc = _caches(dtype=dtype)
    outs, states = [], []
    for index, ((position, t), x) in enumerate(zip(calls, inputs)):
        if contexts_at is not None and index == contexts_at[0]:
            ctx = contexts_at[1]
            for cross in xc:                                   # as CastleStream.set_prompt
                cross["is_init"] = False
        outs.append(_call(model, x, t, ctx, kv, xc, position, mode))
        # New array objects: eager slice assignment updates the cache's array objects in place.
        states.append([{name: mx.array(c[name]) for name in ("k", "v", "k_sink_raw")} for c in kv])
    return outs, states


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("mode", ["absolute", "fixed", "future"])
def test_compiled_matches_uncompiled(weights, mode, runtime):
    """The compiled steady-window graph against the eager runtime: the eager path (use_graph off) passes the
    teacher-forced per-call assertions of test_runtime_matches_native (which runs the compiled one), and a 60-chunk
    free-running rollout is bitwise equal (outputs and every cache tensor after every call) with the graph on and
    off; the compiled run traces its graph once."""
    MEASURED[f"eager_teacher_forced[{runtime},{mode}]"] = _teacher_forced(weights, mode, calls=_schedule(CHUNKS),
                                                                          compiled=False, dit_runtime=runtime)
    rng = np.random.default_rng(6)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    calls = _schedule(CHUNKS, clean_first=True)
    inputs = _inputs(rng, len(calls))
    weights, dtype = _runtime_weights(weights, runtime), RUNTIME_SPECS[runtime].dtype
    compiled, eager = _model(weights, runtime=True, mode=runtime), _model(weights, runtime=True, mode=runtime)
    assert compiled.castle_runtime.use_graph
    eager.castle_runtime.use_graph = False
    (out_c, st_c), (out_e, st_e) = (_rollout(m, mode, calls, inputs, ctx, dtype=dtype) for m in (compiled, eager))
    for index, (a, b, sa, sb) in enumerate(zip(out_c, out_e, st_c, st_e)):
        assert mx.array_equal(a, b), f"call {index}"
        for layer, (ca, cb) in enumerate(zip(sa, sb)):
            for name in ("k", "v", "k_sink_raw"):
                assert mx.array_equal(ca[name], cb[name]), f"call {index} layer {layer} {name}"
    steady = sum(position > WINDOW for position, _ in calls)
    assert steady > 0 and compiled.castle_runtime.replays == steady and eager.castle_runtime.replays == 0
    assert compiled.castle_runtime.captures == 1 and eager.castle_runtime.captures == 0
    print("max rel L2 (output, written slot):", MEASURED)


def test_compile_cache_bounded(weights):
    """One steady-window graph per session (upstream: one CUDA graph): positions, RoPE rotations, timestep
    modulation and a new prompt are inputs, so 60 chunks (with a prompt switch) compile exactly once; clear() drops
    the compiled state and the next steady call compiles again."""
    rng = np.random.default_rng(7)
    contexts = [rng.standard_normal((1, 8, 32)).astype(np.float32) for _ in range(2)]
    model = _model(weights, runtime=True)
    runtime = model.castle_runtime
    calls = _schedule(CHUNKS)
    _rollout(model, "fixed", calls, _inputs(rng, len(calls)), contexts[0], contexts_at=(len(calls) // 2, contexts[1]))
    steady = sum(position > WINDOW for position, _ in calls)
    print(f"compiled entries after {CHUNKS} chunks: {runtime.captures}, replays {runtime.replays} / {steady}")
    assert runtime.captures == 1 and runtime.replays == steady
    assert runtime.entry is not None
    runtime.clear()
    assert runtime.entry is None and not runtime.time_memo
    before = runtime.captures
    _rollout(model, "fixed", _schedule(WINDOW + 2), _inputs(rng, 2 * (WINDOW + 2)), contexts[0])
    assert runtime.captures == before + 1


def test_time_embedding_memo_is_bitwise(weights):
    """The runtime memoizes the float64-accumulated timestep MLP per timestep value: a hit returns exactly what the
    native computation returns, a weight change invalidates it, and clear() empties it."""
    model = _model(weights, runtime=True)
    runtime = model.castle_runtime
    for t in (937.5, 0.0, 937.5, 0.0, 937.5):
        got = runtime.time_embedding(model, mx.full((1, 1), t), mx.bfloat16)
        want = model.time_embedding(mx.full((1, 1), t), mx.bfloat16)
        assert all(mx.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(got, want)), t
    assert len(runtime.time_memo) == 2
    layer = model.time_projection_linear1
    layer.weight = layer.weight * 2
    got = runtime.time_embedding(model, mx.full((1, 1), 0.0), mx.bfloat16)
    want = model.time_embedding(mx.full((1, 1), 0.0), mx.bfloat16)
    assert all(mx.array_equal(a, b) for a, b in zip(got, want))
    runtime.clear()
    assert not runtime.time_memo


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_prime_compiles_and_leaves_clean_caches(weights, runtime):
    """prime (upstream's warm-up before gameplay) runs one steady-window call, so the graph is compiled before the
    first chunk, then erases every KV cache; a rollout from the primed caches is bitwise a rollout on fresh ones and
    compiles nothing more."""
    from matrix_game_mlx.inference.wrappers import WanDiffusionWrapper
    rng = np.random.default_rng(8)
    prompt = mx.array(rng.standard_normal((1, 8, 32)).astype(np.float32))
    weights, dtype = _runtime_weights(weights, runtime), RUNTIME_SPECS[runtime].dtype
    primed, fresh = _model(weights, runtime=True, mode=runtime), _model(weights, runtime=True, mode=runtime)
    gen = WanDiffusionWrapper(primed)
    kv, xc = _caches(dtype=dtype)
    primed.castle_runtime.prime(gen, prompt, (1, 1, 4, 2, 4), kv, xc, FSL, "fixed")
    assert primed.castle_runtime.captures == 1 and primed.castle_runtime.replays == 1
    for c in kv:
        assert (c["global_end_index"], c["local_end_index"]) == (0, 0)
        assert all(not bool(mx.any(c[name])) for name in ("k", "v", "k_sink_raw"))
    assert all(c["is_init"] for c in xc)                       # the real prompt's text K/V is reused, as upstream
    calls = _schedule(CHUNKS // 4)
    inputs = _inputs(rng, len(calls))
    kv_f, xc_f = _caches(dtype=dtype)
    for (position, t), x in zip(calls, inputs):
        a = _call(primed, x, t, prompt, kv, xc, position, "fixed")
        b = _call(fresh, x, t, prompt, kv_f, xc_f, position, "fixed")
        assert mx.array_equal(a, b)
        for ca, cb in zip(kv, kv_f):
            for name in ("k", "v", "k_sink_raw"):
                assert mx.array_equal(ca[name], cb[name])
    assert primed.castle_runtime.captures == 1


def test_prime_keeps_the_runtime_error(weights):
    """prime() resets only the cache tensors that exist: a cache the runtime rejects (here without `k_sink_raw`)
    surfaces the runtime's ValueError, not a KeyError from the cleanup."""
    from matrix_game_mlx.inference.wrappers import WanDiffusionWrapper
    model = _model(weights, runtime=True)
    kv, xc = _caches()
    for cache in kv:
        del cache["k_sink_raw"]
    with pytest.raises(ValueError, match="^" + re.escape(LAYOUT)):
        model.castle_runtime.prime(WanDiffusionWrapper(model), mx.zeros((1, 8, 32)), (1, 1, 4, 2, 4), kv, xc, FSL,
                                   "fixed")
    for cache in kv:
        assert "k_sink_raw" not in cache and (cache["global_end_index"], cache["local_end_index"]) == (0, 0)
        assert not bool(mx.any(cache["k"])) and not bool(mx.any(cache["v"]))


@pytest.mark.parametrize("runtime", ["ring-fused", "apple-fast"])
def test_runtime_second_stream_matches_fresh(weights, runtime):
    """A second stream on a used model (serve restarting a session) is bitwise a stream on a fresh model; apple-fast
    runs in float16 with its own window, set from the runtime as CastleStream does."""
    window = RUNTIME_SPECS[runtime].window if runtime == "apple-fast" else None
    rng = np.random.default_rng(2)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    weights, dtype = _runtime_weights(weights, runtime), RUNTIME_SPECS[runtime].dtype
    used = _model(weights, runtime=True, mode=runtime, window=window)
    window = used.blocks[0].self_attn.local_attn_size
    first_calls, calls = _schedule(window + 2), _schedule(window + 6)  # both reach the steady (compiled) state
    first = _inputs(rng, len(first_calls))
    inputs = _inputs(rng, len(calls))
    kv, xc = _caches(window=window, dtype=dtype)
    for (position, t), x in zip(first_calls, first):
        _call(used, x, t, ctx, kv, xc, position, "fixed")
    assert used.castle_runtime.captures == 1 and used.castle_runtime.replays > 0   # the compiled graph ran
    fresh = _model(weights, runtime=True, mode=runtime, window=window)
    (kv_u, xc_u), (kv_f, xc_f) = _caches(window=window, dtype=dtype), _caches(window=window, dtype=dtype)
    for (position, t), x in zip(calls, inputs):
        a = _call(used, x, t, ctx, kv_u, xc_u, position, "fixed")
        b = _call(fresh, x, t, ctx, kv_f, xc_f, position, "fixed")
        assert mx.array_equal(a, b)
        for cu, cf in zip(kv_u, kv_f):
            for name in ("k", "v", "k_sink_raw"):
                assert mx.array_equal(cu[name], cf[name])
    # The second stream reused the first stream's compiled graph: compiled state holds no session data.
    assert used.castle_runtime.captures == 1 and fresh.castle_runtime.captures == 1


class _PoolVAE:
    """Deterministic stand-in codec with TAEHV's streaming interface (pixels -> one 48-channel latent, latent -> 1
    then 4 frames), so a real CastleStream runs on the tiny DiT without loading a codec."""

    dtype = mx.bfloat16

    def __init__(self):
        self.decoded = 0

    def clear_cache(self):
        self.decoded = 0

    def encode_stream(self, x):                                   # [1, 3, T, 480, 864] -> [1, 48, 1, 30, 54]
        pooled = x[0].astype(mx.float32).mean(axis=1).reshape(3, 30, 16, 54, 16).mean(axis=(2, 4))
        return mx.repeat(pooled, 16, axis=0)[None, :, None].astype(self.dtype)

    def decode_stream(self, z):                                   # [1, 48, 1, 30, 54] -> [1, 3, count, 480, 864]
        count = 1 if self.decoded == 0 else 4
        self.decoded += count
        frame = mx.repeat(mx.repeat(mx.tanh(z[0, :3, 0].astype(mx.float32)), 16, axis=1), 16, axis=2)
        return mx.broadcast_to(frame[None, :, None], (1, 3, count, 480, 864)).astype(self.dtype)


def _stream_model(seed=0, mode="ring-fused"):
    """The tiny DiT at the stream's latent geometry (48 latent + 48 control channels, 30 x 54 latents) behind the
    generator wrapper, in `mode`'s dtype (bf16 weights cast, as the loader does) with `mode` installed."""
    from matrix_game_mlx.inference.wrappers import WanDiffusionWrapper
    mx.random.seed(seed)
    model = CausalWanModel(**{**CFG, "in_dim": 96, "out_dim": 48})
    model.update(tree_map(lambda a: (0.05 * mx.random.normal(a.shape)).astype(mx.bfloat16), model.parameters()))
    model.update(tree_map(lambda a: a.astype(dit_dtype(mode)), model.parameters()))
    gen = WanDiffusionWrapper(model)
    install_runtime(gen, mode)
    return gen


def _stream_chunks(gen, chunks, seed=3):
    """A real CastleStream session on `gen`: `chunks` pushes of deterministic frames, then close()."""
    from matrix_game_mlx.inference.stream import CastleStream
    prompt = mx.array(np.random.default_rng(9).standard_normal((1, 8, 32)).astype(np.float32)).astype(mx.bfloat16)
    # A first_frame sink (chunk 0 is the sink itself) keeps cache position == chunk index, as these tests count.
    stream = CastleStream(gen, _PoolVAE(), prompt, vae_backend="taehv", steps=(1000,), first_steps=(1000,),
                          seed=seed, sink_latent=mx.zeros((1, 1, 48, 30, 54), mx.bfloat16), sink_mode="first_frame")
    key, frame, out = mx.random.key(11), 0, []
    for index in range(chunks):
        count = 1 if index == 0 else 4
        key, sub = mx.random.split(key)
        chunk = stream.push(mx.random.uniform(-1, 1, (count, 3, 480, 864), key=sub), frame_start=frame)
        out.append((chunk.latent, chunk.pixels))
        frame += count
    stream.close()
    return out


class _WanStubVAE:
    """Wan-backend stand-in (bf16): records the dtype of what the stream's incremental decoder feeds the decoder."""

    dtype = mx.bfloat16

    def __init__(self):
        self.decoder_inputs, self.model = [], types.SimpleNamespace(encoder=None, conv2=lambda z: z)
        self.model.decoder = self._decode

    def _scale(self, dtype):
        return [mx.zeros((48,), dtype), mx.ones((48,), dtype)]

    def encode_stream(self, x, feat_cache):                        # [1, T, 480, 864, 3] -> [1, 1, 48, 30, 54]
        return mx.zeros((1, 1, 48, 30, 54), self.dtype)

    def _decode(self, x, feat_cache, feat_idx, first_chunk):       # one latent -> 1 then 4 frames, patch-2 layout
        self.decoder_inputs.append(x.dtype)
        return mx.zeros((1, 1 if first_chunk else 4, 240, 432, 12), self.dtype)


def test_stream_wan_two_steps_with_fp16_dit(monkeypatch):
    """`--preset quality --dit-runtime apple`: fp16 DiT, Wan VAE (bf16), two denoise steps. The apple runtime's dtype
    check sits after patch embedding, so completing shows the embedded input is fp16 (not every DiT input); the
    Wan decoder receives the VAE dtype (bf16), both denoising steps run and the decoded frames are finite."""
    from matrix_game_mlx.inference import stream as stream_module
    from matrix_game_mlx.inference.wan.modules import vae2_2
    monkeypatch.setattr(vae2_2, "count_conv3d", lambda module: 0)
    gen, vae = _stream_model(mode="apple"), _WanStubVAE()
    prompt = mx.zeros((1, 8, 32), mx.bfloat16)
    stream = stream_module.CastleStream(gen, vae, prompt, vae_backend="wan", steps=(1000, 500), first_steps=(1000, 500),
                                         sink_latent=mx.zeros((1, 1, 48, 30, 54), mx.bfloat16))
    assert stream.dtype == mx.float16
    frame = 0
    for count in (1, 4):
        chunk = stream.push(mx.zeros((count, 3, 480, 864)), frame_start=frame)
        frame += count
        assert chunk.denoising_steps == 2
        assert chunk.latent.dtype == mx.float16 and tuple(chunk.pixels.shape) == (count, 3, 480, 864)
        assert bool(mx.all(mx.isfinite(chunk.pixels))) and bool(mx.all(mx.isfinite(chunk.latent)))
    assert vae.decoder_inputs == [mx.bfloat16, mx.bfloat16]
    stream.close()


@pytest.mark.parametrize("mode", ["ring-fused", "apple-fast"])
def test_stream_second_session_matches_fresh_model(mode):
    """Stop -> Play on one model: a CastleStream pushed past the attention window (compiled graph replayed) and
    closed leaves nothing behind; the next stream on the same model is bitwise a stream on a fresh model. The
    stream sets the window from the runtime, so apple-fast's graph starts at position 12 instead of 22."""
    window = RUNTIME_SPECS[mode].window
    chunks = window + 3                                           # the last two positions run the compiled graph
    used, fresh = _stream_model(mode=mode), _stream_model(mode=mode)
    _stream_chunks(used, chunks, seed=5)
    runtime = used.model.castle_runtime
    assert runtime.replays == 2 * 2 + 1 and runtime.captures == 1   # prime + 2 steady chunks x (denoise, clean)
    assert runtime.entry is None and not runtime.time_memo         # close() cleared the session state
    second, reference = _stream_chunks(used, chunks), _stream_chunks(fresh, chunks)
    for index, ((la, pa), (lb, pb)) in enumerate(zip(second, reference)):
        assert mx.array_equal(la, lb) and mx.array_equal(pa, pb), f"chunk {index}"
    assert runtime.captures == 2 and fresh.model.castle_runtime.captures == 1   # each session primes its graph
    assert runtime.replays == 2 * (2 * 2 + 1) and fresh.model.castle_runtime.replays == 2 * 2 + 1


@pytest.mark.needs_weights
@pytest.mark.parametrize("mode", ("ring-fused", "apple"))
@pytest.mark.parametrize("scene, sink_rope_mode", (("castle", "fixed"), ("racing", "future")))
def test_compiled_graph_is_bitwise_eager_on_real_weights(mode, scene, sink_rope_mode):
    """Real castle (fixed sink) and racing (future sink: the sink is re-rotated to position 40) weights, one steady
    position (denoise then cache-update call) on random full-window caches: the compiled steady-window graph equals the eager runtime bit for bit, prediction and every K/V. Guards
    an MLX upgrade: mx.compile is not bitwise for every op (README, Apple fast path parity). The DiT is built in
    `dit_dtype(mode)` (apple: float16); the prompt and control latent stay bf16, as the stream feeds them (the T5
    and the TAEHV are bf16), the latent and caches are in the DiT dtype."""
    from matrix_game_mlx.inference.loading import build_dit
    from matrix_game_mlx.inference.rollout import set_attention_window
    from matrix_game_mlx.inference.wrappers import init_crossattn_cache, init_kv_cache
    from tests.conftest import weights_dir
    for name in ("last_type", "last_value", "last_traceback", "last_exc"):
        sys.__dict__.pop(name, None)          # a failed case's postmortem traceback would keep its DiT alive
    gc.collect()
    mx.clear_cache()
    dtype = dit_dtype(mode)
    pack = weights_dir()
    component = json.loads((pack / "config.json").read_text())["scenes"][scene]["component"]
    gen = build_dit(pack, component, dtype=dtype)
    install_runtime(gen, mode)
    runtime = gen.model.castle_runtime
    set_attention_window(gen, 21, 1)
    gen.model.set_rope_max_len(8192)
    fsl, position, names = 405, 29, ("k", "v", "k_sink_raw")

    def run(use_graph):
        runtime.clear()
        runtime.use_graph = use_graph
        keys = iter(mx.random.split(mx.random.key(0), 200))
        normal = lambda shape, dt=dtype: mx.random.normal(shape, key=next(keys)).astype(dt)
        kv, xc = init_kv_cache(gen, 1, 21 * fsl, dtype), init_crossattn_cache(gen, 1, dtype)
        for cache in kv:
            cache.update(k=normal(cache["k"].shape), v=normal(cache["v"].shape),
                         k_sink_raw=normal(cache["k"][:, :fsl].shape), global_end_index=position * fsl,
                         local_end_index=21 * fsl)
        prompt, preds = normal((1, 512, 4096), mx.bfloat16), []
        for t, control in ((1000.0, True), (0.0, False)):
            cond = {"prompt_embeds": prompt, "control_latent": normal((1, 1, 48, 30, 54), mx.bfloat16)}
            _, pred = gen(noisy_image_or_video=normal((1, 1, 48, 30, 54)), conditional_dict=cond,
                          timestep=mx.full((1, 1), t), kv_cache=kv, crossattn_cache=xc, current_start=position * fsl,
                          sink_rope_mode=sink_rope_mode, drop_control=not control)
            mx.eval(pred, *[c[name] for c in kv for name in names])
            preds.append(pred)
        return preds, [[c[name] for name in names] for c in kv]

    eager = run(False)
    assert (runtime.captures, runtime.replays) == (0, 0)
    compiled = run(True)
    assert (runtime.captures, runtime.replays) == (1, 2)
    assert all(mx.array_equal(a, b) for a, b in zip(eager[0], compiled[0]))
    assert all(mx.array_equal(a, b) for la, lb in zip(eager[1], compiled[1]) for a, b in zip(la, lb))


def test_install_runtime_modes(weights):
    assert RUNTIME_SPECS == {"ring-fused": RuntimeSpec(mx.bfloat16, 21), "apple": RuntimeSpec(mx.float16, 21),
                             "apple-fast": RuntimeSpec(mx.float16, 11)}
    model = _model(weights)
    gen = types.SimpleNamespace(model=model)
    for mode, spec in RUNTIME_SPECS.items():
        install_runtime(gen, mode)
        installed = model.castle_runtime
        assert isinstance(installed, CastleRuntime) and installed.mode == mode
        assert (installed.dtype, installed.window) == (spec.dtype, spec.window)
        assert installed.ring and installed.fused and installed.use_graph
    install_runtime(gen, "native")
    assert model.castle_runtime is None
    assert [dit_dtype(m) for m in ("native", "ring-fused", "apple", "apple-fast")] == [
        mx.bfloat16, mx.bfloat16, mx.float16, mx.float16]
    for mode in ("kernels", "cudagraph", "ring", "ring-graph", "rtx5090", "bogus"):
        with pytest.raises(ValueError, match="Unknown DiT runtime"):
            install_runtime(gen, mode)
        with pytest.raises(ValueError, match="Unknown DiT runtime"):
            dit_dtype(mode)
    for mode in ("ring", "native"):
        with pytest.raises(ValueError, match="Unknown castle runtime"):
            CastleRuntime(mode)


UNSUPPORTED = "Unsupported runtime: "
BATCH = "Castle runtime requires batch1/chunk1"
LAYOUT = "Expected identical bounded windows, one sink, and T2V cross attention"
FUSABLE = "Fused kernels expect bfloat16 modulation, an affine cross-attention LayerNorm and RMS q/k norms"


class OtherCrossAttention(WanT2VCrossAttention):
    pass


def _reject_case(weights, case):
    """(model, call kwargs) for one unsupported configuration."""
    rng = np.random.default_rng(3)
    x = [rng.standard_normal((1, 2, 4, 12)).astype(np.float32)]
    kw = dict(x=x, t=937.5, ctx=rng.standard_normal((1, 8, 32)).astype(np.float32), position=0, mode="fixed")
    model = _model(weights, runtime=True)
    kv, xc = _caches()
    if case == "batch2":
        kw["x"] = x * 2
        kv, xc = _caches(batch=2)
    elif case == "chunk2":
        kw["x"] = [rng.standard_normal((2, 2, 4, 12)).astype(np.float32)]
        kw["frames"] = 2
    elif case == "sink_mode":
        kw["mode"] = "other"
    elif case == "float32":
        model = _model(tree_flatten(tree_map(lambda a: a.astype(mx.float32), dict(weights))), runtime=True)
        kv, xc = _caches(dtype=mx.float32)
    elif case == "cross_attention":
        for block in model.blocks:
            old = block.cross_attn
            block.cross_attn = OtherCrossAttention(CFG["dim"], HEADS, (-1, -1), True, CFG["eps"])
            block.cross_attn.update(old.parameters())
    elif case == "window1":
        for block in model.blocks:
            block.self_attn.local_attn_size = 1
        kv, xc = _caches(window=1)
    elif case == "sink2":
        for block in model.blocks:
            block.self_attn.sink_size = 2
    elif case == "cache_start":
        kw["cache_start"] = FSL
    elif case == "norm3_identity":  # cross_attn_norm=False: no affine LayerNorm for gate_ln_affine
        model.blocks[-1].norm3 = nn.Identity()
    elif case == "modulation_f32":
        model.blocks[-1].modulation = model.blocks[-1].modulation.astype(mx.float32)
    elif case == "qk_norm_off":
        model.blocks[-1].cross_attn.norm_q = nn.Identity()
    return model, kw, kv, xc


@pytest.mark.parametrize("case,message", [
    ("batch2", BATCH), ("chunk2", BATCH), ("sink_mode", UNSUPPORTED), ("float32", UNSUPPORTED),
    ("cross_attention", LAYOUT), ("window1", LAYOUT), ("sink2", LAYOUT), ("cache_start", UNSUPPORTED),
    ("norm3_identity", FUSABLE), ("modulation_f32", FUSABLE), ("qk_norm_off", FUSABLE)])
def test_runtime_rejects_unsupported(weights, case, message):
    model, kw, kv, xc = _reject_case(weights, case)
    frames = kw.get("frames", 1)
    with pytest.raises(ValueError, match="^" + re.escape(message)):
        model([mx.array(u) for u in kw["x"]], t=mx.full((len(kw["x"]), frames), kw["t"]), context=mx.array(kw["ctx"]),
              seq_len=frames * FSL, kv_cache=kv, crossattn_cache=xc, current_start=kw["position"] * FSL,
              cache_start=kw.get("cache_start", 0), sink_rope_mode=kw["mode"])
    # No native fallback and no partial planning: nothing was written.
    assert all(c["global_end_index"] == 0 and c["local_end_index"] == 0 for c in kv)
    assert not any(c["is_init"] for c in xc)


def test_runtime_rejects_unsupported_messages(weights):
    model, kw, kv, xc = _reject_case(weights, "sink_mode")
    with pytest.raises(ValueError) as raised:
        _call(model, kw["x"], kw["t"], kw["ctx"], kv, xc, 0, "other")
    assert str(raised.value) == "Unsupported runtime: extra=False, sink=other, dtype=mlx.core.bfloat16, device=Device(gpu, 0)"


@pytest.mark.parametrize("runtime,model_dtype,cache_dtype,message", [
    ("apple", mx.bfloat16, mx.bfloat16,
     "Unsupported runtime: extra=False, sink=fixed, dtype=mlx.core.bfloat16, device=Device(gpu, 0)"),
    ("ring-fused", mx.float16, mx.float16,
     "Unsupported runtime: extra=False, sink=fixed, dtype=mlx.core.float16, device=Device(gpu, 0)"),
    ("apple", mx.float16, mx.bfloat16, "Expected float16 KV caches"),
    ("apple", "modulation_bf16", mx.float16,
     "Fused kernels expect float16 modulation, an affine cross-attention LayerNorm and RMS q/k norms")],
    ids=["apple-bf16-model", "ring-fused-fp16-model", "apple-bf16-caches", "apple-bf16-modulation"])
def test_runtime_rejects_dtype_mismatch(weights, runtime, model_dtype, cache_dtype, message):
    """Mixed dtypes are a ValueError on the first call (no silent fallback, no promoted float32 path), before any
    cache is planned or any text K/V initialised."""
    rng = np.random.default_rng(3)
    dtype = mx.float16 if model_dtype == "modulation_bf16" else model_dtype
    model = _model([(name, a.astype(dtype)) for name, a in weights], runtime=True, mode=runtime)
    if model_dtype == "modulation_bf16":
        model.blocks[-1].modulation = model.blocks[-1].modulation.astype(mx.bfloat16)
    kv, xc = _caches(dtype=cache_dtype)
    x = rng.standard_normal((1, 1, 2, 4, 12)).astype(np.float32)
    with pytest.raises(ValueError) as raised:
        _call(model, x, 937.5, rng.standard_normal((1, 8, 32)).astype(np.float32), kv, xc, 0, "fixed")
    assert str(raised.value) == message
    assert all(c["global_end_index"] == 0 and c["local_end_index"] == 0 for c in kv)
    assert not any(c["is_init"] for c in xc)


def _spy_rotations(monkeypatch):
    """Record the start frame of every build_causal_rope_rotations call the runtime makes."""
    from matrix_game_mlx.inference import rope_kernel
    starts, real = [], rope_kernel.build_causal_rope_rotations

    def spy(freqs, frames, h, w, start):
        starts.append(start)
        return real(freqs, frames, h, w, start)
    monkeypatch.setattr(rope_kernel, "build_causal_rope_rotations", spy)
    return starts


def _filled_caches(position, window, dtype=mx.bfloat16):
    """Caches as after `position` frames: a full window (position >= window) at the end of the ring layout."""
    kv, xc = _caches(window=window, dtype=dtype)
    for cache in kv:
        cache["global_end_index"] = position * FSL
        cache["local_end_index"] = min(position, window) * FSL
    return kv, xc


@pytest.mark.parametrize("runtime, window, position, sink", [("ring-fused", 21, 22, 33), ("apple-fast", 11, 12, 23)])
def test_future_sink_keys_land_at_position_plus_offset(weights, monkeypatch, runtime, window, position, sink):
    """Past the window the future sink's keys are rotated at position + 11 (the current block stays at position)."""
    starts = _spy_rotations(monkeypatch)
    weights, dtype = _runtime_weights(weights, runtime), RUNTIME_SPECS[runtime].dtype
    model = _model(weights, runtime=True, mode=runtime, window=window)
    assert model.castle_runtime.window == window or runtime == "ring-fused"
    kv, xc = _filled_caches(position, window, dtype)
    rng = np.random.default_rng(0)
    _call(model, _inputs(rng, 1)[0], 937.5, rng.standard_normal((1, 8, 32)).astype(np.float32), kv, xc, position,
          "future")
    assert starts == [position, sink]


@pytest.mark.parametrize("position", [0, 21])  # position <= window
def test_future_sink_keeps_cached_keys_before_the_window_is_full(weights, monkeypatch, position):
    starts = _spy_rotations(monkeypatch)
    model = _model(weights, runtime=True, window=21)
    kv, xc = _filled_caches(position, 21)
    rng = np.random.default_rng(0)
    _call(model, _inputs(rng, 1)[0], 937.5, rng.standard_normal((1, 8, 32)).astype(np.float32), kv, xc, position,
          "future")
    assert starts == [position]


@pytest.mark.parametrize("runtime", ["native", "ring-fused", "apple", "apple-fast"])
def test_future_sink_exhausted_rope_table_raises(weights, runtime):
    """A table of R rows: position R - 5 fits (position + 1 <= R) but the sink at position + 11 does not."""
    rows = 100
    window = RUNTIME_SPECS[runtime].window if runtime == "apple-fast" else 21
    model = _model(_runtime_weights(weights, runtime) if runtime != "native" else weights,
                   runtime=runtime != "native", mode=runtime if runtime != "native" else "ring-fused", window=window)
    model._freqs, model._rope_rows = rope_freqs(rows, model.dim // model.num_heads), rows
    position = rows - 5
    assert position > window
    dtype = RUNTIME_SPECS[runtime].dtype if runtime != "native" else mx.bfloat16
    kv, xc = _filled_caches(position, window, dtype)
    rng = np.random.default_rng(0)
    for mode in ("absolute", "fixed"):   # no sink past the table: these still run
        _call(model, _inputs(rng, 1)[0], 937.5, rng.standard_normal((1, 8, 32)).astype(np.float32), kv, xc, position,
              mode)
    with pytest.raises(ValueError, match="RoPE table has"):
        _call(model, _inputs(rng, 1)[0], 937.5, rng.standard_normal((1, 8, 32)).astype(np.float32), kv, xc, position,
              "future")


def test_graph_compiled_under_fixed_replays_bitwise_under_future(weights):
    """The compiled graph is shared across sink modes: the sink rotations are graph inputs, so a graph traced under
    `fixed` replays under `future` bitwise equal to the eager `future` run, and nothing is re-traced."""
    rng = np.random.default_rng(9)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    position = WINDOW + 3
    x, x2 = _inputs(rng, 2)

    def state(model, mode, kv, xc):
        out = _call(model, x, 937.5, ctx, kv, xc, position, mode)
        return out, [{n: mx.array(c[n]) for n in ("k", "v", "k_sink_raw")} for c in kv]

    def filled(model):
        kv, xc = _caches()
        for i, cache in enumerate(kv):
            cache["k"] = mx.array(np.random.default_rng(i).standard_normal(cache["k"].shape)).astype(mx.bfloat16)
            cache["v"] = mx.array(np.random.default_rng(10 + i).standard_normal(cache["v"].shape)).astype(mx.bfloat16)
            cache["k_sink_raw"] = mx.array(np.random.default_rng(20 + i).standard_normal(cache["k_sink_raw"].shape)
                                           ).astype(mx.bfloat16)
            cache["global_end_index"], cache["local_end_index"] = position * FSL, WINDOW * FSL
        return kv, xc

    compiled, eager = _model(weights, runtime=True), _model(weights, runtime=True)
    eager.castle_runtime.use_graph = False
    kv, xc = filled(compiled)
    _call(compiled, x2, 937.5, ctx, kv, xc, position, "fixed")        # traces the graph under fixed
    assert compiled.castle_runtime.captures == 1
    kv_c, xc_c = filled(compiled)
    kv_e, xc_e = filled(eager)
    out_c, st_c = state(compiled, "future", kv_c, xc_c)
    out_e, st_e = state(eager, "future", kv_e, xc_e)
    assert compiled.castle_runtime.captures == 1 and compiled.castle_runtime.replays == 2
    assert mx.array_equal(out_c, out_e)
    for ca, cb in zip(st_c, st_e):
        for name in ("k", "v", "k_sink_raw"):
            assert mx.array_equal(ca[name], cb[name])
