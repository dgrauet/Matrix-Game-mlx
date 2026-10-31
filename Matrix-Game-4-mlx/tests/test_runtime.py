"""Ring-fused runtime (`CastleRuntime`, unfused blocks) vs the native MLX block loop on a bf16 tiny model.

The runtime may differ from native only through the attention key order (ring KV -> SDPA summation order); RoPE,
norms, modulation, cross-attention and FFN round exactly as native. Teacher-forced checks reset the runtime's cache to
native's (ring order via `from_linear`) before every call, so rows the call does not write must match bitwise and only
the written slot and the output carry the summation-order deviation. Free-running checks bound the drift over a rollout.
"""
import re
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from matrix_game_mlx.inference.runtime import CastleRuntime, from_linear, install_runtime, to_linear
from matrix_game_mlx.inference.wan.modules.causal_model import CausalWanModel
from matrix_game_mlx.inference.wan.modules.model import WanT2VCrossAttention
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


def _model(weights, runtime=False):
    model = CausalWanModel(**CFG)
    model.load_weights(weights, strict=True)
    model.set_rope_max_len(8192)
    if runtime:
        install_runtime(types.SimpleNamespace(model=model), "ring-fused")
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


def _teacher_forced(weights, mode, *, calls, switch_at=None):
    """Run native and runtime per call from the same (native) cache state; assert per-call equivalence."""
    rng = np.random.default_rng(0)
    contexts = [rng.standard_normal((1, 8, 32)).astype(np.float32) for _ in range(2)]
    native, runtime = _model(weights), _model(weights, runtime=True)
    kv_n, xc_n = _caches()
    _, xc_r = _caches()
    worst = [0.0, 0.0]
    for index, ((position, t), x) in enumerate(zip(calls, _inputs(rng, len(calls)))):
        ctx = contexts[int(switch_at is not None and position >= switch_at)]
        if switch_at is not None and position == switch_at and t != 0.0:
            for cross in xc_n + xc_r:           # as CastleStream.set_prompt
                cross["is_init"] = False
        kv_r = [from_linear(_clone(c), FSL, WINDOW) for c in kv_n]
        out_r = _call(runtime, x, t, ctx, kv_r, xc_r, position, mode)
        out_n = _call(native, x, t, ctx, kv_n, xc_n, position, mode)
        msg = f"call {index} position {position} t={t}"
        err = rel_l2(out_r, out_n)
        assert err < TOL, msg
        worst[0] = max(worst[0], err)
        for layer, (cr, cn) in enumerate(zip(kv_r, kv_n)):
            lin = to_linear(cr, FSL, WINDOW)
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
    return worst


@pytest.mark.parametrize("mode", ["absolute", "fixed"])
def test_runtime_matches_native(weights, mode):
    MEASURED[f"teacher_forced[{mode}]"] = _teacher_forced(weights, mode, calls=_schedule(CHUNKS))
    print("max rel L2 (output, written slot):", MEASURED)


def test_runtime_prompt_switch_matches_native(weights):
    MEASURED["prompt_switch"] = _teacher_forced(weights, "fixed", calls=_schedule(CHUNKS), switch_at=10)
    print("max rel L2 (output, written slot):", MEASURED)


@pytest.mark.parametrize("mode", ["absolute", "fixed"])
def test_runtime_clean_first_call_empty_cache(weights, mode):
    MEASURED[f"clean_first[{mode}]"] = _teacher_forced(weights, mode, calls=_schedule(CHUNKS, clean_first=True))
    print("max rel L2 (output, written slot):", MEASURED)


@pytest.mark.parametrize("mode", ["absolute", "fixed"])
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


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
@pytest.mark.parametrize("mode", ["absolute", "fixed"])
def test_runtime_linear_layout_is_bitwise_native(weights, mode, fused):
    """With the ring disabled (native linear slot order) the runtime is bitwise native: GPU RoPE, the composed
    block pieces (or the fused kernels, whose row reductions follow MLX's order at these widths), cross-attention
    and the fixed-sink key set round exactly as native; only the ring order differs."""
    rng = np.random.default_rng(4)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    native, runtime = _model(weights), _model(weights, runtime=True)
    runtime.castle_runtime.ring = False
    runtime.castle_runtime.fused = fused
    (kv_n, xc_n), (kv_r, xc_r) = _caches(), _caches()
    calls = _schedule(CHUNKS, clean_first=True)
    for index, ((position, t), x) in enumerate(zip(calls, _inputs(rng, len(calls)))):
        assert mx.array_equal(_call(runtime, x, t, ctx, kv_r, xc_r, position, mode),
                              _call(native, x, t, ctx, kv_n, xc_n, position, mode)), f"call {index}"
        for cr, cn in zip(kv_r, kv_n):
            for name in ("k", "v", "k_sink_raw"):
                assert mx.array_equal(cr[name], cn[name]), f"call {index} {name}"
    assert runtime.castle_runtime.calls == len(calls)


@pytest.mark.parametrize("mode", ["absolute", "fixed"])
def test_runtime_fused_is_bitwise_unfused(weights, mode, monkeypatch):
    """ring-fused runs the fused kernels by default, and they round exactly as the composed native pieces: a
    60-chunk free-running rollout (clean-first, ring layout) is bitwise equal with fused on and off."""
    from matrix_game_mlx.inference import kernels
    counts = {}
    for name in ("ln_modulate", "gate_ln_affine", "gate_residual", "rms_norm", "gelu_tanh"):
        def counted(*args, _f=getattr(kernels, name), _n=name, **kwargs):
            counts[_n] = counts.get(_n, 0) + 1
            return _f(*args, **kwargs)
        monkeypatch.setattr(kernels, name, counted)
    rng = np.random.default_rng(5)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    fused, unfused = _model(weights, runtime=True), _model(weights, runtime=True)
    assert fused.castle_runtime.fused
    unfused.castle_runtime.fused = False
    (kv_f, xc_f), (kv_u, xc_u) = _caches(), _caches()
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


def test_runtime_second_stream_matches_fresh(weights):
    rng = np.random.default_rng(2)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    first, calls = _inputs(rng, 12), _schedule(10)
    inputs = _inputs(rng, len(calls))
    used = _model(weights, runtime=True)
    kv, xc = _caches()
    for (position, t), x in zip(_schedule(6), first):
        _call(used, x, t, ctx, kv, xc, position, "fixed")
    fresh = _model(weights, runtime=True)
    (kv_u, xc_u), (kv_f, xc_f) = _caches(), _caches()
    for (position, t), x in zip(calls, inputs):
        a = _call(used, x, t, ctx, kv_u, xc_u, position, "fixed")
        b = _call(fresh, x, t, ctx, kv_f, xc_f, position, "fixed")
        assert mx.array_equal(a, b)
        for cu, cf in zip(kv_u, kv_f):
            for name in ("k", "v", "k_sink_raw"):
                assert mx.array_equal(cu[name], cf[name])


def test_install_runtime_modes(weights):
    model = _model(weights)
    gen = types.SimpleNamespace(model=model)
    install_runtime(gen, "ring-fused")
    assert isinstance(model.castle_runtime, CastleRuntime) and model.castle_runtime.mode == "ring-fused"
    assert model.castle_runtime.ring
    install_runtime(gen, "native")
    assert model.castle_runtime is None
    for mode in ("kernels", "cudagraph", "ring", "ring-graph", "rtx5090", "bogus"):
        with pytest.raises(ValueError, match="Unknown DiT runtime"):
            install_runtime(gen, mode)
    with pytest.raises(ValueError, match="Unknown castle runtime"):
        CastleRuntime("ring")


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
