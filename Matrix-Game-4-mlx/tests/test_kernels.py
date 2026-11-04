"""Fused block-glue kernels (kernels.py) vs the native composition of `CausalWanAttentionBlock`.

The references below are the native expressions: `WanLayerNorm` / `WanRMSNorm` modules, the modulation
`mx.split(modulation[:, None] + e, 6, axis=2)`, and the block's `mod()` / `gate()` reshapes. Rounding rule
(upstream kernels.py): every intermediate the separate operations materialise in bf16 is rounded to bf16 in the
kernel too, reductions accumulate in fp32, and only the summation order inside a row reduction may differ. So the
purely elementwise kernels (`gate_residual`, `gelu_tanh`) must be bitwise; the row-reduction kernels must be within
1 bf16 ulp everywhere and bitwise on at least 99 % of the elements (measured share printed with -s).
The `*_fp16` tests hold the fp16 variants to the same contract, measured in fp16 ulps.
"""
import mlx.core as mx
import numpy as np
import pytest

from matrix_game_mlx.inference.kernels import gate_ln_affine, gate_residual, gelu_tanh, ln_modulate, rms_norm
from matrix_game_mlx.inference.wan.modules.model import WanLayerNorm, WanRMSNorm, gelu_tanh as native_gelu_tanh

EPS = 1e-6
# (S, C, frames): the realtime chunk (405 tokens, dim 3072), the tiny test model's width, and a two-frame grid.
SHAPES = [(405, 3072, 1), (7, 128, 1), (10, 128, 2)]
BITWISE_SHARE = {}  # kernel[shape] -> share of bitwise-equal elements (ledger)
MIN_SHARE = 0.99


def _bits(a):
    assert a.dtype in (mx.bfloat16, mx.float16)
    return np.array(a.view(mx.uint16)).astype(np.int32)


def _ordered(bits):
    # bf16 / fp16 bit patterns -> integers whose difference counts ulps (sign-magnitude to two's-complement ordering).
    return np.where(bits & 0x8000, -(bits & 0x7FFF), bits)


def _check_close(name, out, ref):
    """Within 1 bf16 ulp elementwise and bitwise on >= MIN_SHARE of the elements; records the share."""
    assert out.dtype == ref.dtype and out.dtype in (mx.bfloat16, mx.float16) and out.shape == ref.shape
    a, b = _bits(out), _bits(ref)
    ulps = np.abs(_ordered(a) - _ordered(b))
    share = float(np.mean(a == b))
    BITWISE_SHARE[name] = share
    print(f"{name}: bitwise share {share:.6f}, max ulp {int(ulps.max())}")
    assert ulps.max() <= 1, f"{name}: {int(np.sum(ulps > 1))} elements differ by more than 1 ulp"
    assert share >= MIN_SHARE, f"{name}: bitwise share {share}"


def _check_bitwise(name, out, ref):
    assert out.dtype == ref.dtype and out.dtype in (mx.bfloat16, mx.float16) and out.shape == ref.shape
    a, b = _bits(out), _bits(ref)
    BITWISE_SHARE[name] = float(np.mean(a == b))
    assert np.array_equal(a, b), f"{name}: {int(np.sum(a != b))} mismatches"


def _rows(rng, s, c, scale=1.0):
    """Random bf16 [1, S, C] with a constant row (variance 0) and a row of large values (~1e3)."""
    x = rng.standard_normal((1, s, c)).astype(np.float32) * scale
    x[0, 0] = 0.75
    x[0, min(1, s - 1)] = 1e3 * (1 + 0.01 * rng.standard_normal(c))
    return mx.array(x).astype(mx.bfloat16)


def _mod(rng, frames, c):
    e = mx.array(rng.standard_normal((1, frames, 6, c)).astype(np.float32) * 0.5).astype(mx.bfloat16)
    modulation = mx.array(rng.standard_normal((1, 6, c)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    return e, modulation


def _rows16(rng, s, c, scale=1.0):
    return _rows(rng, s, c, scale).astype(mx.float16)  # same values; fp16 holds the 1e3 row


def _mod16(rng, frames, c):
    e, modulation = _mod(rng, frames, c)
    return e.astype(mx.float16), modulation.astype(mx.float16)


def _f16(rng, shape, scale=1.0, offset=0.0):
    return mx.array(offset + rng.standard_normal(shape).astype(np.float32) * scale).astype(mx.float16)


# --- native compositions (CausalWanAttentionBlock.__call__) -------------------------------------------------------

def _native_mod(h, shift, scale, frames):
    tokens = h.shape[1] // frames
    return (h.reshape(h.shape[0], frames, tokens, -1) * (1 + scale) + shift).reshape(h.shape)


def _native_gate(h, g, frames):
    tokens = h.shape[1] // frames
    return (h.reshape(h.shape[0], frames, tokens, -1) * g).reshape(h.shape)


def _native_ln_modulate(x, e, modulation, eps, shift_row, scale_row, residual=None):
    frames = e.shape[1]
    if residual is not None:
        x = x + residual
    mod = mx.split(modulation[:, None] + e, 6, axis=2)
    return _native_mod(WanLayerNorm(x.shape[-1], eps)(x), mod[shift_row], mod[scale_row], frames), x


def _native_gate_ln_affine(x, y, e, modulation, gate_row, weight, bias, eps):
    mod = mx.split(modulation[:, None] + e, 6, axis=2)
    x = x + _native_gate(y, mod[gate_row], e.shape[1])
    norm = WanLayerNorm(x.shape[-1], eps, elementwise_affine=True)
    norm.weight, norm.bias = weight, bias
    return norm(x), x


def _native_gate_residual(x, y, e, modulation, gate_row):
    mod = mx.split(modulation[:, None] + e, 6, axis=2)
    return x + _native_gate(y, mod[gate_row], e.shape[1])


def _native_rms_norm(x, weight, eps):
    norm = WanRMSNorm(x.shape[-1], eps)
    norm.weight = weight
    return norm(x)


# --- equivalence ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("s,c,frames", SHAPES)
@pytest.mark.parametrize("rows", [(0, 1), (3, 4)])
def test_ln_modulate_matches_native(s, c, frames, rows):
    rng = np.random.default_rng(s * 31 + c + rows[0])
    x = _rows(rng, s, c)
    e, modulation = _mod(rng, frames, c)
    z, x_out = ln_modulate(x, e, modulation, EPS, *rows)
    z_ref, _ = _native_ln_modulate(x, e, modulation, EPS, *rows)
    assert x_out is x
    _check_close(f"ln_modulate{rows}[{s}x{c}x{frames}]", z, z_ref)


@pytest.mark.parametrize("s,c,frames", SHAPES)
def test_ln_modulate_residual_matches_native(s, c, frames):
    rng = np.random.default_rng(s * 37 + c)
    x = _rows(rng, s, c)
    residual = mx.array(rng.standard_normal((1, s, c)).astype(np.float32) * 0.3).astype(mx.bfloat16)
    residual[0, 0] = 0  # keeps x + residual constant on row 0 (variance 0)
    e, modulation = _mod(rng, frames, c)
    z, x_out = ln_modulate(x, e, modulation, EPS, 3, 4, residual=residual)
    z_ref, x_ref = _native_ln_modulate(x, e, modulation, EPS, 3, 4, residual=residual)
    _check_bitwise(f"ln_modulate.residual_sum[{s}x{c}x{frames}]", x_out, x_ref)
    _check_close(f"ln_modulate.residual[{s}x{c}x{frames}]", z, z_ref)


@pytest.mark.parametrize("s,c,frames", SHAPES)
def test_gate_ln_affine_matches_native(s, c, frames):
    rng = np.random.default_rng(s * 41 + c)
    x = _rows(rng, s, c)
    y = mx.array(rng.standard_normal((1, s, c)).astype(np.float32)).astype(mx.bfloat16)
    y[0, 0] = 0  # row 0 of x + y * gate stays constant
    e, modulation = _mod(rng, frames, c)
    weight = mx.array(1 + 0.1 * rng.standard_normal(c).astype(np.float32)).astype(mx.bfloat16)
    bias = mx.array(0.1 * rng.standard_normal(c).astype(np.float32)).astype(mx.bfloat16)
    z, x_out = gate_ln_affine(x, y, e, modulation, 2, weight, bias, EPS)
    z_ref, x_ref = _native_gate_ln_affine(x, y, e, modulation, 2, weight, bias, EPS)
    _check_bitwise(f"gate_ln_affine.residual_sum[{s}x{c}x{frames}]", x_out, x_ref)
    _check_close(f"gate_ln_affine[{s}x{c}x{frames}]", z, z_ref)


@pytest.mark.parametrize("s,c,frames", SHAPES)
@pytest.mark.parametrize("row", [2, 5])
def test_gate_residual_is_bitwise_native(s, c, frames, row):
    rng = np.random.default_rng(s * 43 + c + row)
    x = _rows(rng, s, c)
    y = mx.array(rng.standard_normal((1, s, c)).astype(np.float32) * 4).astype(mx.bfloat16)
    e, modulation = _mod(rng, frames, c)
    _check_bitwise(f"gate_residual[{s}x{c}x{frames}]", gate_residual(x, y, e, modulation, row),
                   _native_gate_residual(x, y, e, modulation, row))


@pytest.mark.parametrize("s,c,frames", SHAPES)
def test_rms_norm_matches_native(s, c, frames):
    rng = np.random.default_rng(s * 47 + c)
    x = _rows(rng, s, c, scale=3.0)
    x[0, 0] = 0  # all-zero row: rsqrt(eps) * 0
    weight = mx.array(1 + 0.2 * rng.standard_normal(c).astype(np.float32)).astype(mx.bfloat16)
    _check_close(f"rms_norm[{s}x{c}]", rms_norm(x, weight, EPS), _native_rms_norm(x, weight, EPS))


@pytest.mark.parametrize("s,c", [(405, 14336), (7, 256)])
def test_gelu_tanh_is_bitwise_native(s, c):
    rng = np.random.default_rng(c)
    x = mx.array(rng.standard_normal((1, s, c)).astype(np.float32) * 3).astype(mx.bfloat16)
    x[0, 0, :4] = mx.array([0.0, -0.0, 20.0, -20.0]).astype(mx.bfloat16)
    _check_bitwise(f"gelu_tanh[{s}x{c}]", gelu_tanh(x), native_gelu_tanh(x))


@pytest.mark.parametrize("c", [3072, 128])
@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_row_kernels_bitwise_at_model_widths(c, scale):
    """At the model widths the kernels' row sums use MLX's own contiguous row-reduction layout (`row_reduce_simple`:
    c / 4 threads, 4 consecutive values each), so the statistics and hence every output are bitwise native."""
    for seed in range(3):
        rng = np.random.default_rng(1000 + seed)
        x = _rows(rng, 405, c, scale)
        y = mx.array(rng.standard_normal((1, 405, c)).astype(np.float32) * scale).astype(mx.bfloat16)
        e, modulation = _mod(rng, 1, c)
        w = mx.array(1 + 0.1 * rng.standard_normal(c).astype(np.float32)).astype(mx.bfloat16)
        bias = mx.array(0.1 * rng.standard_normal(c).astype(np.float32)).astype(mx.bfloat16)
        tag = f"[{c} scale {scale} seed {seed}]"
        _check_bitwise("ln_modulate" + tag, ln_modulate(x, e, modulation, EPS, 0, 1)[0],
                       _native_ln_modulate(x, e, modulation, EPS, 0, 1)[0])
        _check_bitwise("ln_modulate.residual" + tag, ln_modulate(x, e, modulation, EPS, 3, 4, residual=y)[0],
                       _native_ln_modulate(x, e, modulation, EPS, 3, 4, residual=y)[0])
        _check_bitwise("gate_ln_affine" + tag, gate_ln_affine(x, y, e, modulation, 2, w, bias, EPS)[0],
                       _native_gate_ln_affine(x, y, e, modulation, 2, w, bias, EPS)[0])
        _check_bitwise("rms_norm" + tag, rms_norm(x, w, EPS), _native_rms_norm(x, w, EPS))


# --- fp16 variants (same contract, fp16 ulps) -----------------------------------------------------------------

@pytest.mark.parametrize("s,c,frames", SHAPES)
@pytest.mark.parametrize("rows", [(0, 1), (3, 4)])
def test_ln_modulate_matches_native_fp16(s, c, frames, rows):
    rng = np.random.default_rng(s * 31 + c + rows[0])
    x = _rows16(rng, s, c)
    e, modulation = _mod16(rng, frames, c)
    z, x_out = ln_modulate(x, e, modulation, EPS, *rows)
    z_ref, _ = _native_ln_modulate(x, e, modulation, EPS, *rows)
    assert x_out is x
    _check_close(f"ln_modulate.fp16{rows}[{s}x{c}x{frames}]", z, z_ref)


@pytest.mark.parametrize("s,c,frames", SHAPES)
def test_ln_modulate_residual_matches_native_fp16(s, c, frames):
    rng = np.random.default_rng(s * 37 + c)
    x = _rows16(rng, s, c)
    residual = _f16(rng, (1, s, c), 0.3)
    residual[0, 0] = 0
    e, modulation = _mod16(rng, frames, c)
    z, x_out = ln_modulate(x, e, modulation, EPS, 3, 4, residual=residual)
    z_ref, x_ref = _native_ln_modulate(x, e, modulation, EPS, 3, 4, residual=residual)
    _check_bitwise(f"ln_modulate.residual_sum.fp16[{s}x{c}x{frames}]", x_out, x_ref)
    _check_close(f"ln_modulate.residual.fp16[{s}x{c}x{frames}]", z, z_ref)


@pytest.mark.parametrize("s,c,frames", SHAPES)
def test_gate_ln_affine_matches_native_fp16(s, c, frames):
    rng = np.random.default_rng(s * 41 + c)
    x = _rows16(rng, s, c)
    y = _f16(rng, (1, s, c))
    y[0, 0] = 0
    e, modulation = _mod16(rng, frames, c)
    weight, bias = _f16(rng, (c,), 0.1, 1.0), _f16(rng, (c,), 0.1)
    z, x_out = gate_ln_affine(x, y, e, modulation, 2, weight, bias, EPS)
    z_ref, x_ref = _native_gate_ln_affine(x, y, e, modulation, 2, weight, bias, EPS)
    _check_bitwise(f"gate_ln_affine.residual_sum.fp16[{s}x{c}x{frames}]", x_out, x_ref)
    _check_close(f"gate_ln_affine.fp16[{s}x{c}x{frames}]", z, z_ref)


@pytest.mark.parametrize("s,c,frames", SHAPES)
@pytest.mark.parametrize("row", [2, 5])
def test_gate_residual_is_bitwise_native_fp16(s, c, frames, row):
    rng = np.random.default_rng(s * 43 + c + row)
    x = _rows16(rng, s, c)
    y = _f16(rng, (1, s, c), 4.0)
    e, modulation = _mod16(rng, frames, c)
    _check_bitwise(f"gate_residual.fp16[{s}x{c}x{frames}]", gate_residual(x, y, e, modulation, row),
                   _native_gate_residual(x, y, e, modulation, row))


@pytest.mark.parametrize("s,c,frames", SHAPES)
def test_rms_norm_matches_native_fp16(s, c, frames):
    rng = np.random.default_rng(s * 47 + c)
    x = _rows16(rng, s, c, scale=3.0)
    x[0, 0] = 0
    weight = _f16(rng, (c,), 0.2, 1.0)
    _check_close(f"rms_norm.fp16[{s}x{c}]", rms_norm(x, weight, EPS), _native_rms_norm(x, weight, EPS))


@pytest.mark.parametrize("s,c", [(405, 14336), (7, 256)])
def test_gelu_tanh_is_bitwise_native_fp16(s, c):
    rng = np.random.default_rng(c)
    x = _f16(rng, (1, s, c), 3.0)
    x[0, 0, :4] = mx.array([0.0, -0.0, 20.0, -20.0]).astype(mx.float16)
    _check_bitwise(f"gelu_tanh.fp16[{s}x{c}]", gelu_tanh(x), native_gelu_tanh(x))


@pytest.mark.parametrize("c", [3072, 128])
@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_row_kernels_bitwise_at_model_widths_fp16(c, scale):
    for seed in range(3):
        rng = np.random.default_rng(1000 + seed)
        x = _rows16(rng, 405, c, scale)
        y = _f16(rng, (1, 405, c), scale)
        e, modulation = _mod16(rng, 1, c)
        w, bias = _f16(rng, (c,), 0.1, 1.0), _f16(rng, (c,), 0.1)
        tag = f".fp16[{c} scale {scale} seed {seed}]"
        _check_bitwise("ln_modulate" + tag, ln_modulate(x, e, modulation, EPS, 0, 1)[0],
                       _native_ln_modulate(x, e, modulation, EPS, 0, 1)[0])
        _check_bitwise("ln_modulate.residual" + tag, ln_modulate(x, e, modulation, EPS, 3, 4, residual=y)[0],
                       _native_ln_modulate(x, e, modulation, EPS, 3, 4, residual=y)[0])
        _check_bitwise("gate_ln_affine" + tag, gate_ln_affine(x, y, e, modulation, 2, w, bias, EPS)[0],
                       _native_gate_ln_affine(x, y, e, modulation, 2, w, bias, EPS)[0])
        _check_bitwise("rms_norm" + tag, rms_norm(x, w, EPS), _native_rms_norm(x, w, EPS))


def test_bitwise_shares_reported():
    """Ledger line (run with -s): measured bitwise share per kernel and shape."""
    print({k: round(v, 6) for k, v in sorted(BITWISE_SHARE.items())})


# --- validation -----------------------------------------------------------------------------------------------

def _good(s=7, c=128, frames=1):
    rng = np.random.default_rng(0)
    x = _rows(rng, s, c)
    e, modulation = _mod(rng, frames, c)
    w = mx.ones((c,), mx.bfloat16)
    return x, mx.array(x), e, modulation, w, mx.zeros((c,), mx.bfloat16)


def test_ln_modulate_rejects_bad_inputs():
    x, y, e, mod, _, _ = _good()
    bad = [
        dict(x=x.astype(mx.float32)), dict(x=x[0]), dict(e=e.astype(mx.float32)), dict(e=e[:, :, :5]),
        dict(e=e[0]), dict(modulation=mod.astype(mx.float32)), dict(modulation=mod[:, :5]),
        dict(e=mx.zeros((1, 2, 6, 128), mx.bfloat16)),  # 7 tokens do not divide into 2 frames
        dict(e=mx.zeros((2, 1, 6, 128), mx.bfloat16)),  # batch mismatch
        dict(residual=y[:, :5]), dict(residual=y.astype(mx.float32)), dict(shift_row=6), dict(scale_row=-1),
    ]
    for override in bad:
        kw = dict(x=x, e=e, modulation=mod, eps=EPS, shift_row=0, scale_row=1, residual=None) | override
        with pytest.raises(ValueError):
            ln_modulate(**kw)


def test_gate_kernels_reject_bad_inputs():
    x, y, e, mod, w, b = _good()
    for override in (dict(y=y[:, :5]), dict(y=y.astype(mx.float32)), dict(x=x.astype(mx.float32)),
                     dict(e=e[:, :, :5]), dict(gate_row=6)):
        kw = dict(x=x, y=y, e=e, modulation=mod, gate_row=5) | override
        with pytest.raises(ValueError):
            gate_residual(**kw)
    for override in (dict(y=y[:, :5]), dict(weight=w[:5]), dict(bias=b.astype(mx.float32)),
                     dict(weight=w.reshape(1, -1)), dict(modulation=mod[0]), dict(gate_row=7)):
        kw = dict(x=x, y=y, e=e, modulation=mod, gate_row=2, weight=w, bias=b, eps=EPS) | override
        with pytest.raises(ValueError):
            gate_ln_affine(**kw)


def test_norm_and_gelu_reject_bad_inputs():
    x, _, _, _, w, _ = _good()
    for args in ((x.astype(mx.float32), w), (x[0], w), (x, w[:5]), (x, w.astype(mx.float32))):
        with pytest.raises(ValueError):
            rms_norm(*args, EPS)
    with pytest.raises(ValueError):
        gelu_tanh(x.astype(mx.float32))
    assert gelu_tanh(x.astype(mx.float16)).dtype == mx.float16


def test_kernels_reject_mixed_dtypes():
    """All arrays of one call share one dtype: an fp16 x with any bf16 companion (and the reverse) is rejected."""
    x, y, e, mod, w, b = _good()
    x, y, e, mod, w, b = (a.astype(mx.float16) for a in (x, y, e, mod, w, b))
    bf = lambda a: a.astype(mx.bfloat16)  # noqa: E731
    good = dict(x=x, e=e, modulation=mod, eps=EPS, shift_row=0, scale_row=1, residual=None)
    ln_modulate(**good)
    for override in (dict(e=bf(e)), dict(modulation=bf(mod)), dict(residual=bf(y))):
        with pytest.raises(ValueError):
            ln_modulate(**(good | override))
    with pytest.raises(ValueError):
        ln_modulate(**(good | dict(x=bf(x), residual=y)))
    good = dict(x=x, y=y, e=e, modulation=mod, gate_row=2, weight=w, bias=b, eps=EPS)
    gate_ln_affine(**good)
    for override in (dict(e=bf(e)), dict(modulation=bf(mod)), dict(y=bf(y)), dict(weight=bf(w)), dict(bias=bf(b))):
        with pytest.raises(ValueError):
            gate_ln_affine(**(good | override))
    good = dict(x=x, y=y, e=e, modulation=mod, gate_row=5)
    gate_residual(**good)
    for override in (dict(e=bf(e)), dict(modulation=bf(mod)), dict(y=bf(y))):
        with pytest.raises(ValueError):
            gate_residual(**(good | override))
    rms_norm(x, w, EPS)
    with pytest.raises(ValueError):
        rms_norm(x, bf(w), EPS)
    with pytest.raises(ValueError):
        rms_norm(bf(x), w, EPS)
    with pytest.raises(ValueError):
        ln_modulate(**(dict(x=x.astype(mx.float32), e=e, modulation=mod, eps=EPS, shift_row=0, scale_row=1)))
