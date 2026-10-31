"""GPU RoPE kernel (rope_kernel.py) vs the float64 CPU reference (causal_rope_apply / rope_apply_frames).

Every comparison is on raw bf16 bits: the kernel must be bitwise identical to the CPU path, which multiplies
in float64 (each product rounded to float64), subtracts/adds in float64, then casts float64 -> bfloat16
(MLX's CPU cast goes through float32: two round-to-nearest-even steps).
"""
import math

import numpy as np
import pytest
import mlx.core as mx

from matrix_game_mlx.inference.rope_kernel import (
    RopeRotations, _float64_lanes_bits, apply_causal_rope, build_causal_rope_rotations, build_rope_rotations_frames)
from matrix_game_mlx.inference.wan.modules.causal_model import CausalWanModel, causal_rope_apply, rope_apply_frames
from matrix_game_mlx.inference.wan.modules.model import _apply_rope_rotation, rope_freqs

SHAPE = (1, 405, 24, 128)  # realtime chunk: grid [[1, 15, 27]], 24 heads, head_dim 128
GRID = [[1, 15, 27]]
F, H, W = GRID[0]


@pytest.fixture(scope="module")
def freqs():
    return rope_freqs(8192, 128)


def _bits(a: mx.array) -> np.ndarray:
    assert a.dtype == mx.bfloat16
    return np.array(a.view(mx.uint16))


def _assert_bitwise(out: mx.array, ref: mx.array) -> None:
    assert out.dtype == mx.bfloat16 and out.shape == ref.shape
    got, want = _bits(out), _bits(ref.astype(mx.bfloat16))
    bad = np.flatnonzero(got != want)[:5]
    assert bad.size == 0, (f"{np.sum(got != want)} mismatches, first at flat {bad.tolist()}: "
                           f"got {[hex(v) for v in got.ravel()[bad]]} want {[hex(v) for v in want.ravel()[bad]]}")


def _from_bits(bits: np.ndarray) -> mx.array:
    return mx.array(bits.astype(np.uint16).reshape(SHAPE)).view(mx.bfloat16)


def _finite_bf16_patterns() -> np.ndarray:
    p = np.arange(65536, dtype=np.uint32)
    return p[((p >> 7) & 0xFF) != 0xFF].astype(np.uint16)  # 65024 patterns: ±0, subnormals, normals, ±max


def test_rotations_validate_like_upstream(freqs):
    with pytest.raises(ValueError, match="RoPE grid dimensions must be positive and start non-negative"):
        build_causal_rope_rotations(freqs, 1, 15, 27, start_frame=-1)
    for dims in ((0, 15, 27), (1, 0, 27), (1, 15, 0)):
        with pytest.raises(ValueError, match="RoPE grid dimensions must be positive and start non-negative"):
            build_causal_rope_rotations(freqs, *dims)
    with pytest.raises(ValueError, match="temporal RoPE position exceeds the precomputed frequency table"):
        build_causal_rope_rotations(freqs, 1, 15, 27, start_frame=8192)
    with pytest.raises(ValueError, match="temporal RoPE position exceeds the precomputed frequency table"):
        build_causal_rope_rotations(freqs, 2, 15, 27, start_frame=8191)
    small = rope_freqs(16, 128)
    with pytest.raises(ValueError, match="spatial RoPE grid exceeds the precomputed frequency table"):
        build_causal_rope_rotations(small, 1, 15, 27)
    with pytest.raises(ValueError, match="freqs must be"):
        build_causal_rope_rotations((mx.zeros(freqs[0].shape), freqs[1]), 1, 15, 27)
    with pytest.raises(ValueError, match="frame_pos must not be empty"):
        build_rope_rotations_frames(freqs, [], 15, 27)
    with pytest.raises(NotImplementedError):
        build_rope_rotations_frames(freqs, [-1], 15, 27)
    with pytest.raises(ValueError, match="temporal RoPE position exceeds the precomputed frequency table"):
        build_rope_rotations_frames(freqs, [8192], 15, 27)
    rot = build_causal_rope_rotations(freqs, 1, 15, 27, start_frame=3)
    assert isinstance(rot, RopeRotations) and rot.seq_len == 405 and rot.pairs == 64
    x = mx.zeros(SHAPE, dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="rotation shape does not match x"):
        apply_causal_rope(x[:, :400], rot)
    with pytest.raises(ValueError, match="rotation shape does not match x"):
        apply_causal_rope(x[..., :64], rot)
    with pytest.raises(ValueError, match=r"expected x=\[B,S,N,D\]"):
        apply_causal_rope(x[0], rot)
    with pytest.raises(TypeError, match="bfloat16"):
        apply_causal_rope(x.astype(mx.float32), rot)


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e3])
@pytest.mark.parametrize("start_frame", [0, 1, 20, 21, 22, 500, 8000])
def test_apply_bitwise_matches_cpu_float64(freqs, start_frame, scale):
    mx.random.seed(1000 + start_frame)
    x = (mx.random.normal(SHAPE) * scale).astype(mx.bfloat16)
    out = apply_causal_rope(x, build_causal_rope_rotations(freqs, F, H, W, start_frame))
    _assert_bitwise(out, causal_rope_apply(x, GRID, freqs, start_frame))


@pytest.mark.parametrize("start_frame", [0, 22, 8000])
def test_apply_bitwise_edge_values(freqs, start_frame):
    """Every finite bf16 bit pattern, as both the real and the imaginary lane, at many (token, pair) slots."""
    pats = _finite_bf16_patterns()
    rng = np.random.default_rng(start_frame)
    n = math.prod(SHAPE)
    copies = [pats, np.roll(pats, 1)]  # the roll moves every pattern to the other lane of its pair
    while sum(c.size for c in copies) < n:
        copies.append(rng.permutation(pats))
    tile = np.concatenate(copies)[:n]
    head = np.concatenate([pats, np.roll(pats, 1)])
    assert np.array_equal(tile[:head.size], head)
    x = _from_bits(tile)
    out = apply_causal_rope(x, build_causal_rope_rotations(freqs, F, H, W, start_frame))
    _assert_bitwise(out, causal_rope_apply(x, GRID, freqs, start_frame))


def _f32_round_bits(v: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return v.astype(np.float32).view(np.uint32)


def _hard_cases(freqs, start_frame: int, rng) -> tuple[np.ndarray, dict]:
    """bf16 pairs that stress the float64 sum and the float32 double rounding.

    Per (token, pair) slot and head: cancellation pairs (a*c ~= b*s for the real lane, a*s ~= -b*c for the
    imaginary lane: the subtraction loses most leading bits) and pairs whose float32 rounding lands exactly on a
    bf16 tie (a direct float64 -> bf16 rounding would differ). Returns x bits [SHAPE] and counts.
    """
    s_len, n_heads, pairs = SHAPE[1], SHAPE[2], SHAPE[3] // 2
    rot = build_causal_rope_rotations(freqs, F, H, W, start_frame)
    tab = np.array(rot.table).view(np.float64)  # [S, P, 2] (cos, sin) bits as float64, kernel's own table
    c, s = tab[..., 0][:, None, :], tab[..., 1][:, None, :]  # [S, 1, P]

    def bf16_round(v):  # float64 -> bf16 bits through float32, as MLX's CPU cast
        u = _f32_round_bits(v).astype(np.uint64)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

    def bf16_val(bits):
        return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    a_bits = bf16_round(rng.standard_normal((s_len, n_heads, pairs)) * np.exp2(rng.integers(-20, 20, (s_len, n_heads,
                                                                                                    pairs))))
    a = bf16_val(a_bits)
    third = n_heads // 3
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        b_real = np.where(s != 0, a * c / s, 0.0)        # real lane cancels: a*c - b*s ~ 0
        b_imag = np.where(c != 0, -a * s / c, 0.0)       # imaginary lane cancels: a*s + b*c ~ 0
    b_bits = np.empty_like(a_bits)
    b_bits[:, :third] = bf16_round(b_real)[:, :third]
    b_bits[:, third:2 * third] = bf16_round(b_imag)[:, third:2 * third]
    b_bits[((b_bits >> 7) & 0xFF) == 0xFF] = 0  # a*c/s can overflow bf16 where s is tiny: keep inputs finite
    # Last third: float32-rounded result exactly on a bf16 tie (searched among 512 random b per slot).
    cand = bf16_round(rng.standard_normal((512, s_len, 1, pairs)) * np.exp2(rng.integers(-20, 20, (512, s_len, 1,
                                                                                                    pairs))))
    a_t = a[:, 2 * third:2 * third + 1]                  # [S, 1, P], one a per slot for the search
    bv = bf16_val(cand)
    with np.errstate(over="ignore", invalid="ignore"):
        real = a_t * c - bv * s
        imag = a_t * s + bv * c
    tie = lambda v: (_f32_round_bits(v) & 0xFFFF) == 0x8000  # noqa: E731
    hit = tie(real) | tie(imag)                           # [512, S, 1, P]
    first = np.argmax(hit, axis=0)                        # [S, 1, P]
    found = np.take_along_axis(hit, first[None], axis=0)[0]
    picked = np.take_along_axis(cand, first[None], axis=0)[0]
    a_bits[:, 2 * third:] = np.where(found, a_bits[:, 2 * third:2 * third + 1], a_bits[:, 2 * third:])
    b_bits[:, 2 * third:] = np.where(found, picked, b_bits[:, 2 * third:])
    x = np.stack([a_bits, b_bits], axis=-1).reshape(SHAPE)

    # Discrimination counts: how many outputs a sloppier emulation would get wrong.
    av, bvv = bf16_val(a_bits), bf16_val(b_bits)
    with np.errstate(over="ignore", invalid="ignore"):
        lanes = np.stack([av * c - bvv * s, av * s + bvv * c], axis=-1)
    f32_bits = _f32_round_bits(lanes)
    direct_ties = int(np.sum(((f32_bits & 0xFFFF) == 0x8000) & (lanes != f32_bits.view(np.float32))))
    return x, {"tie_slots": int(found.sum()), "direct_rounding_differs": direct_ties}


@pytest.mark.parametrize("start_frame", [0, 21, 7000])
def test_apply_bitwise_rounding_hard_cases(freqs, start_frame):
    bits, counts = _hard_cases(freqs, start_frame, np.random.default_rng(77 + start_frame))
    # The inputs must discriminate: a kernel rounding float64 -> bf16 directly would fail on these outputs.
    assert counts["direct_rounding_differs"] >= 50 and counts["tie_slots"] >= 50, counts
    x = _from_bits(bits)
    out = apply_causal_rope(x, build_causal_rope_rotations(freqs, F, H, W, start_frame))
    _assert_bitwise(out, causal_rope_apply(x, GRID, freqs, start_frame))


def _cpu_float64_lanes(x: mx.array, rot: RopeRotations) -> np.ndarray:
    """The reference's float64 lanes (MLX CPU, same ops as causal_rope_apply before its cast), as uint64 bits."""
    tab = np.array(rot.table).view(np.float64)
    b, s, n, d = x.shape
    with mx.stream(mx.cpu):
        lanes = _apply_rope_rotation(x.astype(mx.float64).reshape(b, s, n, d // 2, 2),
                                     mx.array(tab[None, :, None, :, 0], dtype=mx.float64),
                                     mx.array(tab[None, :, None, :, 1], dtype=mx.float64))
        return np.array(lanes).view(np.uint64)


@pytest.mark.parametrize("start_frame", [0, 22, 8000])
def test_float64_emulation_bitwise(freqs, start_frame):
    """The kernel's emulated float64 lanes equal the CPU float64 lanes on all 53 bits (product rounding visible)."""
    pats = _finite_bf16_patterns()
    rng = np.random.default_rng(10 + start_frame)
    n = math.prod(SHAPE)
    x = _from_bits(np.concatenate([pats, np.roll(pats, 1)] + [rng.permutation(pats) for _ in range(18)])[:n])
    rot = build_causal_rope_rotations(freqs, F, H, W, start_frame)
    got = np.array(_float64_lanes_bits(x, rot))
    want = _cpu_float64_lanes(x, rot)
    bad = np.flatnonzero(got.ravel() != want.ravel())
    assert bad.size == 0, f"{bad.size} float64 mismatches, first at {bad[:5].tolist()}"
    # The comparison sees product rounding: a fused a*c - b*s (single rounding) differs on many lanes.
    xv = np.array(x.astype(mx.float32)).astype(np.float64).reshape(-1, 2)
    tab = np.array(rot.table).view(np.float64)
    tok = (np.arange(xv.shape[0]) // (SHAPE[2] * SHAPE[3] // 2)) % SHAPE[1]
    pair = np.arange(xv.shape[0]) % (SHAPE[3] // 2)
    sample = rng.choice(xv.shape[0], 20000, replace=False)
    fused = sum(np.float64(math.fma(xv[i, 0], tab[tok[i], pair[i], 0], -(xv[i, 1] * tab[tok[i], pair[i], 1])))
                .view(np.uint64) != want.reshape(-1, 2)[i, 0] for i in sample)
    assert fused > 100, fused


def test_apply_non_finite_inputs(freqs):
    """inf/NaN lanes: same bits as the CPU path (NaN results are MLX's canonical bf16 NaN 0x7FC0)."""
    rng = np.random.default_rng(3)
    bits = rng.choice(_finite_bf16_patterns(), size=math.prod(SHAPE))
    special = np.array([0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFFF], dtype=np.uint16)
    where = rng.random(bits.size) < 0.05
    bits[where] = rng.choice(special, size=int(where.sum()))
    x = _from_bits(bits)
    for start_frame in (0, 22):
        out = apply_causal_rope(x, build_causal_rope_rotations(freqs, F, H, W, start_frame))
        _assert_bitwise(out, causal_rope_apply(x, GRID, freqs, start_frame))


@pytest.mark.parametrize("position", [22, 100, 7000])
def test_frames_rotations_bitwise_match_rope_apply_frames(freqs, position):
    frame_pos = [p + (position - 21) for p in range(1)]
    mx.random.seed(position)
    x = mx.random.normal(SHAPE).astype(mx.bfloat16)
    out = apply_causal_rope(x, build_rope_rotations_frames(freqs, frame_pos, H, W))
    _assert_bitwise(out, rope_apply_frames(x, GRID, freqs, frame_pos))


def test_frames_rotations_multi_frame(freqs):
    """Non-contiguous multi-frame positions, batch of 2 sharing the rotations."""
    frame_pos, h, w = [3, 100, 4000], 3, 5
    mx.random.seed(5)
    x = mx.random.normal((2, 3 * h * w, 4, 128)).astype(mx.bfloat16)
    out = apply_causal_rope(x, build_rope_rotations_frames(freqs, frame_pos, h, w))
    _assert_bitwise(out, rope_apply_frames(x, [[3, h, w]] * 2, freqs, frame_pos))
    out = apply_causal_rope(x, build_causal_rope_rotations(freqs, 3, h, w, start_frame=7))
    _assert_bitwise(out, causal_rope_apply(x, [[3, h, w]] * 2, freqs, 7))


def test_apply_non_contiguous_input(freqs):
    mx.random.seed(11)
    x = mx.random.normal((1, 24, 405, 128)).astype(mx.bfloat16).transpose(0, 2, 1, 3)  # [B, S, N, D] strided
    out = apply_causal_rope(x, build_causal_rope_rotations(freqs, F, H, W, 21))
    _assert_bitwise(out, causal_rope_apply(x, GRID, freqs, 21))


def test_apply_stays_on_gpu_lazy(freqs, monkeypatch):
    """apply_causal_rope builds one lazy GPU kernel: no stream switch (mx.cpu), no float64 input, no eval."""
    rot = build_causal_rope_rotations(freqs, F, H, W, 21)
    assert rot.table.dtype == mx.uint64
    x = mx.random.normal(SHAPE).astype(mx.bfloat16)
    mx.eval(x)

    def forbidden(*args, **kwargs):
        raise AssertionError("apply_causal_rope must not switch streams or evaluate")
    for name in ("stream", "eval", "async_eval"):
        monkeypatch.setattr(mx, name, forbidden)
    out = apply_causal_rope(x, rot)
    monkeypatch.undo()
    assert out.shape == SHAPE and out.dtype == mx.bfloat16
    mx.eval(out)


def test_freqs_stay_float64_reference(freqs):
    model = CausalWanModel(model_type="ti2v", text_len=8, in_dim=12, dim=256, ffn_dim=256, text_dim=32, out_dim=4,
                           num_heads=2, num_layers=1)
    before = [np.array(t).copy() for t in model.freqs]
    rot = build_causal_rope_rotations(model.freqs, 1, 3, 5, start_frame=2)
    mx.eval(apply_causal_rope(mx.zeros((1, 15, 2, 128), dtype=mx.bfloat16), rot))
    assert all(t.dtype == mx.float64 for t in model.freqs)
    assert all(np.array_equal(np.array(t), b) for t, b in zip(model.freqs, before))
    assert all(t.dtype == mx.float64 for t in freqs)
