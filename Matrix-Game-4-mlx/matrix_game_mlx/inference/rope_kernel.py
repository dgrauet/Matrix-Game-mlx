"""GPU causal RoPE, bitwise identical to the float64 CPU path (MLX port of matrix_game/inference/rope_kernel.py).

The reference (`causal_rope_apply` / `rope_apply_frames`) promotes the bf16 input to float64 on the CPU stream and
computes, per channel pair (a, b) with table values (c, s):

    real = fl64(fl64(a * c) - fl64(b * s))        imag = fl64(fl64(a * s) + fl64(b * c))

(each product rounded to float64: MLX CPU elementwise ops are separate kernels, no FMA contraction), then casts
float64 -> bfloat16. MLX's CPU cast goes through float32: round-to-nearest-even to float32, then
round-to-nearest-even to bfloat16 (double rounding; e.g. 1 + 2^-8 + 2^-30 -> 0x3F80, a direct rounding would give
0x3F81). NaN results are the canonical bf16 NaN 0x7FC0.

Apple GPUs have no float64, so the kernel emulates that chain exactly in 64-bit integer arithmetic: a bf16
significand (8 bits) times a float64 significand (53 bits) is an exact 61-bit integer product, rounded to 53
bits (RNE); the sum aligns the two 53-bit significands with 10 guard bits and a sticky bit (exact rounding, as
in soft-float libraries); the float64 result is rounded to float32 (RNE, with float32 subnormals and overflow)
and the float32 bits to bf16 with MLX's RNE bit trick. The rotation tables carry the raw float64 bits (uint64,
[S, D/2, 2] = (cos, sin) as upstream's view_as_real layout), built on the host once per DiT call and shared by
every layer and by q and k; applying them is a single GPU kernel.

Deviation from upstream: `freqs` is the port's `(cos, sin)` float64 pair (not a complex tensor), the rotations are
a `RopeRotations` dataclass (not a raw tensor), and `apply_causal_rope` takes bfloat16 only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import numpy as np

__all__ = ["RopeRotations", "apply_causal_rope", "build_causal_rope_rotations", "build_rope_rotations_frames"]

_EXP_MIN, _EXP_MAX = 1023 - 512, 1023 + 512  # table magnitudes in [2^-512, 2^512): no float64 under/overflow


@dataclass(frozen=True, eq=False)
class RopeRotations:
    """Expanded RoPE rotations for one frame-aligned grid.

    table: uint64 [seq_len, pairs, 2], the float64 bits of (cos, sin) per token and channel pair (GPU-resident).
    """
    table: mx.array
    seq_len: int
    pairs: int


def _check_freqs(freqs) -> tuple[mx.array, mx.array]:
    try:
        cos, sin = freqs
    except (TypeError, ValueError):
        cos = sin = None
    if not (isinstance(cos, mx.array) and isinstance(sin, mx.array) and cos.ndim == 2 and cos.shape == sin.shape
            and cos.dtype == mx.float64 and sin.dtype == mx.float64):
        raise ValueError("freqs must be a (cos, sin) pair of two-dimensional float64 arrays")
    return cos, sin


def _expand(freqs, time_rows: Sequence[int], height: int, width: int) -> RopeRotations:
    """Gather (time, height, width) parts on the host and expand them over the [frames, height, width] grid."""
    cos, sin = freqs
    frames, c = len(time_rows), cos.shape[1]
    c_h = c // 3
    c_t = c - 2 * c_h
    idx = mx.array(np.asarray(time_rows, dtype=np.int32))
    with mx.stream(mx.cpu):  # float64 slices live on the CPU stream
        parts = [(mx.take(t[:, :c_t], idx, axis=0), t[:height, c_t:c_t + c_h], t[:width, c_t + c_h:])
                 for t in (cos, sin)]
        parts = [[np.array(p) for p in group] for group in parts]
    grids = []
    for tp, hp, wp in parts:
        g = np.concatenate([np.broadcast_to(tp[:, None, None, :], (frames, height, width, c_t)),
                            np.broadcast_to(hp[None, :, None, :], (frames, height, width, c_h)),
                            np.broadcast_to(wp[None, None, :, :], (frames, height, width, c - c_t - c_h))], axis=-1)
        grids.append(g.reshape(frames * height * width, c))
    bits = np.ascontiguousarray(np.stack(grids, axis=-1)).view(np.uint64)
    exp = (bits >> np.uint64(52)) & np.uint64(0x7FF)
    zero = (bits & np.uint64(0x7FFF_FFFF_FFFF_FFFF)) == 0
    if not np.all(zero | ((exp >= _EXP_MIN) & (exp < _EXP_MAX))):
        raise ValueError("RoPE table values must be zero or of magnitude in [2^-512, 2^512)")
    return RopeRotations(table=mx.array(bits), seq_len=frames * height * width, pairs=c)


def _check_spatial(freqs, height: int, width: int) -> None:
    if height > freqs[0].shape[0] or width > freqs[0].shape[0]:
        raise ValueError("spatial RoPE grid exceeds the precomputed frequency table")


def build_causal_rope_rotations(freqs, frames: int, height: int, width: int, start_frame: int = 0) -> RopeRotations:
    """Expand Wan's three-axis RoPE table for a frame-aligned grid, frame axis offset by `start_frame`.

    Same rotations as `causal_rope_apply(x, [[frames, height, width]], freqs, start_frame)`.
    """
    freqs = _check_freqs(freqs)
    if frames <= 0 or height <= 0 or width <= 0 or start_frame < 0:
        raise ValueError("RoPE grid dimensions must be positive and start non-negative")
    if start_frame + frames > freqs[0].shape[0]:
        raise ValueError("temporal RoPE position exceeds the precomputed frequency table")
    _check_spatial(freqs, height, width)
    return _expand(freqs, range(int(start_frame), int(start_frame) + int(frames)), int(height), int(width))


def build_rope_rotations_frames(freqs, frame_pos: Sequence[int], height: int, width: int) -> RopeRotations:
    """Rotations with explicit per-frame temporal positions, as `rope_apply_frames` (test-only helper: production's
    fixed-sink re-RoPE is `build_causal_rope_rotations(freqs, 1, h, w, position - window)`, runtime.py)."""
    freqs = _check_freqs(freqs)
    pos = [int(p) for p in frame_pos]
    if not pos:
        raise ValueError("frame_pos must not be empty")
    if min(pos) < 0:
        raise NotImplementedError("negative RoPE positions are only used by the kv_mem path (not ported)")
    if height <= 0 or width <= 0:
        raise ValueError("RoPE grid dimensions must be positive and start non-negative")
    if max(pos) >= freqs[0].shape[0]:
        raise ValueError("temporal RoPE position exceeds the precomputed frequency table")
    _check_spatial(freqs, height, width)
    return _expand(freqs, pos, int(height), int(width))


_HEADER = r"""
// value = (-1)^neg * m * 2^e with 2^52 <= m < 2^53, or a signed zero.
struct F64 { ulong m; int e; bool neg; bool zero; };

inline F64 f64_zero(bool neg) { F64 r; r.m = 0ul; r.e = 0; r.neg = neg; r.zero = true; return r; }

// v / 2^sh rounded to nearest, ties to even; 1 <= sh <= 63.
inline ulong rne_shift(ulong v, uint sh) {
  ulong q = v >> sh;
  ulong rem = v & ((1ul << sh) - 1ul);
  ulong hb = 1ul << (sh - 1u);
  if (rem > hb || (rem == hb && (q & 1ul) != 0ul)) q += 1ul;
  return q;
}

// Exact value v * 2^e (v != 0) rounded to a 53-bit significand (no float64 subnormals/overflow in range).
inline F64 f64_round(ulong v, int e, bool neg) {
  int len = 64 - int(clz(v));
  if (len > 53) {
    uint sh = uint(len - 53);
    v = rne_shift(v, sh);
    e += int(sh);
    if (v == (1ul << 53)) { v >>= 1; e += 1; }
  } else if (len < 53) {
    uint sh = uint(53 - len);
    v <<= sh;
    e -= int(sh);
  }
  F64 r; r.m = v; r.e = e; r.neg = neg; r.zero = false; return r;
}

// float64 bits -> F64; the host guarantees zero or a normal of moderate magnitude.
inline F64 f64_decode(ulong bits) {
  uint ef = uint(bits >> 52) & 0x7FFu;
  bool neg = (bits >> 63) != 0ul;
  if (ef == 0u) return f64_zero(neg);
  F64 r; r.m = (bits & 0xFFFFFFFFFFFFFul) | (1ul << 52); r.e = int(ef) - 1075; r.neg = neg; r.zero = false;
  return r;
}

// fl64(bf16 a * c), a finite.
inline F64 f64_mul(uint a, F64 c) {
  bool neg = (((a >> 15) & 1u) != 0u) != c.neg;
  uint ea = (a >> 7) & 0xFFu, fa = a & 0x7Fu;
  if ((ea == 0u && fa == 0u) || c.zero) return f64_zero(neg);
  ulong ma = ulong(ea == 0u ? fa : (fa | 0x80u));
  int e = (ea == 0u ? -133 : int(ea) - 134) + c.e;
  return f64_round(ma * c.m, e, neg);  // <= 61-bit exact product
}

// fl64(x + y), IEEE round-to-nearest-even (exact cancellation gives +0).
inline F64 f64_add(F64 x, F64 y) {
  if (x.zero && y.zero) return f64_zero(x.neg && y.neg);
  if (y.zero) return x;
  if (x.zero) return y;
  if (x.e < y.e || (x.e == y.e && x.m < y.m)) { F64 t = x; x = y; y = t; }
  int d = x.e - y.e;
  ulong X = x.m << 10, Y = y.m << 10;   // 10 guard bits; Y gets a sticky bit when shifted bits are lost
  if (d >= 64) Y = 1ul;
  else if (d > 0) Y = (Y >> uint(d)) | ((Y & ((1ul << uint(d)) - 1ul)) != 0ul ? 1ul : 0ul);
  ulong S = (x.neg == y.neg) ? X + Y : X - Y;
  if (S == 0ul) return f64_zero(false);
  return f64_round(S, x.e - 10, x.neg);
}

// float64 -> float32 (RNE, subnormals, overflow to inf) -> bf16 (MLX's RNE bit trick); returns bf16 bits.
inline uint f64_to_bf16(F64 v) {
  uint sign = v.neg ? 0x8000u : 0u;
  if (v.zero) return sign;
  int E = v.e + 52;
  uint f;
  if (E > 127) {
    f = 0x7F800000u;
  } else if (E >= -126) {
    ulong q = rne_shift(v.m, 29u);
    if (q == (1ul << 24)) { q >>= 1; E += 1; }
    f = E > 127 ? 0x7F800000u : ((uint(E + 127) << 23) | (uint(q) & 0x7FFFFFu));
  } else {
    int sh = -149 - v.e;               // float32 subnormal: k * 2^-149, k = m * 2^-sh; sh > 29 here
    f = sh > 63 ? 0u : uint(rne_shift(v.m, uint(sh)));  // k == 2^23 is the smallest normal's bit pattern
  }
  return sign | ((f + 0x7FFFu + ((f >> 16) & 1u)) >> 16);
}

// Non-finite lanes: class of bf16 a * c (a non-finite): 1 +inf, 2 -inf, 3 NaN; 0 when a is finite.
inline uint mul_class(uint a, F64 c) {
  if (((a >> 7) & 0xFFu) != 0xFFu) return 0u;
  if ((a & 0x7Fu) != 0u || c.zero) return 3u;
  return ((((a >> 15) & 1u) != 0u) != c.neg) ? 2u : 1u;
}

// p + (q or -q) where at least one class is non-finite (the other product is finite): bf16 bits.
inline uint nonfinite_lane(uint p, uint q, bool subtract) {
  if (subtract && (q == 1u || q == 2u)) q = 3u - q;
  if (p == 3u || q == 3u) return 0x7FC0u;
  if (p != 0u && q != 0u && p != q) return 0x7FC0u;
  uint k = p != 0u ? p : q;
  return k == 1u ? 0x7F80u : 0xFF80u;
}

// float64 bits of an F64 (test hook: exposes the emulated float64 lanes before the bf16 cast).
inline ulong f64_encode(F64 v) {
  ulong sign = v.neg ? (1ul << 63) : 0ul;
  if (v.zero) return sign;
  return sign | (ulong(v.e + 1075) << 52) | (v.m & 0xFFFFFFFFFFFFFul);
}
"""

_SOURCE_PROLOGUE = r"""
  uint gid = thread_position_in_grid.x;
  uint pair = gid % P;
  uint tok = (gid / (P * N)) % S;
  uint w = x[gid];
  uint a = w & 0xFFFFu, b = w >> 16;   // little-endian: low half is the real lane
  uint ri = (tok * P + pair) * 2u;
  F64 c = f64_decode(rot[ri]);
  F64 s = f64_decode(rot[ri + 1u]);
"""

_SOURCE = _SOURCE_PROLOGUE + r"""
  uint re, im;
  if (((a >> 7) & 0xFFu) != 0xFFu && ((b >> 7) & 0xFFu) != 0xFFu) {
    F64 bs = f64_mul(b, s); bs.neg = !bs.neg;
    re = f64_to_bf16(f64_add(f64_mul(a, c), bs));
    im = f64_to_bf16(f64_add(f64_mul(a, s), f64_mul(b, c)));
  } else {
    re = nonfinite_lane(mul_class(a, c), mul_class(b, s), true);
    im = nonfinite_lane(mul_class(a, s), mul_class(b, c), false);
  }
  out[gid] = re | (im << 16);
"""

_SOURCE_F64 = _SOURCE_PROLOGUE + r"""
  F64 bs = f64_mul(b, s); bs.neg = !bs.neg;   // finite inputs only
  out[2u * gid] = f64_encode(f64_add(f64_mul(a, c), bs));
  out[2u * gid + 1u] = f64_encode(f64_add(f64_mul(a, s), f64_mul(b, c)));
"""

_kernels: dict = {}


def _get_kernel(f64_lanes: bool):
    if f64_lanes not in _kernels:
        _kernels[f64_lanes] = mx.fast.metal_kernel(
            name="mg4_causal_rope_f64_lanes" if f64_lanes else "mg4_causal_rope_f64", input_names=["x", "rot"],
            output_names=["out"], source=_SOURCE_F64 if f64_lanes else _SOURCE, header=_HEADER)
    return _kernels[f64_lanes]


def _launch(x: mx.array, rotations: RopeRotations, f64_lanes: bool) -> mx.array:
    if x.ndim != 4 or rotations.table.ndim != 3:
        raise ValueError("expected x=[B,S,N,D] and rotations=[S,D/2,2]")
    if x.shape[-1] % 2 or rotations.table.shape[-1] != 2:
        raise ValueError("head dimension and rotation pairs must be even")
    if x.shape[1] != rotations.seq_len or x.shape[-1] // 2 != rotations.pairs:
        raise ValueError("rotation shape does not match x")
    if x.dtype != mx.bfloat16:
        raise TypeError(f"x must be bfloat16 (the kernel emulates float64 products of bf16 inputs), got {x.dtype}")
    b, s, n, d = x.shape
    pairs = x.view(mx.uint32)  # [B, S, N, D/2]: one (real, imag) bf16 pair per word
    shape, dtype = ((b, s, n, d), mx.uint64) if f64_lanes else (pairs.shape, mx.uint32)
    (out,) = _get_kernel(f64_lanes)(inputs=[pairs, rotations.table], template=[("S", s), ("N", n), ("P", d // 2)],
                                    grid=(b * s * n * (d // 2), 1, 1), threadgroup=(256, 1, 1),
                                    output_shapes=[shape], output_dtypes=[dtype])
    return out


def apply_causal_rope(x: mx.array, rotations: RopeRotations) -> mx.array:
    """Apply causal RoPE to bf16 `x` [B, S, N, D]; bitwise equal to the float64 CPU path. GPU only, lazy."""
    return _launch(x, rotations, f64_lanes=False).view(mx.bfloat16)


def _float64_lanes_bits(x: mx.array, rotations: RopeRotations) -> mx.array:
    """Test hook: the kernel's emulated float64 results (uint64 bits [B, S, N, D]) before the float32/bf16 cast.

    Finite inputs only. Lets tests check the float64 emulation itself (all 53 bits, so product rounding is
    visible), which the bf16 output hides almost always.
    """
    return _launch(x, rotations, f64_lanes=True)
