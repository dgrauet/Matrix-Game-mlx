"""Metal kernels that fuse the glue between the GEMMs and attention calls of a causal DiT block (MLX port of
matrix_game/inference/kernels.py).

A block is five linear layers and two attention calls connected by long chains of tiny elementwise and row-reduction
operations (LayerNorm, timestep modulation, gated residuals, RMSNorm, the GELU activation). On the native path each
of those is its own MLX kernel (about 60 per block, 1800 per DiT call); each function below replaces one whole chain
with a single launch:

* `ln_modulate`     LayerNorm(x) * (1 + scale) + shift, where shift / scale are rows of `modulation + e`;
                    optionally adds a residual to x first.
* `gate_ln_affine`  x + y * gate, followed by the affine LayerNorm before cross-attention.
* `gate_residual`   x + y * gate.
* `rms_norm`        `WanRMSNorm` (q / k normalisation): fp32 statistics, cast to bf16, times the bf16 weight.
  (port addition, not in upstream's kernels.py)
* `gelu_tanh`       the FFN activation `nn.gelu_approx(x.astype(float32)).astype(bfloat16)` as one compiled
                    kernel (`mx.compile` of the native expression; port addition).

Rounding follows the native MLX block (which is upstream's unfused bf16 block): every intermediate the chain of
separate operations materialises in bf16 is rounded to bf16 here too (`bf()`: float -> bfloat16_t -> float, MLX's
round-to-nearest-even), fp32 intermediates are rounded to fp32 one operation at a time (`#pragma METAL fp
contract(off)`: no fused multiply-add the separate kernels would not do), reductions accumulate in fp32 with the
native formulas (mean = sum * float32(1 / C); variance two-pass, mean of the squared centred values; `rsqrt`
precise, as `mx.rsqrt`). Upstream allows the summation order inside a row reduction to differ; here each row is
summed in the layout of MLX's own contiguous row reduction (`row_reduce_simple`, N_READS = 4: about C / 4 threads,
4 consecutive values per thread added in order, then `simd_sum` and a `simd_sum` over the simdgroup partials), so
where MLX dispatches that kernel with that layout (measured on MLX 0.31.1 at C = 128, 3072, 4096; C = 3072 is the
model width) the statistics, and with them every output, are bitwise native. At other widths only the summation
order differs. `gate_residual`, the residual sums and `gelu_tanh` are bitwise native at any width.

One threadgroup handles one row ([B, S, C] is B * S rows); a row's values stay in registers between the statistics
and the output pass. Inputs are checked on the host and rejected with a ValueError; nothing falls back to an
unfused path. Deviations from upstream: MLX arrays (no strides: `e` is used densely, `metal_kernel` makes inputs
row-contiguous), the two extra kernels above, and the row indices are named `*_row` (upstream `*_k`).
"""
from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ["MODULATION_ROWS", "gate_ln_affine", "gate_residual", "gelu_tanh", "ln_modulate", "rms_norm"]

MODULATION_ROWS = 6  # shift / scale / gate rows of a block's timestep modulation

_HEADER = r"""
#pragma METAL fp contract(off)

// Round a float to bf16 (round to nearest even, as a separate MLX bf16 op) and return it as float.
inline float bf(float v) { return float(bfloat16_t(v)); }

// Sum of one float per thread over the threadgroup (NSG simdgroups of 32); every thread gets the total.
inline float row_sum(float v, threadgroup float* sh, uint lane, uint sg, uint nsg) {
  v = simd_sum(v);
  if (lane == 0) sh[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < nsg ? sh[lane] : 0.0f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// Row layout of MLX's contiguous row reduction (`row_reduce_simple`, N_READS = 4): thread tid owns the columns
// blk * TG * 4 + tid * 4 + (0..3) for blk < BLOCKS, then the same columns of a last partial block below EXTRA.
// Value i of a thread (i < K = 4 * blocks incl. the partial one) is column COL(i), present when VALID(i).
#define COL(i) (uint((i) / 4) * TG * 4 + tid * 4 + uint((i) % 4))
#define VALID(i) ((i) / 4 < BLOCKS || tid * 4 + uint((i) % 4) < EXTRA)
"""

# Shared prologue of the row kernels: row indexing and the row's values (v[i] = column COL(i)).
_ROW_PROLOGUE = r"""
  const uint row = threadgroup_position_in_grid.x;
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  threadgroup float sh[32];
  const uint base = row * C;
  const float inv_n = as_type<float>(INVN);
  const float eps = as_type<float>(EPS);
  float v[K];
"""

_MOD_INDEX = r"""
  const uint eoff = ((row / S) * NF + (row % S) / FT) * 6 * C;  // e[b, f] for this row's frame
"""

# LayerNorm statistics of the row in v: mean and 1 / sqrt(var + eps), as WanLayerNorm (fp32, two-pass).
_LN_STATS = r"""
  const float mean = row_sum(acc, sh, lane, sg, TG / 32) * inv_n;
  float sq = 0.0f;
  for (int i = 0; i < K; ++i) {
    if (VALID(i)) { const float d = v[i] - mean; sq = d * d + sq; }
  }
  const float rstd = metal::precise::rsqrt(row_sum(sq, sh, lane, sg, TG / 32) * inv_n + eps);
"""

_LN_MODULATE = r"""
  float acc = 0.0f;
  for (int i = 0; i < K; ++i) {
    const uint c = COL(i);
    float t = 0.0f;
    if (VALID(i)) {
      t = float(x[base + c]);
      ADD_RESIDUAL
    }
    v[i] = t;
    acc = t + acc;  // MLX's Sum: op(value, total)
  }
""" + _LN_STATS + r"""
  for (int i = 0; i < K; ++i) {
    const uint c = COL(i);
    if (VALID(i)) {
      const float n = bf((v[i] - mean) * rstd);
      const float shift = bf(float(modulation[SHIFT * C + c]) + float(e[eoff + SHIFT * C + c]));
      const float scale = bf(1.0f + bf(float(modulation[SCALE * C + c]) + float(e[eoff + SCALE * C + c])));
      z[base + c] = bfloat16_t(bf(n * scale) + shift);
    }
  }
"""

_ADD_RESIDUAL = "t = bf(t + float(residual[base + c])); x_out[base + c] = bfloat16_t(t);"

_GATE_LN_AFFINE = r"""
  float acc = 0.0f;
  for (int i = 0; i < K; ++i) {
    const uint c = COL(i);
    float t = 0.0f;
    if (VALID(i)) {
      const float g = bf(float(modulation[GATE * C + c]) + float(e[eoff + GATE * C + c]));
      t = bf(float(x[base + c]) + bf(float(y[base + c]) * g));
      x_out[base + c] = bfloat16_t(t);
    }
    v[i] = t;
    acc = t + acc;  // MLX's Sum: op(value, total)
  }
""" + _LN_STATS + r"""
  for (int i = 0; i < K; ++i) {
    const uint c = COL(i);
    if (VALID(i)) {
      // The affine LayerNorm normalises and applies weight / bias in fp32, then rounds once.
      z[base + c] = bfloat16_t(((v[i] - mean) * rstd) * float(weight[c]) + float(bias[c]));
    }
  }
"""

_RMS_NORM = r"""
  float acc = 0.0f;
  for (int i = 0; i < K; ++i) {
    const float t = VALID(i) ? float(x[base + COL(i)]) : 0.0f;
    v[i] = t;
    acc = t * t + acc;
  }
  const float r = metal::precise::rsqrt(row_sum(acc, sh, lane, sg, TG / 32) * inv_n + eps);
  for (int i = 0; i < K; ++i) {
    const uint c = COL(i);
    if (VALID(i)) out[base + c] = bfloat16_t(bf(v[i] * r) * float(weight[c]));
  }
"""

_GATE_RESIDUAL = r"""
  const uint idx = thread_position_in_grid.x;
  if (idx >= N) return;
  const uint c = idx % C;
  const uint row = idx / C;
  const uint eoff = ((row / S) * NF + (row % S) / FT) * 6 * C;
  const float g = bf(float(modulation[GATE * C + c]) + float(e[eoff + GATE * C + c]));
  out[idx] = bfloat16_t(float(x[idx]) + bf(float(y[idx]) * g));
"""


def _f32_bits(v: float) -> int:
    return int(np.array(v, dtype=np.float32).view(np.uint32))


@lru_cache(maxsize=None)
def _kernel(name: str, inputs: tuple, outputs: tuple, source: str):
    return mx.fast.metal_kernel(name=name, input_names=list(inputs), output_names=list(outputs), source=source,
                                header=_HEADER)


def _row_layout(c: int) -> tuple[int, int, int]:
    """(threads, full blocks, extra columns) of MLX's `row_reduce_simple` for a row of `c` values: about c / 4
    threads (a multiple of 32, at most 1024), 4 consecutive values per thread and block."""
    tg = min(1024, 32 * math.ceil(math.ceil(c / 4) / 32))
    blocks = c // (tg * 4)
    return tg, blocks, c - blocks * tg * 4


def _row_kernel(name, inputs, outputs, body, arrays, rows, c, template, out_shapes):
    """Launch a one-threadgroup-per-row kernel over `rows` rows of `c` channels."""
    tg, blocks, extra = _row_layout(c)
    kernel = _kernel(name, inputs, outputs, _ROW_PROLOGUE + body)
    return kernel(inputs=list(arrays),
                  template=[("C", c), ("TG", tg), ("BLOCKS", blocks), ("EXTRA", extra),
                            ("K", 4 * (blocks + (extra > 0))), ("INVN", _f32_bits(1.0 / c))] + list(template.items()),
                  grid=(rows * tg, 1, 1), threadgroup=(tg, 1, 1),
                  output_shapes=out_shapes, output_dtypes=[mx.bfloat16] * len(out_shapes))


# --- input checks ---------------------------------------------------------------------------------------------

def _check_rows(name: str, x) -> None:
    if not isinstance(x, mx.array) or x.dtype != mx.bfloat16 or x.ndim != 3:
        raise ValueError(f"{name} must be a BF16 array [B, S, C], got "
                         f"{getattr(x, 'shape', None)} {getattr(x, 'dtype', type(x))}")


def _check_same(name: str, y, x) -> None:
    _check_rows(name, y)
    if y.shape != x.shape:
        raise ValueError(f"{name} {tuple(y.shape)} does not match x {tuple(x.shape)}")


def _check_vector(name: str, v, channels: int) -> None:
    if not isinstance(v, mx.array) or v.dtype != mx.bfloat16 or v.shape != (channels,):
        raise ValueError(f"{name} must be a BF16 vector of {channels} elements, got "
                         f"{getattr(v, 'shape', None)} {getattr(v, 'dtype', type(v))}")


def _check_row_index(name: str, k) -> None:
    if not isinstance(k, int) or not 0 <= k < MODULATION_ROWS:
        raise ValueError(f"{name} must be a modulation row in 0..{MODULATION_ROWS - 1}, got {k!r}")


def _check_eps(eps) -> None:
    if not isinstance(eps, (int, float)) or not eps > 0:
        raise ValueError(f"eps must be a positive float, got {eps!r}")


def _modulation_layout(x, e, modulation) -> dict:
    """Validate e [B, F, 6, C] / modulation [1, 6, C]; returns the frame template values (S, NF, FT)."""
    b, s, c = x.shape
    if not isinstance(e, mx.array) or e.dtype != mx.bfloat16 or e.ndim != 4 or e.shape[0] != b \
            or e.shape[2:] != (MODULATION_ROWS, c):
        raise ValueError(f"e must be BF16 [B, F, {MODULATION_ROWS}, C] matching x {tuple(x.shape)}, got "
                         f"{getattr(e, 'shape', None)} {getattr(e, 'dtype', type(e))}")
    if not isinstance(modulation, mx.array) or modulation.dtype != mx.bfloat16 \
            or modulation.shape != (1, MODULATION_ROWS, c):
        raise ValueError(f"modulation must be BF16 [1, {MODULATION_ROWS}, C], got "
                         f"{getattr(modulation, 'shape', None)} {getattr(modulation, 'dtype', type(modulation))}")
    if s % e.shape[1]:
        raise ValueError(f"{s} tokens do not divide into {e.shape[1]} frames")
    return {"S": s, "NF": e.shape[1], "FT": s // e.shape[1]}


# --- kernels --------------------------------------------------------------------------------------------------

def ln_modulate(x: mx.array, e: mx.array, modulation: mx.array, eps: float, shift_row: int, scale_row: int,
                residual: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """z = LayerNorm(x') * (1 + (modulation + e)[scale_row]) + (modulation + e)[shift_row]; returns (z, x').

    x' is `x + residual` (a new bf16 array) when `residual` is given, otherwise `x` itself.
    x, residual [B, S, C]; e [B, F, 6, C] (one modulation per frame); modulation [1, 6, C]; all bf16.
    """
    _check_rows("x", x)
    layout = _modulation_layout(x, e, modulation)
    _check_eps(eps)
    _check_row_index("shift_row", shift_row)
    _check_row_index("scale_row", scale_row)
    if residual is not None:
        _check_same("residual", residual, x)
    b, s, c = x.shape
    res = residual is not None
    inputs = ("x", "residual", "e", "modulation") if res else ("x", "e", "modulation")
    outputs = ("z", "x_out") if res else ("z",)
    arrays = (x, residual, e, modulation) if res else (x, e, modulation)
    body = _MOD_INDEX + _LN_MODULATE.replace("ADD_RESIDUAL", _ADD_RESIDUAL if res else "")
    out = _row_kernel(f"mg4_ln_modulate{'_res' if res else ''}", inputs, outputs, body, arrays, b * s, c,
                      {"EPS": _f32_bits(eps), "SHIFT": shift_row, "SCALE": scale_row, **layout},
                      [x.shape] * len(outputs))
    return (out[0], out[1]) if res else (out[0], x)


def gate_ln_affine(x: mx.array, y: mx.array, e: mx.array, modulation: mx.array, gate_row: int,
                   weight: mx.array, bias: mx.array, eps: float) -> tuple[mx.array, mx.array]:
    """x' = x + y * (modulation + e)[gate_row] and z = LayerNorm(x') * weight + bias; returns (z, x').

    x, y [B, S, C]; e [B, F, 6, C]; modulation [1, 6, C]; weight, bias [C]; all bf16. Both results are new arrays.
    """
    _check_rows("x", x)
    _check_same("y", y, x)
    layout = _modulation_layout(x, e, modulation)
    _check_row_index("gate_row", gate_row)
    b, s, c = x.shape
    _check_vector("weight", weight, c)
    _check_vector("bias", bias, c)
    _check_eps(eps)
    z, x_out = _row_kernel("mg4_gate_ln_affine", ("x", "y", "e", "modulation", "weight", "bias"), ("z", "x_out"),
                           _MOD_INDEX + _GATE_LN_AFFINE, (x, y, e, modulation, weight, bias), b * s, c,
                           {"EPS": _f32_bits(eps), "GATE": gate_row,
                            **layout}, [x.shape, x.shape])
    return z, x_out


def gate_residual(x: mx.array, y: mx.array, e: mx.array, modulation: mx.array, gate_row: int) -> mx.array:
    """x + y * (modulation + e)[gate_row], with the modulation sum, the product and the sum each rounded to bf16.

    x, y [B, S, C]; e [B, F, 6, C]; modulation [1, 6, C]; all bf16. Returns a new [B, S, C] array.
    """
    _check_rows("x", x)
    _check_same("y", y, x)
    layout = _modulation_layout(x, e, modulation)
    _check_row_index("gate_row", gate_row)
    n = x.size
    kernel = _kernel("mg4_gate_residual", ("x", "y", "e", "modulation"), ("out",), _GATE_RESIDUAL)
    return kernel(inputs=[x, y, e, modulation],
                  template=[("C", x.shape[2]), ("N", n), ("GATE", gate_row)] + list(layout.items()),
                  grid=(n, 1, 1), threadgroup=(256, 1, 1), output_shapes=[x.shape], output_dtypes=[mx.bfloat16])[0]


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """`WanRMSNorm`: (x * rsqrt(mean(x * x) + eps)) in fp32, rounded to bf16, times the bf16 weight (rounded).

    x [B, S, C] bf16; weight [C] bf16.
    """
    _check_rows("x", x)
    b, s, c = x.shape
    _check_vector("weight", weight, c)
    _check_eps(eps)
    return _row_kernel("mg4_rms_norm", ("x", "weight"), ("out",), _RMS_NORM, (x, weight), b * s, c,
                       {"EPS": _f32_bits(eps)}, [x.shape])[0]


@mx.compile
def _gelu_tanh(x):
    return nn.gelu_approx(x.astype(mx.float32)).astype(mx.bfloat16)


def gelu_tanh(x: mx.array) -> mx.array:
    """The FFN activation of the native block (`model.gelu_tanh`: fp32 opmath, one cast) as one compiled kernel."""
    if not isinstance(x, mx.array) or x.dtype != mx.bfloat16:
        raise ValueError(f"x must be a BF16 array, got {getattr(x, 'dtype', type(x))}")
    return _gelu_tanh(x)
