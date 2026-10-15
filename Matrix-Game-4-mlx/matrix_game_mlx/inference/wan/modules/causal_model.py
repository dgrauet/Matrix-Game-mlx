"""Causal Wan DiT with rolling KV cache (MLX port of matrix_game/inference/wan/modules/causal_model.py).

Only the KV-cache inference path is ported. Deviation: the RoPE complex product is computed in
float32 (upstream float64; MLX has no float64 on GPU). Tables are built in float64 (model.rope_params).
"""
import mlx.core as mx

from matrix_game_mlx.inference.wan.modules.model import _apply_rope_rotation


def _check_rope_len(freqs, need: int) -> None:
    """Raise ValueError if the RoPE table has fewer than `need` rows."""
    rows = freqs[0].shape[0]
    if need > rows:
        raise ValueError(f"RoPE table has {rows} rows, position {need - 1} requested; raise rope_max_len")


def _split(freqs, c: int):
    """Split the (cos, sin) tables along the channel-pair axis into (time, height, width) parts."""
    c_t, c_h = c - 2 * (c // 3), c // 3
    cos, sin = freqs
    return ((cos[:, :c_t], sin[:, :c_t]), (cos[:, c_t:c_t + c_h], sin[:, c_t:c_t + c_h]),
            (cos[:, c_t + c_h:], sin[:, c_t + c_h:]))


def _rope_rows(x: mx.array, grid_sizes, freqs, time_rows) -> mx.array:
    """Shared body of causal_rope_apply / rope_apply_frames: time_rows(i, f) -> (cos_t, sin_t) [f, c_t]."""
    n, c = x.shape[2], x.shape[3] // 2
    (_, _), (ch, sh), (cw, sw) = _split(freqs, c)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes):
        seq_len = f * h * w
        ct, st = time_rows(i, f)

        def grid(t, axis_shape):
            return mx.broadcast_to(t.reshape(axis_shape), (f, h, w, t.shape[-1]))

        cos_i = mx.concatenate([grid(ct, (f, 1, 1, -1)), grid(ch[:h], (1, h, 1, -1)), grid(cw[:w], (1, 1, w, -1))],
                               axis=-1).reshape(seq_len, 1, -1)
        sin_i = mx.concatenate([grid(st, (f, 1, 1, -1)), grid(sh[:h], (1, h, 1, -1)), grid(sw[:w], (1, 1, w, -1))],
                               axis=-1).reshape(seq_len, 1, -1)
        x_i = _apply_rope_rotation(x[i, :seq_len].astype(mx.float32).reshape(seq_len, n, -1, 2), cos_i, sin_i)
        output.append(mx.concatenate([x_i, x[i, seq_len:].astype(mx.float32)], axis=0))
    return mx.stack(output).astype(x.dtype)


def causal_rope_apply(x: mx.array, grid_sizes, freqs, start_frame: int = 0) -> mx.array:
    """Apply 3D RoPE with the frame axis offset by `start_frame`; returns x.dtype."""
    _check_rope_len(freqs, int(start_frame) + max(g[0] for g in grid_sizes))
    (ct, st), _, _ = _split(freqs, x.shape[3] // 2)
    return _rope_rows(x, grid_sizes, freqs, lambda i, f: (ct[start_frame:start_frame + f], st[start_frame:start_frame + f]))


def rope_apply_frames(x: mx.array, grid_sizes, freqs, frame_pos) -> mx.array:
    """Apply 3D RoPE with explicit (non-negative) per-frame positions; returns x.dtype.

    frame_pos is 1-D and shared by all samples; its length must equal f of every grid entry.
    """
    pos = [int(p) for p in frame_pos]
    if not pos:
        raise ValueError("frame_pos must not be empty")
    for f, _, _ in grid_sizes:
        if len(pos) != f:
            raise ValueError(f"frame_pos has {len(pos)} entries but the grid has {f} frames")
    if min(pos) < 0:
        raise NotImplementedError("negative RoPE positions are only used by the kv_mem path (not ported)")
    _check_rope_len(freqs, max(pos) + 1)
    (ct, st), _, _ = _split(freqs, x.shape[3] // 2)
    idx = mx.array(pos)
    return _rope_rows(x, grid_sizes, freqs, lambda i, f: (ct[idx], st[idx]))
