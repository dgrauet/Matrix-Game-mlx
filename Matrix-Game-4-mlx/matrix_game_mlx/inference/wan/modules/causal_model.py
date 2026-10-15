"""Causal Wan DiT with rolling KV cache (MLX port of matrix_game/inference/wan/modules/causal_model.py).

Only the KV-cache inference path is ported. Deviation: the RoPE complex product is computed in
float32 (upstream float64; MLX has no float64 on GPU). Tables are built in float64 (model.rope_params).
"""
import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from matrix_game_mlx.inference.wan.modules.attention import attention
from matrix_game_mlx.inference.wan.modules.model import WanRMSNorm, _apply_rope_rotation


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


class CausalWanSelfAttention(nn.Module):
    """Self-attention over a rolling KV cache with attention sink and optional fixed-sink RoPE.

    Deviation: kv_cache["global_end_index"] / ["local_end_index"] are Python ints (upstream: 1-element tensors).
    """

    def __init__(self, dim: int, num_heads: int, local_attn_size: int = -1, sink_size: int = 0,
                 qk_norm: bool = True, eps: float = 1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = None
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def __call__(self, x: mx.array, seq_lens, grid_sizes: list, freqs, block_mask=None, kv_cache: dict = None,
                 current_start: int = 0, cache_start: Optional[int] = None,
                 sink_rope_mode: Optional[str] = None) -> mx.array:
        """x [B, L, C] -> [B, L, C]; updates kv_cache in place (k, v, k_sink_raw, global/local_end_index)."""
        if kv_cache is None:
            raise NotImplementedError("only the KV-cache inference path is ported")
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q(x)).reshape(b, s, n, d)
        k = self.norm_k(self.k(x)).reshape(b, s, n, d)
        v = self.v(x).reshape(b, s, n, d)

        frame_seqlen = math.prod(grid_sizes[0][1:])
        current_start_frame = current_start // frame_seqlen
        rope_frame = current_start_frame
        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=rope_frame).astype(v.dtype)
        roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=rope_frame).astype(v.dtype)

        current_end = current_start + roped_query.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = roped_query.shape[1]
        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"]) and (
                num_new_tokens + kv_cache["local_end_index"] > kv_cache_size):
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
            src = slice(sink_tokens + num_evicted_tokens, sink_tokens + num_evicted_tokens + num_rolled_tokens)
            # MLX slice assignment is functional (the RHS is read from the previous array), so no clone is needed
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["k"][:, src]
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["v"][:, src]
            local_end_index = kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"] - num_evicted_tokens
        else:
            local_end_index = kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"]
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v
        if "k_sink_raw" in kv_cache and local_start_index < sink_tokens:
            n_sink_w = min(sink_tokens, local_end_index) - local_start_index
            kv_cache["k_sink_raw"][:, local_start_index:local_start_index + n_sink_w] = k[:, :n_sink_w]
        max_attention_size = local_end_index if self.local_attn_size == -1 else self.local_attn_size * frame_seqlen
        recent_start = local_end_index - (max_attention_size - sink_tokens)
        W_att = self.local_attn_size
        if sink_rope_mode not in (None, "absolute", "fixed"):
            raise ValueError(f"sink_rope_mode must be absolute or fixed, got {sink_rope_mode}")
        if sink_rope_mode == "fixed" and W_att != -1 and sink_tokens > 0 and rope_frame > W_att:
            assert "k_sink_raw" in kv_cache, "sink_rope_mode=fixed needs kv_cache['k_sink_raw']"
            assert recent_start >= sink_tokens, (recent_start, sink_tokens)
            sink_grid = [[self.sink_size, *grid_sizes[0][1:]]]
            sink_pos = [p + (rope_frame - W_att) for p in range(self.sink_size)]
            k_sink = rope_apply_frames(kv_cache["k_sink_raw"][:, :sink_tokens], sink_grid, freqs, sink_pos).astype(v.dtype)
            k_att = mx.concatenate([k_sink, kv_cache["k"][:, recent_start:local_end_index]], axis=1)
            v_att = mx.concatenate([kv_cache["v"][:, :sink_tokens], kv_cache["v"][:, recent_start:local_end_index]], axis=1)
        elif self.local_attn_size != -1 and sink_tokens > 0 and recent_start > sink_tokens:
            k_att = mx.concatenate([kv_cache["k"][:, :sink_tokens], kv_cache["k"][:, recent_start:local_end_index]], axis=1)
            v_att = mx.concatenate([kv_cache["v"][:, :sink_tokens], kv_cache["v"][:, recent_start:local_end_index]], axis=1)
        else:
            k_att = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
            v_att = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]
        x = attention(roped_query, k_att, v_att)
        kv_cache["global_end_index"] = current_end
        kv_cache["local_end_index"] = local_end_index
        return self.o(x.reshape(b, s, n * d))
