"""Causal Wan DiT with rolling KV cache (MLX port of matrix_game/inference/wan/modules/causal_model.py).

Only the KV-cache inference path is ported. The RoPE complex product runs in float64 on the MLX CPU stream, as
upstream (MLX has no float64 on GPU); tables are float64 (model.rope_params).
"""
import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from matrix_game_mlx.inference.wan.modules.attention import attention
from matrix_game_mlx.inference.wan.modules.model import (
    WAN_CROSSATTENTION_CLASSES, WanLayerNorm, WanRMSNorm, _apply_rope_rotation, _on_cpu, gelu_tanh, rope_freqs, silu,
    sinusoidal_embedding_1d)


@_on_cpu
def linear_f64_acc(layer: nn.Linear, x: mx.array) -> mx.array:
    """Correctly rounded bf16 nn.Linear for the timestep MLP: float64 accumulation (bias included), rounded once.

    The MLP has one row per entry of t (M = B x F; 1 in the stream, which denoises one latent frame per call). At
    M=1 MLX's bf16 addmm (GEMV path) rounds the product to bf16 before adding the bias (2.9e-3 relative off torch).
    At any M an fp32 accumulation still flips ~1e-4 of the outputs depending on summation order, and e0 modulates
    every token of every block; float64 accumulation gives the correctly rounded value that torch's bf16 linear
    returns up to its own order-dependent flips.
    """
    return mx.addmm(layer.bias.astype(mx.float64), x.astype(mx.float64), layer.weight.astype(mx.float64).T).astype(x.dtype)


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
        x_i = _apply_rope_rotation(x[i, :seq_len].astype(mx.float64).reshape(seq_len, n, -1, 2), cos_i, sin_i)
        output.append(mx.concatenate([x_i, x[i, seq_len:].astype(mx.float64)], axis=0))
    return mx.stack(output).astype(x.dtype)


@_on_cpu
def causal_rope_apply(x: mx.array, grid_sizes, freqs, start_frame: int = 0) -> mx.array:
    """Apply 3D RoPE with the frame axis offset by `start_frame`; returns x.dtype."""
    _check_rope_len(freqs, int(start_frame) + max(g[0] for g in grid_sizes))
    (ct, st), _, _ = _split(freqs, x.shape[3] // 2)
    return _rope_rows(x, grid_sizes, freqs, lambda i, f: (ct[start_frame:start_frame + f], st[start_frame:start_frame + f]))


@_on_cpu
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


class CausalWanAttentionBlock(nn.Module):
    """Self-attention (KV cache) + cross-attention + FFN block with per-frame adaLN modulation."""

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0,
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim, self.ffn_dim, self.num_heads = dim, ffn_dim, num_heads
        self.local_attn_size, self.qk_norm, self.cross_attn_norm, self.eps = local_attn_size, qk_norm, cross_attn_norm, eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn_linear1 = nn.Linear(dim, ffn_dim)
        self.ffn_linear2 = nn.Linear(ffn_dim, dim)
        self.modulation = mx.zeros((1, 6, dim))

    def ffn(self, x):
        return self.ffn_linear2(gelu_tanh(self.ffn_linear1(x)))

    def __call__(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, block_mask=None, kv_cache=None,
                 crossattn_cache=None, current_start=0, cache_start=None, sink_rope_mode=None):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = mx.split(self.modulation[:, None] + e, 6, axis=2)
        def mod(h, shift, scale):
            return (h.reshape(h.shape[0], num_frames, frame_seqlen, -1) * (1 + scale) + shift).reshape(h.shape)
        def gate(h, g):
            return (h.reshape(h.shape[0], num_frames, frame_seqlen, -1) * g).reshape(h.shape)
        y = self.self_attn(mod(self.norm1(x), e[0], e[1]), seq_lens, grid_sizes, freqs, block_mask, kv_cache,
                           current_start, cache_start, sink_rope_mode=sink_rope_mode)
        x = x + gate(y, e[2])

        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(mod(self.norm2(x), e[3], e[4]))
            return x + gate(y, e[5])

        return cross_attn_ffn(x, context, context_lens, e, crossattn_cache)


class CausalHead(nn.Module):
    """Output head with per-frame shift/scale modulation."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim, self.out_dim, self.patch_size, self.eps = dim, out_dim, patch_size, eps
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = mx.zeros((1, 2, dim))

    def __call__(self, x, e):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = mx.split(self.modulation[:, None] + e, 2, axis=2)
        h = self.norm(x).reshape(x.shape[0], num_frames, frame_seqlen, -1) * (1 + e[1]) + e[0]
        return self.head(h)


class CausalWanModel(nn.Module):
    """Causal Wan DiT, KV-cache inference forward only (frames are [F, H, W, C] channels-last)."""

    def __init__(self, model_type="t2v", patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=2048, ffn_dim=8192,
                 freq_dim=256, text_dim=4096, out_dim=16, num_heads=16, num_layers=32, local_attn_size=-1,
                 sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6):
        super().__init__()
        assert model_type in ("t2v", "ti2v"), "i2v (CLIP branch) is not used by Matrix-Game 4.0"
        self.model_type, self.patch_size, self.text_len = model_type, tuple(patch_size), text_len
        self.in_dim, self.dim, self.ffn_dim, self.freq_dim = in_dim, dim, ffn_dim, freq_dim
        self.text_dim, self.out_dim, self.num_heads, self.num_layers = text_dim, out_dim, num_heads, num_layers
        self.local_attn_size, self.qk_norm, self.cross_attn_norm, self.eps = local_attn_size, qk_norm, cross_attn_norm, eps
        # Conv3d(kernel=stride=patch) stored as Linear over (C, t, h, w)-flattened patches (mlx-forge layout).
        self.patch_embedding = nn.Linear(in_dim * math.prod(patch_size), dim)
        self.text_embedding_linear1 = nn.Linear(text_dim, dim)
        self.text_embedding_linear2 = nn.Linear(dim, dim)
        self.time_embedding_linear1 = nn.Linear(freq_dim, dim)
        self.time_embedding_linear2 = nn.Linear(dim, dim)
        self.time_projection_linear1 = nn.Linear(dim, dim * 6)
        self.blocks = [CausalWanAttentionBlock("t2v_cross_attn", dim, ffn_dim, num_heads, local_attn_size, sink_size,
                                               qk_norm, cross_attn_norm, eps) for _ in range(num_layers)]
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self._rope_rows = 0
        self.set_rope_max_len(1024)

    def set_rope_max_len(self, n: int) -> None:
        n = int(n)
        if n <= self._rope_rows:
            return
        self._freqs = rope_freqs(n, self.dim // self.num_heads)
        self._rope_rows = n

    @property
    def freqs(self):
        return self._freqs

    def __call__(self, x, t, context, seq_len, kv_cache=None, crossattn_cache=None, current_start=0, cache_start=0,
                 sink_rope_mode=None):
        if kv_cache is None:
            raise NotImplementedError("only _forward_inference (KV cache) is ported")
        return self._forward_inference(x, t, context, seq_len, kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                                       current_start=current_start, cache_start=cache_start,
                                       sink_rope_mode=sink_rope_mode)

    def _forward_inference(self, x, t, context, seq_len, kv_cache=None, crossattn_cache=None, current_start=0,
                           cache_start=0, sink_rope_mode=None):
        pt, ph, pw = self.patch_size
        dtype = self.patch_embedding.weight.dtype
        tokens, grid_sizes = [], []
        for u in x:
            f, h, w, c = u.shape
            fp, hp, wp = f // pt, h // ph, w // pw
            grid_sizes.append([fp, hp, wp])
            u = u.reshape(fp, pt, hp, ph, wp, pw, c).transpose(0, 2, 4, 6, 1, 3, 5)
            tokens.append(self.patch_embedding(u.reshape(1, fp * hp * wp, c * pt * ph * pw).astype(dtype)))
        seq_lens = mx.array([u.shape[1] for u in tokens])
        assert max(u.shape[1] for u in tokens) <= seq_len
        x = mx.concatenate(tokens, axis=0)

        e = linear_f64_acc(self.time_embedding_linear2, silu(linear_f64_acc(
            self.time_embedding_linear1, sinusoidal_embedding_1d(self.freq_dim, t.reshape(-1)).astype(x.dtype))))
        e0 = linear_f64_acc(self.time_projection_linear1, silu(e)).reshape(*t.shape, 6, self.dim)

        ctx = context if isinstance(context, mx.array) else mx.stack(list(context))
        if ctx.shape[1] < self.text_len:
            ctx = mx.concatenate([ctx, mx.zeros((ctx.shape[0], self.text_len - ctx.shape[1], ctx.shape[2]), ctx.dtype)], 1)
        context = self.text_embedding_linear2(gelu_tanh(self.text_embedding_linear1(ctx.astype(dtype))))

        for block_index, block in enumerate(self.blocks):
            x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, None, None,
                      kv_cache=kv_cache[block_index], crossattn_cache=crossattn_cache[block_index],
                      current_start=current_start, cache_start=cache_start, sink_rope_mode=sink_rope_mode)

        x = self.head(x, e.reshape(*t.shape, 1, self.dim))
        return mx.stack(self.unpatchify(x.reshape(x.shape[0], -1, x.shape[-1]), grid_sizes))

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        pt, ph, pw = self.patch_size
        out = []
        for i, (fp, hp, wp) in enumerate(grid_sizes):
            u = x[i, :fp * hp * wp].reshape(fp, hp, wp, pt, ph, pw, c).transpose(0, 3, 1, 4, 2, 5, 6)
            out.append(u.reshape(fp * pt, hp * ph, wp * pw, c))
        return out
