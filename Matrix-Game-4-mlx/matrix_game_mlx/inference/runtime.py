"""Ring KV runtime for Apple silicon (MLX port of matrix_game/inference/runtime.py).

`CastleRuntime` takes over the DiT block loop (hooked in `CausalWanModel._forward_inference`, as upstream): it plans
every layer's KV cache as a ring (`plan_cache(..., ring=True)`), builds the RoPE rotations once per call and applies
them with the GPU kernel (`rope_kernel.apply_causal_rope`, bitwise equal to the float64 CPU path), then runs
`run_blocks`. Only the bounded, batch-1, chunk-1, bf16 inference path is supported; anything else is a ValueError
(never a silent fallback to the native loop). The unfused `run_blocks` composes the native block pieces with the
native rounding, so the only numeric difference from `native` is the attention key order (ring slots -> SDPA
summation order).

Planner part: `plan_cache` (upstream semantics on MLX caches whose end indices are Python ints, a documented
deviation from upstream's 1-element tensors) plus `to_linear` / `from_linear`, which convert between the ring slot
order and the native linear order (sink frame, then the rolling window oldest to newest) so the parity suite can
compare against torch fixtures.

Deviations from upstream: no CUDA graphs (`prime`, `graph`) and no varlen `cu_seqlens` in `Plan`; the only mode is
`ring-fused`; caches are also required to be bfloat16; every layer is validated before any cache is planned (upstream
plans layer by layer and can leave earlier layers planned when a later one fails).
"""
from dataclasses import dataclass

import mlx.core as mx


def plan_cache(cache: dict, current_start: int, tokens: int, window: int, ring: bool = False) -> tuple:
    """Plan the KV slot of the frame at `current_start`; return (start, end) of the slot to write.

    Never evicts on same-position rewrites. A ring eviction copies exactly one frame (the previous tail) into the
    oldest history slot; linear eviction shifts the window left (native layout). Mutates `cache` (k, v rebound).
    """
    old_global, old_local = int(cache["global_end_index"]), int(cache["local_end_index"])
    current_end = current_start + tokens
    if current_end not in (old_global, old_global + tokens):
        raise ValueError("Only sequential writes and same-position rewrites are supported")
    evicted = 0
    if current_end > old_global and old_local + tokens > window * tokens:
        evicted = tokens
        for name in ("k", "v"):
            if ring:
                if window > 2:
                    slot = 1 + (current_start // tokens - window) % (window - 2)
                    cache[name][:, slot * tokens:(slot + 1) * tokens] = cache[name][:, -tokens:]
            else:
                cache[name][:, tokens:old_local - tokens] = cache[name][:, 2 * tokens:old_local]
    end = old_local + current_end - old_global - evicted
    if end < tokens or end > window * tokens:
        raise ValueError("Invalid bounded KV range")
    cache["global_end_index"] = current_end
    cache["local_end_index"] = end
    return end - tokens, end


def _ring_perm(cache: dict, tokens: int, window: int):
    """Frame permutation `perm` with linear_slot[i] = ring_slot[perm[i]], or None when the orders coincide.

    Slot 0 is the sink and slot window-1 the newest frame in both layouts. With F the newest frame (F >= window,
    i.e. at least one eviction), history frame g in F-window+2 .. F-1 sits in ring slot 1 + (g + 1 - window) % (window - 2)
    (it was copied there when frame g + 1 was planned); the linear slot of g is g - F + window - 1.
    """
    newest = int(cache["global_end_index"]) // tokens - 1
    if window <= 2 or newest < window:
        return None
    perm = list(range(window))
    for g in range(newest - window + 2, newest):
        perm[g - newest + window - 1] = 1 + (g + 1 - window) % (window - 2)
    return perm


def _permute(cache: dict, tokens: int, window: int, inverse: bool) -> dict:
    out = dict(cache)
    perm = _ring_perm(cache, tokens, window)
    if perm is None:
        return out
    if inverse:
        inv = [0] * window
        for i, p in enumerate(perm):
            inv[p] = i
        perm = inv
    idx = mx.array(perm)
    for name in ("k", "v"):
        a = cache[name]
        frames = a.reshape(a.shape[0], window, tokens, *a.shape[2:])
        out[name] = mx.take(frames, idx, axis=1).reshape(a.shape)
    return out


def to_linear(cache: dict, tokens: int, window: int) -> dict:
    """Ring cache -> new cache in native linear order (other keys copied). Depends only on the end indices."""
    return _permute(cache, tokens, window, inverse=False)


def from_linear(cache: dict, tokens: int, window: int) -> dict:
    """Native linear cache -> new cache in ring order (inverse of `to_linear`)."""
    return _permute(cache, tokens, window, inverse=True)


@dataclass
class Plan:
    block: object
    cache: dict
    cross: dict
    start: int
    end: int


def _modulate(h, shift, scale, frames, tokens):
    """Native `CausalWanAttentionBlock` mod(): per-frame adaLN shift/scale, same expression and rounding."""
    return (h.reshape(h.shape[0], frames, tokens, -1) * (1 + scale) + shift).reshape(h.shape)


def _gate(h, g, frames, tokens):
    """Native `CausalWanAttentionBlock` gate()."""
    return (h.reshape(h.shape[0], frames, tokens, -1) * g).reshape(h.shape)


def _qk_norm(norm, h, fused):
    """Native `WanRMSNorm` (q / k), or the fused kernel with the same rounding."""
    if fused:
        from matrix_game_mlx.inference.kernels import rms_norm
        return rms_norm(h, norm.weight, norm.eps)
    return norm(h)


def run_blocks(plans, x, e, rotations, sink_rotations, fused=False):
    """Run the DiT blocks over planned caches (upstream `run_blocks`).

    Unfused: exactly the native `CausalWanAttentionBlock` arithmetic, with RoPE from the GPU kernel and attention
    over the cache in ring order (`cache[:, :end]`, or the re-roped sink + `cache[:, tokens:end]` in fixed-sink
    steady state). Fused (upstream's structure): the LayerNorm + modulation, gate + residual (+ affine LayerNorm)
    chains, the q / k RMSNorms and the FFN activation run as single kernels (`kernels.py`) with the native rounding;
    cross-attention is inlined as upstream does (same projections, norms and attention call as the native module).
    """
    from matrix_game_mlx.inference.rope_kernel import apply_causal_rope
    from matrix_game_mlx.inference.wan.modules.attention import attention
    if fused:
        from matrix_game_mlx.inference.kernels import gate_ln_affine, gate_residual, gelu_tanh, ln_modulate
    frames, tokens = e.shape[1], x.shape[1] // e.shape[1]
    for p in plans:
        block, cache = p.block, p.cache
        if fused:
            z, _ = ln_modulate(x, e, block.modulation, block.norm1.eps, 0, 1)
        else:
            mod = mx.split(block.modulation[:, None] + e, 6, axis=2)
            z = _modulate(block.norm1(x), mod[0], mod[1], frames, tokens)
        attn = block.self_attn
        b, s, n, d = z.shape[0], z.shape[1], attn.num_heads, attn.head_dim
        q = _qk_norm(attn.norm_q, attn.q(z), fused).reshape(b, s, n, d)
        k = _qk_norm(attn.norm_k, attn.k(z), fused).reshape(b, s, n, d)
        v = attn.v(z).reshape(b, s, n, d)
        q = apply_causal_rope(q, rotations).astype(v.dtype)
        cache["k"][:, p.start:p.end] = apply_causal_rope(k, rotations).astype(v.dtype)
        cache["v"][:, p.start:p.end] = v
        if p.start == 0:
            cache["k_sink_raw"] = k
        if sink_rotations is None:
            keys, values = cache["k"][:, :p.end], cache["v"][:, :p.end]
        else:
            sink = apply_causal_rope(cache["k_sink_raw"], sink_rotations).astype(v.dtype)
            keys = mx.concatenate([sink, cache["k"][:, tokens:p.end]], axis=1)
            values = mx.concatenate([cache["v"][:, :tokens], cache["v"][:, tokens:p.end]], axis=1)
        y = attn.o(attention(q, keys, values).reshape(b, s, n * d))
        if fused:
            normed, x = gate_ln_affine(x, y, e, block.modulation, 2, block.norm3.weight, block.norm3.bias,
                                       block.norm3.eps)
            cross = block.cross_attn
            n, d = cross.num_heads, cross.head_dim
            q = _qk_norm(cross.norm_q, cross.q(normed), fused).reshape(b, s, n, d)
            y = cross.o(attention(q, p.cross["k"], p.cross["v"]).reshape(b, s, n * d))
            z, x = ln_modulate(x, e, block.modulation, block.norm2.eps, 3, 4, residual=y)
            x = gate_residual(x, block.ffn_linear2(gelu_tanh(block.ffn_linear1(z))), e, block.modulation, 5)
        else:
            x = x + _gate(y, mod[2], frames, tokens)
            # Cross K/V were initialised by the planner, so the native module reads them from the cache.
            x = x + block.cross_attn(block.norm3(x), None, None, crossattn_cache=p.cross)
            y = block.ffn(_modulate(block.norm2(x), mod[3], mod[4], frames, tokens))
            x = x + _gate(y, mod[5], frames, tokens)
    return x


def _fusable(block, e) -> bool:
    """Whether the fused kernels can run `block` (checked for every layer before any cache is planned)."""
    from matrix_game_mlx.inference.wan.modules.model import WanLayerNorm, WanRMSNorm
    norms = (block.self_attn.norm_q, block.self_attn.norm_k, block.cross_attn.norm_q)
    if not (isinstance(block.norm3, WanLayerNorm) and block.norm3.elementwise_affine
            and all(isinstance(norm, WanRMSNorm) for norm in norms)):
        return False
    tensors = [e, block.modulation, block.norm3.weight, block.norm3.bias] + [norm.weight for norm in norms]
    return all(t.dtype == mx.bfloat16 for t in tensors)


class CastleRuntime:
    """Ring KV + GPU RoPE DiT runtime. Upstream's CUDA modes (kernels, cudagraph, ring, ring-graph) do not exist."""

    def __init__(self, mode):
        if mode != "ring-fused":
            raise ValueError("Unknown castle runtime")
        self.mode = mode
        self.ring = True
        self.fused = True

    def clear(self):
        """Drop per-session state (none yet: no graphs; kept for the upstream call sites)."""

    def run(self, model, x, e, grid_sizes, context, kv, xc, current_start, *, sink_rope_mode, unsupported=False):
        from matrix_game_mlx.inference.rope_kernel import build_causal_rope_rotations
        from matrix_game_mlx.inference.wan.modules.causal_model import _check_rope_len
        device = mx.default_device()
        if (unsupported or sink_rope_mode not in (None, "absolute", "fixed") or x.dtype != mx.bfloat16
                or device.type != mx.gpu):
            raise ValueError(f"Unsupported runtime: extra={unsupported}, sink={sink_rope_mode}, "
                             f"dtype={x.dtype}, device={device}")
        if len(grid_sizes) != 1 or x.shape[0] != 1 or grid_sizes[0][0] != 1 or e.shape[:2] != (1, 1):
            raise ValueError("Castle runtime requires batch1/chunk1")
        _, h, w = (int(g) for g in grid_sizes[0])
        tokens = h * w
        position = current_start // tokens
        if x.shape[1] != tokens or current_start % tokens or kv is None or xc is None:
            raise ValueError("Expected frame-aligned unpadded input and explicit caches")
        if len(kv) != len(model.blocks) or len(xc) != len(kv):
            raise ValueError("Cache layer count mismatch")
        window = model.blocks[0].self_attn.local_attn_size
        for block, cache in zip(model.blocks, kv):
            a = block.self_attn
            if (a.local_attn_size != window or window < 2 or a.sink_size != 1
                    or cache["k"].shape[1] != window * tokens
                    or "k_sink_raw" not in cache or cache["k_sink_raw"].shape[1] != tokens
                    or type(block.cross_attn).__name__ != "WanT2VCrossAttention"):
                raise ValueError("Expected identical bounded windows, one sink, and T2V cross attention")
            if any(cache[name].dtype != mx.bfloat16 for name in ("k", "v", "k_sink_raw")):
                raise ValueError("Expected bfloat16 KV caches")
            if self.fused and not _fusable(block, e):
                raise ValueError("Fused kernels expect bfloat16 modulation, an affine cross-attention LayerNorm "
                                 "and RMS q/k norms")
        _check_rope_len(model.freqs, position + 1)  # native's message for an exhausted RoPE table
        rotations = build_causal_rope_rotations(model.freqs, 1, h, w, position)
        steady = position > window
        fixed = steady and sink_rope_mode == "fixed"
        sink_rotations = build_causal_rope_rotations(model.freqs, 1, h, w, position - window) if fixed else None
        plans = []
        for block, cache, cross in zip(model.blocks, kv, xc):
            start, end = plan_cache(cache, current_start, tokens, window, self.ring)
            if start < 0 or end > window * tokens or (fixed and end != window * tokens):
                raise ValueError("Invalid bounded KV range")
            if not cross["is_init"]:
                a = block.cross_attn
                cross["k"] = a.norm_k(a.k(context)).reshape(1, -1, a.num_heads, a.head_dim)
                cross["v"] = a.v(context).reshape(1, -1, a.num_heads, a.head_dim)
                cross["is_init"] = True
            plans.append(Plan(block, cache, cross, start, end))
        return run_blocks(plans, x, e, rotations, sink_rotations, self.fused)


def install_runtime(generator, mode):
    """Install the DiT runtime for `mode` on `generator.model` ("native" removes it)."""
    if mode not in ("native", "ring-fused"):
        raise ValueError("Unknown DiT runtime")
    old = getattr(generator.model, "castle_runtime", None)
    if old is not None:
        old.clear()
    generator.model.castle_runtime = None if mode == "native" else CastleRuntime(mode)
