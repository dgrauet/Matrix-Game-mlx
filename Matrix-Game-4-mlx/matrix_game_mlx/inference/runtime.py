"""Ring KV runtime for Apple silicon (MLX port of matrix_game/inference/runtime.py).

Planner part: `plan_cache` (upstream semantics on MLX caches whose end indices are Python ints, a documented
deviation from upstream's 1-element tensors) plus `to_linear` / `from_linear`, which convert between the ring slot
order and the native linear order (sink frame, then the rolling window oldest to newest) so the parity suite can
compare against torch fixtures.
"""
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
