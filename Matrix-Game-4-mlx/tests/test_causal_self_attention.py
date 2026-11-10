import numpy as np
import pytest
import torch
import mlx.core as mx
from tests.conftest import import_upstream, assert_close

from matrix_game_mlx.inference.wan.modules.causal_model import CausalWanSelfAttention
from tests.test_causal_rope import _freqs

up_c = import_upstream("matrix_game.inference.wan.modules.causal_model")


def _copy_attn(a, b):
    for name in ("q", "k", "v", "o"):
        getattr(a, name).weight = mx.array(getattr(b, name).weight.detach().numpy())
        getattr(a, name).bias = mx.array(getattr(b, name).bias.detach().numpy())
    a.norm_q.weight = mx.array(b.norm_q.weight.detach().numpy())
    a.norm_k.weight = mx.array(b.norm_k.weight.detach().numpy())


@pytest.mark.parametrize("sink_rope_mode", ["fixed", "absolute", "future"])
def test_kv_cache_rollout_30_chunks(sink_rope_mode):
    torch.manual_seed(0)
    dim, heads, W, S, grid = 128, 2, 4, 1, [[1, 2, 3]]
    fsl = 6
    b = up_c.CausalWanSelfAttention(dim, heads, W, S, True, 1e-6).eval()
    torch.nn.init.normal_(b.norm_q.weight); torch.nn.init.normal_(b.norm_k.weight)
    a = CausalWanSelfAttention(dim, heads, W, S, True, 1e-6)
    _copy_attn(a, b)
    fm, fp = _freqs(8192, dim // heads)
    cache_b = {"k": torch.zeros(1, W * fsl, heads, dim // heads), "v": torch.zeros(1, W * fsl, heads, dim // heads),
               "global_end_index": torch.tensor([0]), "local_end_index": torch.tensor([0]),
               "k_sink_raw": torch.zeros(1, S * fsl, heads, dim // heads)}
    cache_a = {"k": mx.zeros((1, W * fsl, heads, dim // heads)), "v": mx.zeros((1, W * fsl, heads, dim // heads)),
               "global_end_index": 0, "local_end_index": 0, "k_sink_raw": mx.zeros((1, S * fsl, heads, dim // heads))}
    rng = np.random.default_rng(0)
    for chunk in range(30):
        # each chunk is written twice like the stream: denoise pass then clean pass at the same start, with a
        # different input so the clean pass must overwrite what the denoise pass wrote
        for pass_name in ("denoise", "clean"):
            x = rng.standard_normal((1, fsl, dim)).astype(np.float32)
            with torch.no_grad():
                ref = b(torch.from_numpy(x), torch.tensor([fsl]), torch.tensor(grid), fp, None,
                        kv_cache=cache_b, current_start=chunk * fsl, sink_rope_mode=sink_rope_mode)
            out = a(mx.array(x), mx.array([fsl]), grid, fm, None, kv_cache=cache_a,
                    current_start=chunk * fsl, sink_rope_mode=sink_rope_mode)
            msg = f"chunk {chunk} {pass_name}"
            assert_close(out, ref, atol=1e-4, msg=msg)
            assert cache_a["global_end_index"] == int(cache_b["global_end_index"]), msg
            assert cache_a["local_end_index"] == int(cache_b["local_end_index"]), msg
            for name in ("k", "v", "k_sink_raw"):
                assert_close(cache_a[name], cache_b[name], atol=1e-4, msg=f"{msg} {name}")


def test_native_future_sink_keys_are_rotated_at_rope_frame_plus_offset(monkeypatch):
    from matrix_game_mlx.inference.sink_policy import FUTURE_SINK_OFFSET
    from matrix_game_mlx.inference.wan.modules import causal_model
    dim, heads, W, S, fsl, grid = 128, 2, 4, 1, 6, [[1, 2, 3]]
    attn = CausalWanSelfAttention(dim, heads, W, S, True, 1e-6)
    fm, _ = _freqs(8192, dim // heads)
    rng = np.random.default_rng(0)
    cache = {"k": mx.zeros((1, W * fsl, heads, dim // heads)), "v": mx.zeros((1, W * fsl, heads, dim // heads)),
             "global_end_index": 0, "local_end_index": 0, "k_sink_raw": mx.zeros((1, S * fsl, heads, dim // heads))}
    seen, real = [], causal_model.rope_apply_frames

    def spy(x, grid_sizes, freqs, frame_pos):
        seen.append(list(frame_pos))
        return real(x, grid_sizes, freqs, frame_pos)
    monkeypatch.setattr(causal_model, "rope_apply_frames", spy)
    for chunk in range(W + 3):
        seen.clear()
        x = mx.array(rng.standard_normal((1, fsl, dim)).astype(np.float32))
        attn(x, mx.array([fsl]), grid, fm, None, kv_cache=cache, current_start=chunk * fsl, sink_rope_mode="future")
        # one call re-ropes the sink once the window is full (chunk > W)
        assert seen == ([[chunk + FUTURE_SINK_OFFSET]] if chunk > W else []), chunk
