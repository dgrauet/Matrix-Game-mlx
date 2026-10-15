import numpy as np
import torch
import mlx.core as mx
from tests.conftest import import_upstream, assert_close

from matrix_game_mlx.inference.wan.modules import model as M
from matrix_game_mlx.inference.wan.modules.attention import attention

up = import_upstream("matrix_game.inference.wan.modules.model")


def _load_linear(dst, src):
    dst.weight = mx.array(src.weight.detach().numpy())
    dst.bias = mx.array(src.bias.detach().numpy())


def test_sinusoidal_embedding():
    pos = np.array([0.0, 250.0, 937.5, 1000.0], np.float32)
    assert_close(M.sinusoidal_embedding_1d(256, mx.array(pos)),
                 up.sinusoidal_embedding_1d(256, torch.from_numpy(pos)).float(), atol=1e-5, rtol=0)


def test_rope_params_long_table():
    cos, sin = M.rope_params(8192, 44)
    ref = up.rope_params(8192, 44)
    assert_close(cos, ref.real.float(), atol=1e-6, rtol=0)
    assert_close(sin, ref.imag.float(), atol=1e-6, rtol=0)


def test_norms():
    x = np.random.default_rng(0).standard_normal((2, 7, 64)).astype(np.float32)
    rms_a, rms_b = M.WanRMSNorm(64, eps=1e-6), up.WanRMSNorm(64, eps=1e-6)
    torch.nn.init.normal_(rms_b.weight)
    rms_a.weight = mx.array(rms_b.weight.detach().numpy())
    assert_close(rms_a(mx.array(x)), rms_b(torch.from_numpy(x)), atol=1e-5)
    ln_a, ln_b = M.WanLayerNorm(64, 1e-6, elementwise_affine=True), up.WanLayerNorm(64, 1e-6, elementwise_affine=True)
    torch.nn.init.normal_(ln_b.weight); torch.nn.init.normal_(ln_b.bias)
    ln_a.weight = mx.array(ln_b.weight.detach().numpy()); ln_a.bias = mx.array(ln_b.bias.detach().numpy())
    assert_close(ln_a(mx.array(x)), ln_b(torch.from_numpy(x)), atol=1e-5)


def test_cross_attention_with_cache():
    torch.manual_seed(0)
    dim, heads = 128, 4
    b = up.WanT2VCrossAttention(dim, heads, (-1, -1), True, 1e-6).eval()
    a = M.WanT2VCrossAttention(dim, heads, (-1, -1), True, 1e-6)
    for name in ("q", "k", "v", "o"):
        _load_linear(getattr(a, name), getattr(b, name))
    a.norm_q.weight = mx.array(b.norm_q.weight.detach().numpy())
    a.norm_k.weight = mx.array(b.norm_k.weight.detach().numpy())
    x = torch.randn(1, 40, dim); ctx = torch.randn(1, 16, dim)
    cache_a, cache_b = {"is_init": False}, {"k": None, "v": None, "is_init": False}
    with torch.no_grad():
        ref1 = b(x, ctx, None, crossattn_cache=cache_b)
        ref2 = b(x * 2, ctx * 0, None, crossattn_cache=cache_b)   # second call must reuse cached K/V
    out1 = a(mx.array(x.numpy()), mx.array(ctx.numpy()), None, crossattn_cache=cache_a)
    out2 = a(mx.array((x * 2).numpy()), mx.array((ctx * 0).numpy()), None, crossattn_cache=cache_a)
    assert cache_a["is_init"] is True and "k" in cache_a and "v" in cache_a
    assert_close(out1, ref1, atol=1e-4)
    assert_close(out2, ref2, atol=1e-4)


def test_cross_attention_bf16_dtype():
    a = M.WanT2VCrossAttention(128, 4)
    a.set_dtype(mx.bfloat16)
    out = a(mx.ones((1, 8, 128), mx.bfloat16), mx.ones((1, 4, 128), mx.bfloat16), None)
    assert out.dtype == mx.bfloat16


def test_norms_bf16_dtype():
    x = mx.ones((2, 7, 64), mx.bfloat16)
    rms = M.WanRMSNorm(64); rms.set_dtype(mx.bfloat16)
    ln = M.WanLayerNorm(64, elementwise_affine=True); ln.set_dtype(mx.bfloat16)
    assert rms(x).dtype == mx.bfloat16
    assert ln(x).dtype == mx.bfloat16


def test_rope_apply_matches_upstream_and_keeps_dtype():
    d, grids = 44, [[3, 4, 5]]
    seq = 3 * 4 * 5
    x = np.random.default_rng(0).standard_normal((1, seq + 6, 2, d)).astype(np.float32)
    cos, sin = M.rope_params(16, d)
    freqs = up.rope_params(16, d)
    out = M.rope_apply(mx.array(x), grids, (cos, sin))
    ref = up.rope_apply(torch.from_numpy(x), torch.tensor(grids), freqs)
    assert_close(out, ref, atol=1e-5)
    assert M.rope_apply(mx.array(x).astype(mx.bfloat16), grids, (cos, sin)).dtype == mx.bfloat16


def test_attention_matches_sdpa():
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal(s).astype(np.float32) for s in ((1, 10, 4, 32), (1, 30, 4, 32), (1, 30, 4, 32)))
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(torch.from_numpy(t).transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
    assert_close(attention(mx.array(q), mx.array(k), mx.array(v)), ref, atol=1e-5)


def _bf16_samples(n=200_000, scale=3.0):
    x = (np.random.default_rng(0).standard_normal(n) * scale).astype(np.float32)
    return mx.array(x).astype(mx.bfloat16), torch.from_numpy(x).bfloat16()


def test_bf16_activations_match_torch_opmath():
    """torch computes bf16 SiLU / tanh-GELU in float32 and rounds once (per-op bf16 rounding diverged on ~40 %)."""
    xm, xt = _bf16_samples()
    silu = np.array(M.silu(xm).astype(mx.float32))
    assert np.array_equal(silu, torch.nn.functional.silu(xt).float().numpy())
    gelu = np.array(M.gelu_tanh(xm).astype(mx.float32))
    ref = torch.nn.functional.gelu(xt, approximate="tanh").float().numpy()
    assert (gelu != ref).mean() < 0.02 and np.abs(gelu - ref).max() < 1e-4


def test_linear_fp32_acc_matches_torch_bf16_single_row():
    """Timestep MLP rows (M=1): bias added before the single bf16 rounding, as torch's bf16 nn.Linear."""
    import mlx.nn as nn
    from tests.conftest import rel_l2
    from matrix_game_mlx.inference.wan.modules.causal_model import linear_fp32_acc
    torch.manual_seed(0)
    ref_layer = torch.nn.Linear(256, 3072).bfloat16()
    x = torch.randn(1, 256).bfloat16()
    layer = nn.Linear(256, 3072)
    layer.weight = mx.array(ref_layer.weight.detach().float().numpy()).astype(mx.bfloat16)
    layer.bias = mx.array(ref_layer.bias.detach().float().numpy()).astype(mx.bfloat16)
    with torch.no_grad():
        ref = ref_layer(x).float().numpy()
    out = linear_fp32_acc(layer, mx.array(x.float().numpy()).astype(mx.bfloat16))
    assert out.dtype == mx.bfloat16
    assert rel_l2(out, ref) < 1e-6
