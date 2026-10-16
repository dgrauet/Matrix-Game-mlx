"""TAEHV (taew2_2) MLX port vs upstream torch, with random weights (fp32) plus the real pack (bf16)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from tests.conftest import import_upstream, assert_close, weights_dir

try:
    up = import_upstream("matrix_game.inference.taehv")
except ImportError as e:                                       # upstream imports diffusers at module level
    pytest.skip(f"upstream taehv.py not importable: {e}", allow_module_level=True)

from matrix_game_mlx.inference import taehv as mt

H, W = 64, 96                                                  # 4 x 6 latents


def _convert(sd):
    out = {}
    for k, v in sd.items():
        v = v.detach().float().numpy()
        if k.endswith(".weight") and v.ndim == 4:
            v = v.transpose(0, 2, 3, 1)                        # (O, I, kh, kw) -> (O, kh, kw, I)
        out[k] = mx.array(v)
    return out


@pytest.fixture(scope="module")
def ref():
    torch.manual_seed(0)
    return up._TAEHVCore(checkpoint_path=None, arch_name="taew2_2").eval()


@pytest.fixture(scope="module")
def core(ref):
    m = mt._TAEHVCore(checkpoint_path=None, arch_name="taew2_2")
    m.load_weights(list(_convert(ref.state_dict()).items()), strict=True)
    return m


def _nthwc(x):                                                 # torch NTCHW -> mx NTHWC
    return mx.array(x.permute(0, 1, 3, 4, 2).numpy())


def _ntchw(x):                                                 # mx NTHWC -> numpy NTCHW
    return np.array(x).transpose(0, 1, 4, 2, 3)


def test_pixel_shuffle_helpers_exact():
    x = torch.randn(2, 3, 12, 8, 6)                            # NTCHW
    xm = _nthwc(x)
    un = torch.nn.functional.pixel_unshuffle(x.flatten(0, 1), 2).reshape(2, 3, 12 * 4, 4, 3)
    assert np.array_equal(_ntchw(mt.pixel_unshuffle(xm, 2)), un.numpy())
    y = torch.randn(2, 3, 12, 4, 3)
    sh = torch.nn.functional.pixel_shuffle(y.flatten(0, 1), 2).reshape(2, 3, 3, 8, 6)
    assert np.array_equal(_ntchw(mt.pixel_shuffle(_nthwc(y), 2)), sh.numpy())


def _nhwc(x):
    return mx.array(x.permute(0, 2, 3, 1).numpy())


def _nchw(x):
    return np.array(x).transpose(0, 3, 1, 2)


def _conv_pair(ref_mod, mod):
    mod.load_weights(list(_convert(ref_mod.state_dict()).items()), strict=True)


def test_memblock():
    torch.manual_seed(1)
    r = up.MemBlock(8, 16).eval()
    m = mt.MemBlock(8, 16)
    _conv_pair(r, m)
    x, past = torch.randn(3, 8, 6, 5), torch.randn(3, 8, 6, 5)
    with torch.no_grad():
        want = r(x, past)
    assert_close(_nchw(m(_nhwc(x), _nhwc(past))), want, atol=1e-4)


def test_memblock_identity_skip():
    torch.manual_seed(1)
    r = up.MemBlock(8, 8).eval()
    m = mt.MemBlock(8, 8)
    _conv_pair(r, m)
    x, past = torch.randn(2, 8, 6, 5), torch.randn(2, 8, 6, 5)
    with torch.no_grad():
        want = r(x, past)
    assert_close(_nchw(m(_nhwc(x), _nhwc(past))), want, atol=1e-4)


@pytest.mark.parametrize("stride", [1, 2])
def test_tpool_tgrow(stride):
    torch.manual_seed(2)
    rp, rg = up.TPool(8, stride).eval(), up.TGrow(8, stride).eval()
    mp, mg = mt.TPool(8, stride), mt.TGrow(8, stride)
    _conv_pair(rp, mp)
    _conv_pair(rg, mg)
    x = torch.randn(6, 8, 4, 3)                                # 6 frames (batch-major N*T)
    with torch.no_grad():
        want_p, want_g = rp(x), rg(x)
    assert_close(_nchw(mp(_nhwc(x))), want_p, atol=1e-4)
    assert_close(_nchw(mg(_nhwc(x))), want_g, atol=1e-4)


def test_clamp_and_upsample():
    x = torch.randn(2, 4, 5, 3) * 5
    assert_close(_nchw(mt.Clamp()(_nhwc(x))), up.Clamp()(x), atol=1e-6)
    want = torch.nn.Upsample(scale_factor=2)(x)
    assert_close(_nchw(mt.Upsample(2)(_nhwc(x))), want, atol=0)


def _video(t, seed=3):
    torch.manual_seed(seed)
    return torch.rand(1, t, 3, H, W)                           # NTCHW in [0, 1]


def test_encode_decode_parallel_match_torch(ref, core):
    v = _video(8)
    with torch.no_grad():
        lat = ref.encode_video(v, parallel=True, show_progress_bar=False)
    got = core.encode_video(_nthwc(v), parallel=True, show_progress_bar=False)
    assert lat.shape == (1, 2, 48, 4, 6)
    assert float(lat.std()) > 0.01                             # non-degenerate reference
    assert_close(_ntchw(got), lat, atol=1e-4)
    z = torch.randn(1, 2, 48, 4, 6)
    with torch.no_grad():
        dec = ref.decode_video(z, parallel=True, show_progress_bar=False)
    got = core.decode_video(_nthwc(z), parallel=True, show_progress_bar=False)
    assert dec.shape == (1, 5, 3, H, W)
    assert float(dec.std()) > 0.01
    assert_close(_ntchw(got), dec, atol=1e-4)


def test_sequential_equals_parallel(core):
    v = _nthwc(_video(8))
    z = mx.array(np.random.RandomState(0).randn(1, 2, 4, 6, 48).astype(np.float32))
    a = core.encode_video(v, parallel=False, show_progress_bar=False)
    b = core.encode_video(v, parallel=True, show_progress_bar=False)
    assert_close(a, b, atol=1e-4)
    a = core.decode_video(z, parallel=False, show_progress_bar=False)
    b = core.decode_video(z, parallel=True, show_progress_bar=False)
    assert_close(a, b, atol=1e-4)


@pytest.fixture(scope="module")
def adapters(ref, tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "taew2_2.pth"
    torch.save(ref.state_dict(), path)
    r = up.TAEHV(str(path)).eval()
    m = mt.TAEHV(str(path), dtype=mx.float32, load=False)
    m.core.load_weights(list(_convert(ref.state_dict()).items()), strict=True)
    return r, m


def _nc(x):                                                    # torch NCTHW -> mx NCTHW
    return mx.array(x.numpy())


def test_adapter_rejects_other_arch():
    with pytest.raises(ValueError):
        mt.TAEHV("taehv.pth")


def test_adapter_requires_safetensors_unless_load_false(tmp_path):
    with pytest.raises(ValueError, match="load=False"):
        mt.TAEHV(str(tmp_path / "taew2_2.pth"))


@pytest.mark.parametrize("name", ["taehv1_5", "taeh3", "taeltx", "taecvx", "taew2_2_super"])
def test_core_rejects_dropped_archs(name):
    with pytest.raises(NotImplementedError):
        mt._TAEHVCore(checkpoint_path=None, arch_name=name)


def test_adapter_streaming_matches_torch(adapters):
    r, m = adapters
    torch.manual_seed(4)
    x = torch.rand(1, 3, 17, H, W) * 2 - 1
    chunks = [x[:, :, :1]] + [x[:, :, 1 + 4 * i: 5 + 4 * i] for i in range(4)]
    r.clear_cache()
    m.clear_cache()
    lats_t, lats_m = [], []
    with torch.no_grad():
        for c in chunks:
            lats_t.append(r.encode_stream(c).latent_dist.mode())
    for c in chunks:
        lats_m.append(m.encode_stream(_nc(c)))
    for lt, lm in zip(lats_t, lats_m):
        assert tuple(lm.shape) == tuple(lt.shape) == (1, 48, 1, 4, 6)
        assert_close(lm, lt, atol=1e-4)
    counts = []
    with torch.no_grad():
        for lt, lm in zip(lats_t, lats_m):
            want = r.decode_stream(lt).sample
            got = m.decode_stream(lm)
            counts.append(got.shape[2])
            assert tuple(got.shape) == tuple(want.shape)
            assert_close(got, want, atol=1e-4)
    assert counts == [1, 4, 4, 4, 4]


def test_adapter_encode_decode_match_torch(adapters):
    r, m = adapters
    r.clear_cache()
    m.clear_cache()
    torch.manual_seed(5)
    x = torch.rand(1, 3, 9, H, W) * 2 - 1
    with torch.no_grad():
        lat = r.encode(x).latent_dist.mode()
    got = m.encode(_nc(x))
    assert tuple(got.shape) == tuple(lat.shape) == (1, 48, 3, 4, 6)
    assert_close(got, lat, atol=1e-4)
    with torch.no_grad():
        want = r.decode(lat).sample
    assert_close(m.decode(got), want, atol=1e-4)


def test_encode_chunked_matches_single_pass(adapters):
    _, m = adapters
    torch.manual_seed(6)
    x = _nc(torch.rand(1, 3, 17, H, W) * 2 - 1)
    full = m.encode(x)
    chunked = m._encode_chunked(x, chunk_frames=8).transpose(0, 4, 1, 2, 3)
    assert_close(chunked, full, atol=1e-4)


def test_clear_cache_resets(adapters):
    _, m = adapters
    torch.manual_seed(7)
    x = _nc(torch.rand(1, 3, 5, H, W) * 2 - 1)
    z = mx.array(np.random.RandomState(1).randn(1, 48, 1, 4, 6).astype(np.float32))
    m.clear_cache()
    e1, d1 = m.encode_stream(x[:, :, :1]), m.decode_stream(z)
    m.encode_stream(x[:, :, 1:5])
    m.decode_stream(z)
    m.clear_cache()
    e2, d2 = m.encode_stream(x[:, :, :1]), m.decode_stream(z)
    assert_close(e1, e2, atol=0)
    assert_close(d1, d2, atol=0)


@pytest.mark.needs_weights
def test_real_pack_loads_strict_bf16():
    m = mt.TAEHV.from_pack(weights_dir() / "taehv.safetensors")
    params = tree_flatten(m.parameters())
    assert params and all(v.dtype == mx.bfloat16 for _, v in params)
    shapes = [tuple(l.conv.weight.shape) for l in m.core.decoder if isinstance(l, mt.TGrow)]
    assert shapes == [(256, 1, 1, 256), (256, 1, 1, 128), (128, 1, 1, 64)]


@pytest.mark.needs_weights
def test_real_decoder8_bf16_conv_matches_torch_bf16():
    """Regression: decoder.8 (3x3, 256->128; arbitrary 120x216 test input) through the MLX Metal bf16 conv2d kernel was 3e-2 rel L2 off
    fp64 (torch CPU bf16: 1.7e-3); TAEHV convs now accumulate in float32 and round once, like torch."""
    from tests.conftest import rel_l2
    m = mt.TAEHV.from_pack(weights_dir() / "taehv.safetensors")
    layer = m.core.decoder[8]
    assert isinstance(layer, mt.Conv2d) and tuple(layer.weight.shape) == (128, 3, 3, 256)
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 256, 120, 216), dtype=np.float32)).to(torch.bfloat16)
    w = torch.from_numpy(np.array(layer.weight.astype(mx.float32)).transpose(0, 3, 1, 2)).to(torch.bfloat16)
    want = torch.nn.functional.conv2d(x, w, padding=1).float().numpy()                 # native torch CPU bf16
    got = layer(mx.array(x.float().numpy()).astype(mx.bfloat16).transpose(0, 2, 3, 1))
    assert got.dtype == mx.bfloat16
    err = rel_l2(np.array(got.astype(mx.float32)).transpose(0, 3, 1, 2), want)
    assert err < 1e-3, err
