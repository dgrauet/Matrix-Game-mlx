import pytest
import torch
import mlx.core as mx
from mlx.utils import tree_flatten
from tests.conftest import import_upstream, assert_close, weights_dir

from matrix_game_mlx.inference.wan.modules.vae2_2 import WanVAE_, count_conv3d
from matrix_game_mlx.inference.decoder import IncrementalDecoder
from matrix_game_mlx.inference.wrappers import WanVAE22Wrapper

up_v = import_upstream("matrix_game.inference.wan.modules.vae2_2")
KW = dict(dim=32, z_dim=48, dim_mult=[1, 2, 4, 4], temperal_downsample=[False, True, True])


def _convert(sd):
    out = {}
    for k, v in sd.items():
        v = v.detach().float().numpy()
        if k.endswith(".weight") and v.ndim == 5:
            v = v.transpose(0, 2, 3, 4, 1)
        elif k.endswith(".weight") and v.ndim == 4:
            v = v.transpose(0, 2, 3, 1)
        elif k.endswith("gamma"):
            v = v.reshape(-1)
        out[k] = mx.array(v)
    return out


def _pair():
    torch.manual_seed(0)
    ref = up_v.WanVAE_(**KW).eval()
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.05)
    m = WanVAE_(**KW)
    m.load_weights(list(_convert(ref.state_dict()).items()), strict=True)
    return m, ref


def test_streaming_encode_equals_full_encode():
    m, ref = _pair()
    x = torch.rand(1, 3, 9, 64, 64) * 2 - 1                        # 1 + 4 + 4 frames
    scale = [torch.zeros(48), torch.ones(48)]
    with torch.no_grad():
        full = ref.encode(x, scale)                                # [1, 48, 3, 4, 4]
    feats = [None] * count_conv3d(m.encoder)
    wrapper = WanVAE22Wrapper.from_model(m, mean=[0.0] * 48, std=[1.0] * 48, dtype=mx.float32)
    xm = mx.array(x.permute(0, 2, 3, 4, 1).numpy())                # [B, T, H, W, C]
    chunks = [wrapper.encode_stream(xm[:, s:e], feats) for s, e in ((0, 1), (1, 5), (5, 9))]
    out = mx.concatenate(chunks, axis=1)                           # [1, 3, 48, 4, 4]
    assert_close(out, full.permute(0, 2, 1, 3, 4), atol=1e-4)


def test_incremental_decoder_equals_full_decode():
    m, ref = _pair()
    z = torch.randn(1, 48, 3, 4, 4)
    scale = [torch.zeros(48), torch.ones(48)]
    with torch.no_grad():
        full = ref.decode(z, scale).clamp(-1, 1)                   # [1, 3, 9, 64, 64]
    wrapper = WanVAE22Wrapper.from_model(m, mean=[0.0] * 48, std=[1.0] * 48, dtype=mx.float32)
    dec = IncrementalDecoder(wrapper)
    zm = mx.array(z.permute(0, 2, 1, 3, 4).numpy())                # [1, n, 48, h, w]
    frames = mx.concatenate([dec.push(zm[:, i:i + 1]) for i in range(3)], axis=0)   # [9, 3, 64, 64]
    assert_close(frames, full[0].permute(1, 0, 2, 3), atol=1e-4)


@pytest.mark.needs_weights
def test_wrapper_loads_real_shared_vae_strictly():
    from matrix_game_mlx.inference.loader import resolve_shared_files
    vae_path = resolve_shared_files(weights_dir())["vae.safetensors"]
    w = WanVAE22Wrapper(vae_path)
    params = tree_flatten(w.model.parameters())
    assert len(params) == 196
    assert all(v.dtype == mx.bfloat16 for _, v in params)


def test_rms_norm_and_silu_match_torch_bf16():
    """bf16 RMS_norm (F.normalize * scale * gamma) and SiLU round where torch does (fp32 norm / opmath)."""
    import numpy as np
    from matrix_game_mlx.inference.wan.modules.vae2_2 import RMS_norm, _SiLU_placeholder
    torch.manual_seed(0)
    ref = up_v.RMS_norm(160, images=False)
    torch.nn.init.normal_(ref.gamma, mean=1.0, std=0.2)
    x = (torch.randn(1, 160, 2, 8, 8) * 3).bfloat16()
    m = RMS_norm(160, images=False)
    m.gamma = mx.array(ref.gamma.detach().reshape(-1).numpy()).astype(mx.bfloat16)
    xm = mx.array(x.float().permute(0, 2, 3, 4, 1).numpy()).astype(mx.bfloat16)
    with torch.no_grad():
        want = ref.bfloat16()(x).float().permute(0, 2, 3, 4, 1).numpy()
        want_silu = torch.nn.functional.silu(x).float().permute(0, 2, 3, 4, 1).numpy()
    got = np.array(m(xm).astype(mx.float32))
    assert (got != want).mean() < 1e-3
    assert np.array_equal(np.array(_SiLU_placeholder()(xm).astype(mx.float32)), want_silu)
