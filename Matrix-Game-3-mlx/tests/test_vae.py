"""Validation tests for VAE (vae2_2.py).

Compares MLX channels-last implementation against PyTorch channels-first reference.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib.util
_conftest_spec = importlib.util.spec_from_file_location(
    "conftest", os.path.join(os.path.dirname(__file__), "conftest.py"))
_conftest = importlib.util.module_from_spec(_conftest_spec)
_conftest_spec.loader.exec_module(_conftest)
assert_close = _conftest.assert_close
torch_to_mlx = _conftest.torch_to_mlx
mlx_to_torch = _conftest.mlx_to_torch

import mlx.core as mx
import mlx.nn as mlx_nn
import torch
import torch.nn as pt_nn
import torch.nn.functional as F

# Import MLX implementations
from wan.modules.vae2_2 import (
    CausalConv3d as MlxCausalConv3d,
    RMS_norm as MlxRMS_norm,
    ResidualBlock as MlxResidualBlock,
    AttentionBlock as MlxAttentionBlock,
    Encoder3d as MlxEncoder3d,
    Decoder3d as MlxDecoder3d,
    WanVAE_ as MlxWanVAE_,
    Resample as MlxResample,
    patchify as mlx_patchify,
    unpatchify as mlx_unpatchify,
    _count_conv3d,
)

# ============================================================================
# PyTorch reference implementations (inline to avoid dependency issues)
# ============================================================================

CACHE_T = 2


class PtCausalConv3d(pt_nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2], self.padding[2],
            self.padding[1], self.padding[1],
            2 * self.padding[0], 0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class PtRMS_norm(pt_nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = pt_nn.Parameter(torch.ones(shape))
        self.bias = pt_nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        dims = (1 if self.channel_first else -1)
        rms = (x.pow(2).mean(dims, keepdim=True) + 1e-6).sqrt()
        return (x / rms) * self.gamma + self.bias


class PtUpsample(pt_nn.Upsample):
    def forward(self, x):
        return super().forward(x).type_as(x)


class PtResample(pt_nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = pt_nn.Sequential(
                PtUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                pt_nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = pt_nn.Sequential(
                PtUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                pt_nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = PtCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = pt_nn.Sequential(
                pt_nn.ZeroPad2d((0, 1, 0, 1)),
                pt_nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = pt_nn.Sequential(
                pt_nn.ZeroPad2d((0, 1, 0, 1)),
                pt_nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = PtCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = pt_nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        from einops import rearrange
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    if cache_x.shape[2] < 2:
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                            cache_x], dim=2)
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
                if first_chunk:
                    x = x[:, :, 1:, :, :]

        t_now = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t_now)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class PtResidualBlock(pt_nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = pt_nn.Sequential(
            PtRMS_norm(in_dim, images=False),
            pt_nn.SiLU(),
            PtCausalConv3d(in_dim, out_dim, 3, padding=1),
            PtRMS_norm(out_dim, images=False),
            pt_nn.SiLU(),
            pt_nn.Dropout(dropout),
            PtCausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            PtCausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim else pt_nn.Identity())

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, PtCausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                        cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class PtAttentionBlock(pt_nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = PtRMS_norm(dim)
        self.to_qkv = pt_nn.Conv2d(dim, dim * 3, 1)
        self.proj = pt_nn.Conv2d(dim, dim, 1)
        pt_nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        from einops import rearrange
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x).reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1))
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


# ============================================================================
# Weight transfer utilities
# ============================================================================

def transfer_conv3d_weights(pt_conv, mlx_conv):
    """Transfer Conv3d weights from PyTorch (O,I,D,H,W) to MLX (O,D,H,W,I)."""
    w = pt_conv.weight.detach().numpy()
    w_mlx = np.transpose(w, (0, 2, 3, 4, 1))
    mlx_conv.weight = mx.array(w_mlx)
    if pt_conv.bias is not None:
        mlx_conv.bias = mx.array(pt_conv.bias.detach().numpy())


def transfer_conv2d_weights(pt_conv, mlx_conv):
    """Transfer Conv2d weights from PyTorch (O,I,H,W) to MLX (O,H,W,I)."""
    w = pt_conv.weight.detach().numpy()
    w_mlx = np.transpose(w, (0, 2, 3, 1))
    mlx_conv.weight = mx.array(w_mlx)
    if pt_conv.bias is not None:
        mlx_conv.bias = mx.array(pt_conv.bias.detach().numpy())


def transfer_causal_conv3d(pt_cc, mlx_cc):
    """Transfer CausalConv3d weights (direct weight/bias, no .conv wrapper)."""
    w = pt_cc.weight.detach().numpy()
    w_mlx = np.transpose(w, (0, 2, 3, 4, 1))
    mlx_cc.weight = mx.array(w_mlx)
    if pt_cc.bias is not None:
        mlx_cc.bias = mx.array(pt_cc.bias.detach().numpy())


def transfer_rms_norm(pt_norm, mlx_norm):
    """Transfer RMS_norm weights."""
    gamma = pt_norm.gamma.detach().numpy().flatten()
    mlx_norm.gamma = mx.array(gamma)


def transfer_resample(pt_res, mlx_res):
    """Transfer Resample weights."""
    mode = pt_res.mode
    # PyTorch uses nn.Sequential with Upsample/ZeroPad at [0] and Conv2d at [1]
    # MLX uses self.resample = [None, nn.Conv2d] matching PyTorch nn.Sequential indexing
    if mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d"):
        transfer_conv2d_weights(pt_res.resample[1], mlx_res.resample[1])

    if mode in ("upsample3d", "downsample3d"):
        transfer_causal_conv3d(pt_res.time_conv, mlx_res.time_conv)


def transfer_residual_block(pt_block, mlx_block):
    """Transfer ResidualBlock weights."""
    transfer_rms_norm(pt_block.residual[0], mlx_block.residual[0])
    transfer_causal_conv3d(pt_block.residual[2], mlx_block.residual[2])
    transfer_rms_norm(pt_block.residual[3], mlx_block.residual[3])
    transfer_causal_conv3d(pt_block.residual[6], mlx_block.residual[6])

    if not isinstance(pt_block.shortcut, pt_nn.Identity):
        transfer_causal_conv3d(pt_block.shortcut, mlx_block.shortcut)


def transfer_attention_block(pt_attn, mlx_attn):
    """Transfer AttentionBlock weights."""
    transfer_rms_norm(pt_attn.norm, mlx_attn.norm)
    transfer_conv2d_weights(pt_attn.to_qkv, mlx_attn.to_qkv)
    transfer_conv2d_weights(pt_attn.proj, mlx_attn.proj)


# ============================================================================
# Tests
# ============================================================================

class TestPatchify:
    """Test patchify/unpatchify operations."""

    def test_patchify_5d(self):
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 8, 8, 3).astype(np.float32)
        x_mlx = mx.array(x_np)
        y_mlx = mlx_patchify(x_mlx, 2)
        mx.eval(y_mlx)

        from einops import rearrange
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))
        y_pt = rearrange(x_pt, "b c f (h q) (w r) -> b (c r q) f h w", q=2, r=2)

        y_pt_cl = y_pt.numpy().transpose(0, 2, 3, 4, 1)
        y_mlx_np = np.array(y_mlx)

        assert y_mlx_np.shape == y_pt_cl.shape, f"{y_mlx_np.shape} != {y_pt_cl.shape}"
        np.testing.assert_allclose(y_mlx_np, y_pt_cl, atol=1e-6)

    def test_unpatchify_5d(self):
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 4, 4, 12).astype(np.float32)
        x_mlx = mx.array(x_np)
        y_mlx = mlx_unpatchify(x_mlx, 2)
        mx.eval(y_mlx)

        from einops import rearrange
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))
        y_pt = rearrange(x_pt, "b (c r q) f h w -> b c f (h q) (w r)", q=2, r=2)
        y_pt_cl = y_pt.numpy().transpose(0, 2, 3, 4, 1)
        y_mlx_np = np.array(y_mlx)

        assert y_mlx_np.shape == y_pt_cl.shape
        np.testing.assert_allclose(y_mlx_np, y_pt_cl, atol=1e-6)

    def test_roundtrip(self):
        np.random.seed(42)
        x_np = np.random.randn(1, 2, 8, 8, 3).astype(np.float32)
        x_mlx = mx.array(x_np)
        y = mlx_unpatchify(mlx_patchify(x_mlx, 2), 2)
        mx.eval(y)
        np.testing.assert_allclose(np.array(y), x_np, atol=1e-6)


class TestCausalConv3d:
    """Test CausalConv3d."""

    def test_basic_shapes(self):
        torch.manual_seed(42)
        in_c, out_c = 4, 8

        pt = PtCausalConv3d(in_c, out_c, 3, padding=1)
        mlx_conv = MlxCausalConv3d(in_c, out_c, 3, padding=1)
        transfer_causal_conv3d(pt, mlx_conv)

        x_np = np.random.randn(1, 3, 4, 4, in_c).astype(np.float32)
        x_mlx = mx.array(x_np)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt = pt(x_pt)
        y_mlx = mlx_conv(x_mlx)
        mx.eval(y_mlx)

        y_pt_cl = y_pt.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx, y_pt_cl, atol=1e-4, msg="CausalConv3d output mismatch")

    def test_with_cache(self):
        torch.manual_seed(42)
        in_c, out_c = 4, 8

        pt = PtCausalConv3d(in_c, out_c, 3, padding=1)
        mlx_conv = MlxCausalConv3d(in_c, out_c, 3, padding=1)
        transfer_causal_conv3d(pt, mlx_conv)

        x_np = np.random.randn(1, 5, 4, 4, in_c).astype(np.float32)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt_full = pt(x_pt)

        x1_np = x_np[:, :3, :, :, :]
        x2_np = x_np[:, 3:, :, :, :]

        y1_mlx = mlx_conv(mx.array(x1_np))
        cache = mx.array(x1_np[:, -CACHE_T:, :, :, :])
        y2_mlx = mlx_conv(mx.array(x2_np), cache_x=cache)
        y_mlx_full = mx.concatenate([y1_mlx, y2_mlx], axis=1)
        mx.eval(y_mlx_full)

        y_pt_cl = y_pt_full.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx_full, y_pt_cl, atol=1e-4, msg="CausalConv3d cache mismatch")

    def test_kernel_1x1x1(self):
        torch.manual_seed(42)
        in_c, out_c = 4, 4

        pt = PtCausalConv3d(in_c, out_c, 1)
        mlx_conv = MlxCausalConv3d(in_c, out_c, 1)
        transfer_causal_conv3d(pt, mlx_conv)

        x_np = np.random.randn(1, 3, 4, 4, in_c).astype(np.float32)
        x_mlx = mx.array(x_np)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt = pt(x_pt)
        y_mlx = mlx_conv(x_mlx)
        mx.eval(y_mlx)

        y_pt_cl = y_pt.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx, y_pt_cl, atol=1e-4, msg="CausalConv3d 1x1x1 mismatch")

    def test_strided(self):
        torch.manual_seed(42)
        in_c, out_c = 4, 4

        pt = PtCausalConv3d(in_c, out_c, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        mlx_conv = MlxCausalConv3d(in_c, out_c, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        transfer_causal_conv3d(pt, mlx_conv)

        x_np = np.random.randn(1, 4, 4, 4, in_c).astype(np.float32)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt = pt(x_pt)
        y_mlx = mlx_conv(mx.array(x_np))
        mx.eval(y_mlx)

        y_pt_cl = y_pt.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx, y_pt_cl, atol=1e-4, msg="Strided CausalConv3d mismatch")


class TestRMSNorm:
    """Test RMS_norm."""

    def test_channel_first(self):
        torch.manual_seed(42)
        dim = 8

        pt = PtRMS_norm(dim, channel_first=True, images=False)
        mlx_norm = MlxRMS_norm(dim, channel_first=True, images=False)
        transfer_rms_norm(pt, mlx_norm)

        x_np = np.random.randn(2, 3, 4, 4, dim).astype(np.float32)
        x_mlx = mx.array(x_np)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt = pt(x_pt)
        y_mlx = mlx_norm(x_mlx)
        mx.eval(y_mlx)

        y_pt_cl = y_pt.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx, y_pt_cl, atol=1e-4, msg="RMS_norm mismatch")


class TestResidualBlock:
    """Test ResidualBlock."""

    def test_same_dim(self):
        torch.manual_seed(42)
        dim = 8

        pt = PtResidualBlock(dim, dim)
        mlx_block = MlxResidualBlock(dim, dim)
        transfer_residual_block(pt, mlx_block)

        x_np = np.random.randn(1, 3, 4, 4, dim).astype(np.float32)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt = pt(x_pt)
        y_mlx = mlx_block(mx.array(x_np))
        mx.eval(y_mlx)

        y_pt_cl = y_pt.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx, y_pt_cl, atol=1e-3, msg="ResidualBlock same-dim mismatch")

    def test_diff_dim(self):
        torch.manual_seed(42)
        in_dim, out_dim = 4, 8

        pt = PtResidualBlock(in_dim, out_dim)
        mlx_block = MlxResidualBlock(in_dim, out_dim)
        transfer_residual_block(pt, mlx_block)

        x_np = np.random.randn(1, 3, 4, 4, in_dim).astype(np.float32)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt = pt(x_pt)
        y_mlx = mlx_block(mx.array(x_np))
        mx.eval(y_mlx)

        y_pt_cl = y_pt.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx, y_pt_cl, atol=1e-3, msg="ResidualBlock diff-dim mismatch")


class TestAttentionBlock:
    """Test AttentionBlock."""

    def test_attention(self):
        torch.manual_seed(42)
        dim = 16

        pt = PtAttentionBlock(dim)
        mlx_attn = MlxAttentionBlock(dim)
        transfer_attention_block(pt, mlx_attn)

        x_np = np.random.randn(1, 2, 4, 4, dim).astype(np.float32)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt = pt(x_pt)
        y_mlx = mlx_attn(mx.array(x_np))
        mx.eval(y_mlx)

        y_pt_cl = y_pt.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx, y_pt_cl, atol=1e-3, msg="AttentionBlock mismatch")


class TestResample:
    """Test Resample module with weight transfer."""

    def test_downsample2d(self):
        torch.manual_seed(42)
        dim = 8

        pt = PtResample(dim, "downsample2d")
        mlx_res = MlxResample(dim, "downsample2d")
        transfer_conv2d_weights(pt.resample[1], mlx_res.resample[1])

        x_np = np.random.randn(1, 3, 8, 8, dim).astype(np.float32)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt = pt(x_pt)
        y_mlx = mlx_res(mx.array(x_np))
        mx.eval(y_mlx)

        y_pt_cl = y_pt.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx, y_pt_cl, atol=1e-4, msg="Resample downsample2d mismatch")

    def test_upsample2d(self):
        torch.manual_seed(42)
        dim = 8

        pt = PtResample(dim, "upsample2d")
        mlx_res = MlxResample(dim, "upsample2d")
        transfer_conv2d_weights(pt.resample[1], mlx_res.resample[1])

        x_np = np.random.randn(1, 3, 4, 4, dim).astype(np.float32)
        x_pt = torch.tensor(x_np.transpose(0, 4, 1, 2, 3))

        y_pt = pt(x_pt)
        y_mlx = mlx_res(mx.array(x_np))
        mx.eval(y_mlx)

        y_pt_cl = y_pt.detach().numpy().transpose(0, 2, 3, 4, 1)
        assert_close(y_mlx, y_pt_cl, atol=1e-4, msg="Resample upsample2d mismatch")


class TestVAEShapes:
    """Test WanVAE_ encode/decode shapes and conv3d counting."""

    def test_vae_encode_decode_shapes(self):
        vae = MlxWanVAE_(
            dim=32, dec_dim=32, z_dim=4,
            dim_mult=[1, 2, 2], num_res_blocks=2,
            temperal_downsample=[True, False],
        )

        x = mx.random.normal((1, 5, 16, 16, 3))
        scale = [mx.array(0.0), mx.array(1.0)]

        mu = vae.encode(x, scale)
        mx.eval(mu)
        assert mu.shape[0] == 1
        assert mu.shape[-1] == 4

        recon = vae.decode(mu, scale)
        mx.eval(recon)
        assert recon.shape[0] == 1
        assert recon.shape[-1] == 3

    def test_conv3d_count(self):
        vae = MlxWanVAE_(
            dim=32, dec_dim=32, z_dim=4,
            dim_mult=[1, 2, 2], num_res_blocks=2,
            temperal_downsample=[True, False],
        )
        enc_count = _count_conv3d(vae.encoder)
        dec_count = _count_conv3d(vae.decoder)
        assert enc_count > 0, "Encoder should have CausalConv3d layers"
        assert dec_count > 0, "Decoder should have CausalConv3d layers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
