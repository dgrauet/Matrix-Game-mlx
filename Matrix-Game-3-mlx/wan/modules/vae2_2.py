"""Wan2.2 VAE — MLX port.

Channels-last layout throughout: (B, T, H, W, C) for 5D, (B, H, W, C) for 4D.
PyTorch reference: Matrix-Game-3/wan/modules/vae2_2.py
"""
import logging
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

__all__ = ["Wan2_2_VAE"]

CACHE_T = 2


# ---------------------------------------------------------------------------
# CausalConv3d
# ---------------------------------------------------------------------------

class CausalConv3d(nn.Module):
    """Causal 3D convolution with left-only temporal padding.

    Input/output layout: (B, T, H, W, C) — channels-last.

    Stores weight and bias as direct attributes (like PyTorch nn.Conv3d)
    so that state_dict keys are ``weight`` / ``bias`` without an extra
    ``.conv.`` level.  MLX weight layout: (O, D, H, W, I).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        # Causal: double the temporal padding, applied only on the left
        self.temporal_padding = 2 * padding[0]

        # Store conv parameters for the functional call
        self._kernel_size = kernel_size
        self._stride = stride
        # Spatial-only padding (temporal is handled manually)
        self._padding = (0, padding[1], padding[2])

        # Weight: MLX Conv3d layout (O, D, H, W, I)
        self.weight = mx.zeros((out_channels, *kernel_size, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array, cache_x: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, H, W, C).
            cache_x: Cached temporal frames from previous chunk, shape (B, T', H, W, C).

        Returns:
            Output tensor of shape (B, T_out, H_out, W_out, C_out).
        """
        pad_t = self.temporal_padding
        if cache_x is not None and self.temporal_padding > 0:
            x = mx.concatenate([cache_x, x], axis=1)
            pad_t = self.temporal_padding - cache_x.shape[1]

        if pad_t > 0:
            # Pad temporal dim (axis=1) on the left only
            x = mx.pad(x, [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)])

        y = mx.conv3d(
            x,
            self.weight,
            stride=self._stride,
            padding=self._padding,
        )
        if "bias" in self:
            y = y + self.bias
        return y


# ---------------------------------------------------------------------------
# RMS_norm
# ---------------------------------------------------------------------------

class RMS_norm(nn.Module):
    """RMS normalization for channel-last tensors.

    Normalizes over the channel (last) dimension.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        # In MLX channels-last, gamma is always shape (dim,)
        self.gamma = mx.ones((dim,))
        self._has_bias = bias
        if bias:
            self.bias = mx.zeros((dim,))
        self.scale = dim ** 0.5

    def __call__(self, x: mx.array) -> mx.array:
        # Normalize over last dimension (channels)
        rms = (x.square().mean(axis=-1, keepdims=True) + 1e-6).sqrt()
        out = (x / rms) * self.gamma
        if self._has_bias:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Upsample (nearest-neighbor 2x spatial)
# ---------------------------------------------------------------------------

def _upsample_nearest_2x(x: mx.array) -> mx.array:
    """Nearest-neighbor 2x spatial upsample for (B, H, W, C) tensors."""
    B, H, W, C = x.shape
    # Repeat along H and W
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C))
    x = x.reshape(B, H * 2, W * 2, C)
    return x


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------

class Resample(nn.Module):
    """Up/downsample module with optional temporal resampling."""

    def __init__(self, dim: int, mode: str):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        elif mode == "upsample3d":
            self.conv = nn.Conv2d(dim, dim, 3, padding=1)
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            # ZeroPad2d(0,1,0,1) + Conv2d(stride=2) — pad right and bottom
            self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=0)
        elif mode == "downsample3d":
            self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=0)
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def __call__(
        self,
        x: mx.array,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        first_chunk: bool = False,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: (B, T, H, W, C)
        """
        if feat_idx is None:
            feat_idx = [0]

        b, t, h, w, c = x.shape

        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"

                cache_x = x[:, -CACHE_T:, :, :, :]

                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    if cache_x.shape[1] < 2:
                        cache_x = mx.concatenate([
                            feat_cache[idx][:, -1:, :, :, :],
                            cache_x,
                        ], axis=1)
                    x = self.time_conv(x, feat_cache[idx])

                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                # x shape: (B, T, H, W, 2*C) -> interleave to (B, T*2, H, W, C)
                x = x.reshape(b, t, h, w, 2, c)
                # Interleave: stack the two halves along time
                x0 = x[:, :, :, :, 0, :]  # (B, T, H, W, C)
                x1 = x[:, :, :, :, 1, :]  # (B, T, H, W, C)
                # Stack and reshape to interleave: [t0_a, t0_b, t1_a, t1_b, ...]
                x = mx.stack([x0, x1], axis=2)  # (B, T, 2, H, W, C)
                x = x.reshape(b, t * 2, h, w, c)

                if first_chunk:
                    x = x[:, 1:, :, :, :]

        t_now = x.shape[1]

        # Apply 2D conv per frame: reshape (B,T,H,W,C) -> (B*T,H,W,C)
        x_2d = x.reshape(b * t_now, x.shape[2], x.shape[3], c if self.mode.startswith("upsample") else x.shape[4])

        if self.mode.startswith("upsample"):
            x_2d = _upsample_nearest_2x(x_2d)
            x_2d = self.conv(x_2d)
        elif self.mode.startswith("downsample"):
            # ZeroPad: pad right=1, bottom=1 -> (B*T, H+1, W+1, C)
            x_2d = mx.pad(x_2d, [(0, 0), (0, 1), (0, 1), (0, 0)])
            x_2d = self.conv(x_2d)
        else:
            pass  # mode == "none"

        # Reshape back: (B*T, H', W', C') -> (B, T, H', W', C')
        new_h, new_w, new_c = x_2d.shape[1], x_2d.shape[2], x_2d.shape[3]
        x = x_2d.reshape(b, t_now, new_h, new_w, new_c)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, -1:, :, :, :]
                    x = self.time_conv(
                        mx.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

        return x


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------

class _SiLU_placeholder(nn.Module):
    """Non-parametric SiLU placeholder for Sequential-like lists."""
    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(x)


class _Dropout_placeholder(nn.Module):
    """Non-parametric Dropout placeholder for Sequential-like lists."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self._p = p
    def __call__(self, x: mx.array) -> mx.array:
        return x  # no-op at inference


class ResidualBlock(nn.Module):
    """Residual block with CausalConv3d and RMS normalization.

    Uses a list ``self.residual`` to mirror PyTorch ``nn.Sequential`` indexing:
      0 — RMS_norm      (has gamma)
      1 — SiLU          (no params)
      2 — CausalConv3d  (has weight/bias)
      3 — RMS_norm      (has gamma)
      4 — SiLU          (no params)
      5 — Dropout       (no params)
      6 — CausalConv3d  (has weight/bias)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Residual path as a list matching PyTorch nn.Sequential indices
        self.residual = [
            RMS_norm(in_dim, images=False),        # 0
            _SiLU_placeholder(),                   # 1
            CausalConv3d(in_dim, out_dim, 3, padding=1),  # 2
            RMS_norm(out_dim, images=False),        # 3
            _SiLU_placeholder(),                   # 4
            _Dropout_placeholder(dropout),         # 5
            CausalConv3d(out_dim, out_dim, 3, padding=1),  # 6
        ]

        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim else None
        )

    def __call__(
        self,
        x: mx.array,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        first_chunk: bool = False,
    ) -> mx.array:
        if feat_idx is None:
            feat_idx = [0]

        # Shortcut
        if self.shortcut is not None:
            h = self.shortcut(x)
        else:
            h = x

        # Residual path: iterate through the Sequential-like list
        # Conv layers (indices 2, 6) need cache handling
        conv_indices = {2, 6}
        for i, layer in enumerate(self.residual):
            if i in conv_indices and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :]
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([
                        feat_cache[idx][:, -1:, :, :, :],
                        cache_x,
                    ], axis=1)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x + h


# ---------------------------------------------------------------------------
# AttentionBlock
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """Spatial self-attention (2D, per frame)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        """Forward pass.

        Args:
            x: (B, T, H, W, C)
        """
        identity = x
        b, t, h, w, c = x.shape

        # Reshape to 2D: (B*T, H, W, C)
        x = x.reshape(b * t, h, w, c)
        x = self.norm(x)

        # QKV via 1x1 conv: (B*T, H, W, 3*C)
        qkv = self.to_qkv(x)

        # Reshape to (B*T, H*W, 3*C) then split
        qkv = qkv.reshape(b * t, h * w, 3 * c)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Add head dimension: (B*T, 1, H*W, C)
        q = q[:, None, :, :]
        k = k[:, None, :, :]
        v = v[:, None, :, :]

        # Scaled dot-product attention
        scale = c ** -0.5
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        # Remove head dim and reshape: (B*T, H*W, C) -> (B*T, H, W, C)
        x = x.squeeze(1)
        x = x.reshape(b * t, h, w, c)

        # Output projection
        x = self.proj(x)

        # Reshape back: (B, T, H, W, C)
        x = x.reshape(b, t, h, w, c)
        return x + identity


# ---------------------------------------------------------------------------
# patchify / unpatchify (channels-last)
# ---------------------------------------------------------------------------

def patchify(x: mx.array, patch_size: int) -> mx.array:
    """Spatial patchification matching PyTorch einops layout.

    PyTorch: "b c f (h q) (w r) -> b (c r q) f h w" with channels-first.
    MLX equivalent for channels-last: flatten channel dims in order (C, r, q).

    For 5D (B, T, H, W, C): rearrange to (B, T, H//p, W//p, C*p*p)
    For 4D (B, H, W, C): rearrange to (B, H//p, W//p, C*p*p)
    """
    if patch_size == 1:
        return x

    p = patch_size
    if x.ndim == 4:
        B, H, W, C = x.shape
        # (B, H//p, q, W//p, r, C) -> (B, H//p, W//p, C, r, q) -> flatten
        x = x.reshape(B, H // p, p, W // p, p, C)
        x = x.transpose(0, 1, 3, 5, 4, 2)  # (B, H//p, W//p, C, r, q)
        x = x.reshape(B, H // p, W // p, C * p * p)
    elif x.ndim == 5:
        B, T, H, W, C = x.shape
        # (B, T, H//p, q, W//p, r, C) -> (B, T, H//p, W//p, C, r, q) -> flatten
        x = x.reshape(B, T, H // p, p, W // p, p, C)
        x = x.transpose(0, 1, 2, 4, 6, 5, 3)  # (B, T, H//p, W//p, C, r, q)
        x = x.reshape(B, T, H // p, W // p, C * p * p)
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x: mx.array, patch_size: int) -> mx.array:
    """Reverse spatial patchification matching PyTorch einops layout.

    PyTorch: "b (c r q) f h w -> b c f (h q) (w r)" with channels-first.
    MLX equivalent for channels-last: channel order is (C, r, q).

    For 5D (B, T, H, W, C*p*p) -> (B, T, H*p, W*p, C)
    For 4D (B, H, W, C*p*p) -> (B, H*p, W*p, C)
    """
    if patch_size == 1:
        return x

    p = patch_size
    if x.ndim == 4:
        B, H, W, Cpp = x.shape
        C = Cpp // (p * p)
        # Channels are in order (C, r, q) -> reshape to (B, H, W, C, r, q)
        x = x.reshape(B, H, W, C, p, p)
        # -> (B, H, q, W, r, C)
        x = x.transpose(0, 1, 5, 2, 4, 3)
        x = x.reshape(B, H * p, W * p, C)
    elif x.ndim == 5:
        B, T, H, W, Cpp = x.shape
        C = Cpp // (p * p)
        # Channels are in order (C, r, q) -> reshape to (B, T, H, W, C, r, q)
        x = x.reshape(B, T, H, W, C, p, p)
        # -> (B, T, H, q, W, r, C)
        x = x.transpose(0, 1, 2, 6, 3, 5, 4)
        x = x.reshape(B, T, H * p, W * p, C)
    return x


# ---------------------------------------------------------------------------
# AvgDown3D / DupUp3D
# ---------------------------------------------------------------------------

class AvgDown3D(nn.Module):
    """Average downsample 3D for channels-last layout (B, T, H, W, C)."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Input: (B, T, H, W, C)."""
        # Pad temporal dim if needed
        pad_t = (self.factor_t - x.shape[1] % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = mx.pad(x, [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)])

        B, T, H, W, C = x.shape

        # PyTorch: (B, C, T, H, W) -> view -> permute -> view -> view -> mean
        # MLX channels-last: (B, T, H, W, C)
        # Reshape to separate factors
        x = x.reshape(
            B,
            T // self.factor_t, self.factor_t,
            H // self.factor_s, self.factor_s,
            W // self.factor_s, self.factor_s,
            C,
        )
        # Move factor dims next to C: (B, T', H', W', factor_t, factor_s, factor_s, C)
        x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
        # Merge factors with C: (B, T', H', W', C * factor)
        x = x.reshape(
            B,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
            C * self.factor,
        )
        # Split into groups and average
        x = x.reshape(
            B,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
            self.out_channels,
            self.group_size,
        )
        x = x.mean(axis=5)
        return x


class DupUp3D(nn.Module):
    """Duplicate upsample 3D for channels-last layout (B, T, H, W, C)."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        """Forward pass. Input: (B, T, H, W, C)."""
        # repeat_interleave along C (last dim)
        # In PyTorch: repeat_interleave(repeats, dim=1) on (B,C,T,H,W)
        # In MLX channels-last: repeat_interleave along dim=-1 on (B,T,H,W,C)
        B, T, H, W, C = x.shape
        x = mx.repeat(x, self.repeats, axis=-1)  # (B, T, H, W, C*repeats)

        # Reshape to separate spatial/temporal factors
        # PyTorch had: (B, out_C, factor_t, factor_s, factor_s, T, H, W) then permute
        # MLX: (B, T, H, W, out_C, factor_t, factor_s, factor_s)
        x = x.reshape(B, T, H, W, self.out_channels, self.factor_t, self.factor_s, self.factor_s)

        # Permute to interleave: (B, T, factor_t, H, factor_s, W, factor_s, out_C)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)

        # Final reshape: (B, T*factor_t, H*factor_s, W*factor_s, out_C)
        x = x.reshape(B, T * self.factor_t, H * self.factor_s, W * self.factor_s, self.out_channels)

        if first_chunk:
            x = x[:, self.factor_t - 1:, :, :, :]

        return x


# ---------------------------------------------------------------------------
# Down_ResidualBlock / Up_ResidualBlock
# ---------------------------------------------------------------------------

class Down_ResidualBlock(nn.Module):
    """Downsample block with residual connection and optional temporal downsampling."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        downsamples = []
        cur_dim = in_dim
        for _ in range(mult):
            downsamples.append(ResidualBlock(cur_dim, out_dim, dropout))
            cur_dim = out_dim

        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = downsamples

    def __call__(
        self,
        x: mx.array,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        first_chunk: bool = False,
    ) -> mx.array:
        if feat_idx is None:
            feat_idx = [0]

        x_copy = x
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    """Upsample block with residual connection and optional temporal upsampling."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ):
        super().__init__()
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim, out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        upsamples = []
        cur_dim = in_dim
        for _ in range(mult):
            upsamples.append(ResidualBlock(cur_dim, out_dim, dropout))
            cur_dim = out_dim

        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = upsamples

    def __call__(
        self,
        x: mx.array,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        first_chunk: bool = False,
    ) -> mx.array:
        if feat_idx is None:
            feat_idx = [0]

        x_main = x
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx, first_chunk)

        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


# ---------------------------------------------------------------------------
# Encoder3d
# ---------------------------------------------------------------------------

class Encoder3d(nn.Module):
    """3D encoder with causal convolutions."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [True, True, False],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]

        # Init block (input channels = 12 due to patchify with patch_size=2: 3*2*2=12)
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # Downsample blocks
        downsamples = []
        out_dim = dims[0]
        for i, (in_dim_i, out_dim_i) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(Down_ResidualBlock(
                in_dim=in_dim_i,
                out_dim=out_dim_i,
                dropout=dropout,
                mult=num_res_blocks,
                temperal_downsample=t_down_flag,
                down_flag=i != len(dim_mult) - 1,
            ))
            out_dim = out_dim_i
        self.downsamples = downsamples

        # Middle: list matching nn.Sequential(ResidualBlock, AttentionBlock, ResidualBlock)
        self.middle = [
            ResidualBlock(out_dim, out_dim, dropout),   # 0
            AttentionBlock(out_dim),                     # 1
            ResidualBlock(out_dim, out_dim, dropout),   # 2
        ]

        # Head: list matching nn.Sequential(RMS_norm, SiLU, CausalConv3d)
        # SiLU at index 1 has no params; use placeholder
        self.head = [
            RMS_norm(out_dim, images=False),             # 0
            _SiLU_placeholder(),                         # 1
            CausalConv3d(out_dim, z_dim, 3, padding=1),  # 2
        ]

    def __call__(
        self,
        x: mx.array,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> mx.array:
        if feat_idx is None:
            feat_idx = [0]

        # Initial conv with cache
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([
                    feat_cache[idx][:, -1:, :, :, :],
                    cache_x,
                ], axis=1)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Downsample blocks
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # Middle blocks: middle[0]=ResidualBlock, middle[1]=AttentionBlock, middle[2]=ResidualBlock
        if feat_cache is not None:
            x = self.middle[0](x, feat_cache, feat_idx)
        else:
            x = self.middle[0](x)
        x = self.middle[1](x)
        if feat_cache is not None:
            x = self.middle[2](x, feat_cache, feat_idx)
        else:
            x = self.middle[2](x)

        # Head: head[0]=RMS_norm, head[1]=SiLU, head[2]=CausalConv3d
        x = self.head[0](x)
        x = self.head[1](x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([
                    feat_cache[idx][:, -1:, :, :, :],
                    cache_x,
                ], axis=1)
            x = self.head[2](x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.head[2](x)

        return x


# ---------------------------------------------------------------------------
# Decoder3d
# ---------------------------------------------------------------------------

class Decoder3d(nn.Module):
    """3D decoder with causal convolutions."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_upsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # Init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # Middle: list matching nn.Sequential(ResidualBlock, AttentionBlock, ResidualBlock)
        self.middle = [
            ResidualBlock(dims[0], dims[0], dropout),   # 0
            AttentionBlock(dims[0]),                     # 1
            ResidualBlock(dims[0], dims[0], dropout),   # 2
        ]

        # Upsample blocks
        upsamples = []
        for i, (in_dim_i, out_dim_i) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(Up_ResidualBlock(
                in_dim=in_dim_i,
                out_dim=out_dim_i,
                dropout=dropout,
                mult=num_res_blocks + 1,
                temperal_upsample=t_up_flag,
                up_flag=i != len(dim_mult) - 1,
            ))
        self.upsamples = upsamples

        # Head: list matching nn.Sequential(RMS_norm, SiLU, CausalConv3d)
        out_dim = dims[-1]
        self.head = [
            RMS_norm(out_dim, images=False),             # 0
            _SiLU_placeholder(),                         # 1
            CausalConv3d(out_dim, 12, 3, padding=1),     # 2
        ]

    def __call__(
        self,
        x: mx.array,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        first_chunk: bool = False,
    ) -> mx.array:
        if feat_idx is None:
            feat_idx = [0]

        # Initial conv with cache
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([
                    feat_cache[idx][:, -1:, :, :, :],
                    cache_x,
                ], axis=1)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Middle blocks: middle[0]=ResidualBlock, middle[1]=AttentionBlock, middle[2]=ResidualBlock
        if feat_cache is not None:
            x = self.middle[0](x, feat_cache, feat_idx)
        else:
            x = self.middle[0](x)
        x = self.middle[1](x)
        if feat_cache is not None:
            x = self.middle[2](x, feat_cache, feat_idx)
        else:
            x = self.middle[2](x)

        # Upsample blocks
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        # Head: head[0]=RMS_norm, head[1]=SiLU, head[2]=CausalConv3d
        x = self.head[0](x)
        x = self.head[1](x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([
                    feat_cache[idx][:, -1:, :, :, :],
                    cache_x,
                ], axis=1)
            x = self.head[2](x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.head[2](x)

        return x


# ---------------------------------------------------------------------------
# Helper: count CausalConv3d layers
# ---------------------------------------------------------------------------

def _count_conv3d(model: nn.Module) -> int:
    """Count CausalConv3d layers recursively."""
    count = 0
    children = model.children()
    for name, child in children.items():
        if isinstance(child, CausalConv3d):
            count += 1
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, CausalConv3d):
                    count += 1
                elif isinstance(item, nn.Module):
                    count += _count_conv3d(item)
        elif isinstance(child, nn.Module):
            count += _count_conv3d(child)
    return count


# ---------------------------------------------------------------------------
# WanVAE_
# ---------------------------------------------------------------------------

class WanVAE_(nn.Module):
    """VAE model with encode/decode and chunk-based caching."""

    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 16,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [True, True, False],
        dropout: float = 0.0,
        pruning_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # Pruning-compatible
        dim = max(1, int(round(dim * (1.0 - pruning_rate))))
        dec_dim = max(1, int(round(dec_dim * (1.0 - pruning_rate))))

        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks,
            attn_scales, self.temperal_downsample, dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim, z_dim, dim_mult, num_res_blocks,
            attn_scales, self.temperal_upsample, dropout,
        )

        # Initialize cache
        self.clear_cache()

    def encode(self, x: mx.array, scale: list) -> mx.array:
        """Encode video to latent space.

        Args:
            x: Input video (B, T, H, W, C) in channels-last.
            scale: [mean, 1/std] for normalization.
        """
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[1]  # temporal dim in channels-last
        iter_ = 1 + (t - 1) // 4

        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :1, :, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, 1 + 4 * (i - 1):1 + 4 * i, :, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = mx.concatenate([out, out_], axis=1)

        conv1_out = self.conv1(out)
        mu = conv1_out[:, :, :, :, :self.z_dim]
        # log_var = conv1_out[:, :, :, :, self.z_dim:]  # not used for inference

        # Apply normalization scale
        if isinstance(scale[0], mx.array) and scale[0].ndim > 0:
            mean = scale[0].reshape(1, 1, 1, 1, self.z_dim)
            inv_std = scale[1].reshape(1, 1, 1, 1, self.z_dim)
            mu = (mu - mean) * inv_std
        else:
            mu = (mu - scale[0]) * scale[1]

        self.clear_cache()
        return mu

    def decode(self, z: mx.array, scale: list) -> mx.array:
        """Decode latent to video.

        Args:
            z: Latent tensor (B, T, H, W, C) in channels-last.
            scale: [mean, 1/std] for denormalization.
        """
        self.clear_cache()

        # Denormalize
        if isinstance(scale[0], mx.array) and scale[0].ndim > 0:
            mean = scale[0].reshape(1, 1, 1, 1, self.z_dim)
            inv_std = scale[1].reshape(1, 1, 1, 1, self.z_dim)
            z = z / inv_std + mean
        else:
            z = z / scale[1] + scale[0]

        iter_ = z.shape[1]  # temporal dim
        x = self.conv2(z)

        out = None
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, i:i + 1, :, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_ = self.decoder(
                    x[:, i:i + 1, :, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = mx.concatenate([out, out_], axis=1)

        out = unpatchify(out, patch_size=2)
        out = mx.clip(out, -1, 1)
        self.clear_cache()
        return out

    def clear_cache(self):
        """Clear temporal caches."""
        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


# ---------------------------------------------------------------------------
# Wan2_2_VAE (high-level wrapper)
# ---------------------------------------------------------------------------

class Wan2_2_VAE:
    """High-level VAE wrapper with normalization scale.

    This is the main entry point for VAE encode/decode operations.
    """

    # Normalization statistics (48 channels)
    MEAN = [
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
        -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
        -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
        -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
         0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
    ]

    STD = [
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    ]

    def __init__(
        self,
        z_dim: int = 48,
        c_dim: int = 160,
        vae_pth: Optional[str] = None,
        dim_mult: List[int] = [1, 2, 4, 4],
        temperal_downsample: List[bool] = [False, True, True],
        dtype: mx.Dtype = mx.float32,
    ):
        self.dtype = dtype
        self.z_dim = z_dim

        mean = mx.array(self.MEAN, dtype=dtype)
        std = mx.array(self.STD, dtype=dtype)
        self.scale = [mean, 1.0 / std]

        # Build model
        self.model = WanVAE_(
            dim=c_dim,
            dec_dim=c_dim,  # Will be overridden by actual config
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=temperal_downsample,
            dropout=0.0,
        )

        # Load weights if provided
        if vae_pth is not None:
            import os
            if os.path.exists(vae_pth):
                logging.info(f"Loading VAE weights from {vae_pth}")
                weights = mx.load(vae_pth)
                self.model.load_weights(list(weights.items()))

    def encode(self, videos: List[mx.array]) -> List[mx.array]:
        """Encode a list of videos to latent space.

        Args:
            videos: List of video tensors, each (T, H, W, C).
        """
        return [
            self.model.encode(
                mx.expand_dims(v, axis=0).astype(self.dtype),
                self.scale,
            ).squeeze(0)
            for v in videos
        ]

    def decode(self, zs: List[mx.array]) -> List[mx.array]:
        """Decode a list of latent tensors to videos.

        Args:
            zs: List of latent tensors, each (T, H, W, C).
        """
        return [
            mx.clip(
                self.model.decode(
                    mx.expand_dims(z, axis=0).astype(self.dtype),
                    self.scale,
                ),
                -1, 1,
            ).squeeze(0)
            for z in zs
        ]
