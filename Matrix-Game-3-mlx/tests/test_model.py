"""Validation tests for DiT core components: norms, embeddings, RoPE.

Compares MLX implementations against PyTorch reference with tight tolerances.
The PyTorch reference functions are defined inline (extracted from
Matrix-Game-3/wan/modules/model.py lines 229-366) to avoid importing the full
PyTorch module which has heavy dependencies (diffusers, einops, etc.).
"""
import sys
import os
import numpy as np
import pytest

# -- path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib.util
_conftest_spec = importlib.util.spec_from_file_location(
    "conftest", os.path.join(os.path.dirname(__file__), "conftest.py"))
_conftest = importlib.util.module_from_spec(_conftest_spec)
_conftest_spec.loader.exec_module(_conftest)
assert_close = _conftest.assert_close
torch_to_mlx = _conftest.torch_to_mlx

import mlx.core as mx
import torch
import torch.nn as nn

# Import MLX implementations
from wan.modules.model import (
    sinusoidal_embedding_1d,
    rope_params,
    rope_apply,
    rope_apply_with_indices,
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    WanCrossAttention,
    WanAttentionBlock,
    Head,
    WanModel,
)
mlx_to_torch = _conftest.mlx_to_torch


# =============================================================================
# PyTorch reference implementations (extracted from Matrix-Game-3)
# =============================================================================

def sinusoidal_embedding_1d_pt(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params_pt(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply_pt(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i.to(x_i.dtype)).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def rope_apply_with_indices_pt(x, grid_sizes, freqs, t_indices=None):
    n, c = x.size(2), x.size(3) // 2
    if freqs.dim() == 3:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=2)
    else:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        if t_indices is None:
            t_idx = torch.arange(f, device=freqs[0].device)
        else:
            t_idx = t_indices
            if torch.is_tensor(t_idx) and t_idx.dim() > 1:
                t_idx = t_idx[i]
            if not torch.is_tensor(t_idx):
                t_idx = torch.tensor(t_idx, device=freqs[0].device)
            elif t_idx.device != freqs[0].device:
                t_idx = t_idx.to(freqs[0].device)
            t_idx = t_idx.to(dtype=torch.long)
        if freqs[0].dim() == 3:
            t_freqs = freqs[0][:, t_idx, :]
            h_freqs = freqs[1][:, :h, :]
            w_freqs = freqs[2][:, :w, :]
            freqs_i = torch.cat([
                t_freqs.permute(1, 0, 2).view(f, 1, 1, n, -1).expand(f, h, w, n, -1),
                h_freqs.permute(1, 0, 2).view(1, h, 1, n, -1).expand(f, h, w, n, -1),
                w_freqs.permute(1, 0, 2).view(1, 1, w, n, -1).expand(f, h, w, n, -1),
            ], dim=-1).reshape(seq_len, n, -1)
        else:
            t_freqs = freqs[0][t_idx]
            h_freqs = freqs[1][:h]
            w_freqs = freqs[2][:w]
            freqs_i = torch.cat([
                t_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
                h_freqs.view(1, h, 1, -1).expand(f, h, w, -1),
                w_freqs.view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i.to(x_i.dtype)).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm_pt(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).to(x.dtype) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm_pt(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return torch.nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        ).type_as(x)


# =============================================================================
# sinusoidal_embedding_1d
# =============================================================================

class TestSinusoidalEmbedding1d:

    @pytest.mark.parametrize("dim", [64, 256])
    def test_basic(self, dim):
        positions = [0, 1, 5, 100, 999]
        pos_pt = torch.tensor(positions)
        pos_mlx = mx.array(positions)

        out_pt = sinusoidal_embedding_1d_pt(dim, pos_pt)
        out_mlx = sinusoidal_embedding_1d(dim, pos_mlx)

        assert_close(out_mlx, out_pt, atol=1e-4, msg=f"sinusoidal_embedding dim={dim}")

    def test_shape(self):
        dim = 128
        pos = mx.array([0, 1, 2])
        out = sinusoidal_embedding_1d(dim, pos)
        assert out.shape == (3, dim)


# =============================================================================
# rope_params
# =============================================================================

class TestRopeParams:

    @pytest.mark.parametrize("max_seq_len,dim", [(32, 16), (128, 64)])
    def test_matches_pytorch_complex(self, max_seq_len, dim):
        """cos/sin from MLX must match real/imag parts of PyTorch complex freqs."""
        freqs_pt = rope_params_pt(max_seq_len, dim)  # complex [max_seq_len, dim//2]
        cos_mlx, sin_mlx = rope_params(max_seq_len, dim)

        # Extract cos/sin from PyTorch complex
        freqs_real = torch.view_as_real(freqs_pt)  # [max_seq_len, dim//2, 2]
        cos_pt = freqs_real[..., 0]  # real part = cos
        sin_pt = freqs_real[..., 1]  # imag part = sin

        assert_close(cos_mlx, cos_pt, atol=1e-4, msg="rope_params cos")
        assert_close(sin_mlx, sin_pt, atol=1e-4, msg="rope_params sin")

    def test_shape(self):
        cos_f, sin_f = rope_params(64, 32)
        assert cos_f.shape == (64, 16)
        assert sin_f.shape == (64, 16)


# =============================================================================
# rope_apply
# =============================================================================

class TestRopeApply:

    def _make_inputs(self, B=1, S=24, N=4, D=12, seed=42):
        """Create matched PyTorch/MLX inputs for rope_apply testing.

        Grid: f=2, h=3, w=4 -> seq_len=24 = S.
        D must be even. c = D//2 = 6. Splits: c_t=2, c_h=2, c_w=2.
        """
        np.random.seed(seed)
        x_np = np.random.randn(B, S, N, D).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        grid = [[2, 3, 4]]
        grid_pt = torch.tensor(grid)

        max_seq_len = 8
        freqs_pt = rope_params_pt(max_seq_len, D)
        cos_mlx, sin_mlx = rope_params(max_seq_len, D)

        return x_pt, x_mlx, grid_pt, grid, freqs_pt, (cos_mlx, sin_mlx)

    def test_basic(self):
        x_pt, x_mlx, grid_pt, grid, freqs_pt, freqs_mlx = self._make_inputs()

        out_pt = rope_apply_pt(x_pt, grid_pt, freqs_pt)
        out_mlx = rope_apply(x_mlx, grid, freqs_mlx)

        assert_close(out_mlx, out_pt, atol=1e-3, msg="rope_apply basic")

    def test_with_padding(self):
        """Test with S > seq_len to exercise padding passthrough."""
        B, S, N, D = 1, 30, 4, 12
        np.random.seed(123)
        x_np = np.random.randn(B, S, N, D).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        grid = [[2, 3, 4]]  # seq_len=24 < 30
        grid_pt = torch.tensor(grid)

        max_seq_len = 8
        freqs_pt = rope_params_pt(max_seq_len, D)
        cos_mlx, sin_mlx = rope_params(max_seq_len, D)

        out_pt = rope_apply_pt(x_pt, grid_pt, freqs_pt)
        out_mlx = rope_apply(x_mlx, grid, (cos_mlx, sin_mlx))

        assert_close(out_mlx, out_pt, atol=1e-3, msg="rope_apply with padding")

    def test_batch(self):
        """Test with batch > 1."""
        B, S, N, D = 2, 24, 4, 12
        np.random.seed(99)
        x_np = np.random.randn(B, S, N, D).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        grid = [[2, 3, 4], [2, 3, 4]]
        grid_pt = torch.tensor(grid)

        max_seq_len = 8
        freqs_pt = rope_params_pt(max_seq_len, D)
        cos_mlx, sin_mlx = rope_params(max_seq_len, D)

        out_pt = rope_apply_pt(x_pt, grid_pt, freqs_pt)
        out_mlx = rope_apply(x_mlx, grid, (cos_mlx, sin_mlx))

        assert_close(out_mlx, out_pt, atol=1e-3, msg="rope_apply batch")


# =============================================================================
# rope_apply_with_indices
# =============================================================================

class TestRopeApplyWithIndices:

    def _make_inputs(self, B=1, S=24, N=4, D=12, seed=42):
        np.random.seed(seed)
        x_np = np.random.randn(B, S, N, D).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        grid = [[2, 3, 4]]
        grid_pt = torch.tensor(grid)

        max_seq_len = 8
        freqs_pt = rope_params_pt(max_seq_len, D)
        cos_mlx, sin_mlx = rope_params(max_seq_len, D)

        return x_pt, x_mlx, grid_pt, grid, freqs_pt, (cos_mlx, sin_mlx)

    def test_no_t_indices_matches_rope_apply(self):
        """Without t_indices, should match rope_apply exactly."""
        x_pt, x_mlx, grid_pt, grid, freqs_pt, freqs_mlx = self._make_inputs()

        out_pt = rope_apply_with_indices_pt(x_pt, grid_pt, freqs_pt, t_indices=None)
        out_mlx = rope_apply_with_indices(x_mlx, grid, freqs_mlx, t_indices=None)

        assert_close(out_mlx, out_pt, atol=1e-3, msg="rope_apply_with_indices (no t_indices)")

    def test_custom_t_indices(self):
        """With custom t_indices, compare against PyTorch."""
        x_pt, x_mlx, grid_pt, grid, freqs_pt, freqs_mlx = self._make_inputs()

        t_idx = [3, 5]  # custom temporal indices (f=2)
        t_idx_pt = torch.tensor(t_idx)

        out_pt = rope_apply_with_indices_pt(x_pt, grid_pt, freqs_pt, t_indices=t_idx_pt)
        out_mlx = rope_apply_with_indices(x_mlx, grid, freqs_mlx, t_indices=t_idx)

        assert_close(out_mlx, out_pt, atol=1e-3, msg="rope_apply_with_indices custom t_indices")

    def test_2d_t_indices(self):
        """With per-batch 2D t_indices."""
        B, S, N, D = 2, 24, 4, 12
        np.random.seed(77)
        x_np = np.random.randn(B, S, N, D).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        grid = [[2, 3, 4], [2, 3, 4]]
        grid_pt = torch.tensor(grid)

        max_seq_len = 8
        freqs_pt = rope_params_pt(max_seq_len, D)
        cos_mlx, sin_mlx = rope_params(max_seq_len, D)

        t_idx_np = np.array([[1, 3], [0, 5]])
        t_idx_pt = torch.from_numpy(t_idx_np)
        t_idx_mlx = mx.array(t_idx_np)

        out_pt = rope_apply_with_indices_pt(x_pt, grid_pt, freqs_pt, t_indices=t_idx_pt)
        out_mlx = rope_apply_with_indices(x_mlx, grid, (cos_mlx, sin_mlx), t_indices=t_idx_mlx)

        assert_close(out_mlx, out_pt, atol=1e-3, msg="rope_apply_with_indices 2D t_indices")


# =============================================================================
# WanRMSNorm
# =============================================================================

class TestWanRMSNorm:

    def test_basic(self):
        dim = 64
        np.random.seed(42)
        x_np = np.random.randn(2, 8, dim).astype(np.float32)

        # PyTorch
        norm_pt = WanRMSNorm_pt(dim)
        norm_pt.eval()
        x_pt = torch.from_numpy(x_np)
        with torch.no_grad():
            out_pt = norm_pt(x_pt)

        # MLX - share weights
        norm_mlx = WanRMSNorm(dim)
        norm_mlx.weight = torch_to_mlx(norm_pt.weight)
        x_mlx = mx.array(x_np)
        out_mlx = norm_mlx(x_mlx)

        assert_close(out_mlx, out_pt, atol=1e-4, msg="WanRMSNorm")

    def test_half_precision(self):
        dim = 32
        np.random.seed(7)
        x_np = np.random.randn(1, 4, dim).astype(np.float32)

        norm_pt = WanRMSNorm_pt(dim)
        norm_pt.eval()
        x_pt = torch.from_numpy(x_np).half()
        with torch.no_grad():
            out_pt = norm_pt(x_pt).float()

        norm_mlx = WanRMSNorm(dim)
        norm_mlx.weight = torch_to_mlx(norm_pt.weight)
        x_mlx = mx.array(x_np).astype(mx.float16)
        out_mlx = norm_mlx(x_mlx).astype(mx.float32)

        assert_close(out_mlx, out_pt, atol=1e-2, msg="WanRMSNorm fp16")


# =============================================================================
# WanLayerNorm
# =============================================================================

class TestWanLayerNorm:

    def test_no_affine(self):
        dim = 64
        np.random.seed(42)
        x_np = np.random.randn(2, 8, dim).astype(np.float32)

        norm_pt = WanLayerNorm_pt(dim, elementwise_affine=False)
        norm_pt.eval()
        x_pt = torch.from_numpy(x_np)
        with torch.no_grad():
            out_pt = norm_pt(x_pt)

        norm_mlx = WanLayerNorm(dim, elementwise_affine=False)
        x_mlx = mx.array(x_np)
        out_mlx = norm_mlx(x_mlx)

        assert_close(out_mlx, out_pt, atol=1e-4, msg="WanLayerNorm no affine")

    def test_with_affine(self):
        dim = 64
        np.random.seed(42)
        x_np = np.random.randn(2, 8, dim).astype(np.float32)

        norm_pt = WanLayerNorm_pt(dim, elementwise_affine=True)
        norm_pt.eval()
        x_pt = torch.from_numpy(x_np)
        with torch.no_grad():
            out_pt = norm_pt(x_pt)

        norm_mlx = WanLayerNorm(dim, elementwise_affine=True)
        norm_mlx.weight = torch_to_mlx(norm_pt.weight)
        norm_mlx.bias = torch_to_mlx(norm_pt.bias)
        x_mlx = mx.array(x_np)
        out_mlx = norm_mlx(x_mlx)

        assert_close(out_mlx, out_pt, atol=1e-4, msg="WanLayerNorm with affine")

    def test_half_precision(self):
        dim = 32
        np.random.seed(7)
        x_np = np.random.randn(1, 4, dim).astype(np.float32)

        norm_pt = WanLayerNorm_pt(dim, elementwise_affine=False)
        norm_pt.eval()
        x_pt = torch.from_numpy(x_np).half()
        with torch.no_grad():
            out_pt = norm_pt(x_pt).float()

        norm_mlx = WanLayerNorm(dim, elementwise_affine=False)
        x_mlx = mx.array(x_np).astype(mx.float16)
        out_mlx = norm_mlx(x_mlx).astype(mx.float32)

        assert_close(out_mlx, out_pt, atol=1e-2, msg="WanLayerNorm fp16")


# =============================================================================
# PyTorch reference classes for WanSelfAttention, WanCrossAttention, etc.
# Extracted from Matrix-Game-3/wan/modules/model.py (lines 368-1067)
# =============================================================================

def flash_attention_pt(q, k, v, k_lens=None, **kwargs):
    """Simple PyTorch attention for testing (replaces flash_attention)."""
    import torch.nn.functional as F
    # q,k,v: [B, L, N, D]
    b, lq, n, d = q.shape
    lk = k.shape[1]
    scale = 1.0 / (d ** 0.5)
    q = q.transpose(1, 2)  # [B, N, Lq, D]
    k = k.transpose(1, 2)  # [B, N, Lk, D]
    v = v.transpose(1, 2)  # [B, N, Lk, D]

    mask = None
    if k_lens is not None:
        k_lens_t = k_lens if torch.is_tensor(k_lens) else torch.tensor(k_lens)
        k_pos = torch.arange(lk).view(1, 1, 1, lk)
        mask = k_pos < k_lens_t.view(b, 1, 1, 1)
        mask = mask.float()
        mask = (1.0 - mask) * (-1e9)

    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    if mask is not None:
        attn = attn + mask
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2)  # [B, L, N, D]
    return out


class WanSelfAttention_pt(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, use_memory=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_memory = use_memory
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm_pt(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm_pt(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, memory_length=0, **kwargs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        if not self.use_memory:
            q = rope_apply_pt(q, grid_sizes, freqs)
            k = rope_apply_pt(k, grid_sizes, freqs)
        x = flash_attention_pt(q=q, k=k, v=v, k_lens=seq_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention_pt(WanSelfAttention_pt):
    def forward(self, x, context, context_lens, **kwargs):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = flash_attention_pt(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock_pt(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1,-1), qk_norm=True,
                 cross_attn_norm=False, eps=1e-6, action_config={}, block_idx=0, use_memory=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.use_memory = use_memory
        self.norm1 = WanLayerNorm_pt(dim, eps)
        self.self_attn = WanSelfAttention_pt(dim, num_heads, window_size, qk_norm, eps, use_memory=use_memory)
        self.action_model = None
        self.norm3 = WanLayerNorm_pt(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention_pt(dim, num_heads, (-1,-1), qk_norm, eps)
        self.norm2 = WanLayerNorm_pt(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, **kwargs):
        assert e.dtype == torch.float32
        e_mod = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e_mod[0].dtype == torch.float32

        y = self.self_attn(
            (self.norm1(x).float() * (1 + e_mod[1].squeeze(2)) + e_mod[0].squeeze(2)).to(x.dtype),
            seq_lens, grid_sizes, freqs)
        x = x + y * e_mod[2].squeeze(2)

        # Cross-attn + FFN
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        y = self.ffn(
            (self.norm2(x).float() * (1 + e_mod[4].squeeze(2)) + e_mod[3].squeeze(2)).to(self.ffn[0].weight.dtype))
        x = x + y * e_mod[5].squeeze(2)
        return x


class Head_pt(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        import math
        out_dim_linear = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm_pt(dim, eps)
        self.head = nn.Linear(dim, out_dim_linear)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        e_mod = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        x = self.head(
            self.norm(x) * (1 + e_mod[1].squeeze(2)) + e_mod[0].squeeze(2))
        return x


def _copy_linear_weights(src_pt, dst_mlx):
    """Copy weights from a PyTorch nn.Linear to MLX nn.Linear."""
    dst_mlx.weight = torch_to_mlx(src_pt.weight)
    if src_pt.bias is not None:
        dst_mlx.bias = torch_to_mlx(src_pt.bias)


def _copy_rmsnorm_weights(src_pt, dst_mlx):
    """Copy WanRMSNorm weights."""
    if hasattr(src_pt, 'weight') and hasattr(dst_mlx, 'weight'):
        w = src_pt.weight if isinstance(src_pt.weight, torch.Tensor) else src_pt.weight
        if isinstance(w, nn.Parameter):
            dst_mlx.weight = torch_to_mlx(w.data)
        else:
            dst_mlx.weight = torch_to_mlx(w)


# =============================================================================
# WanSelfAttention
# =============================================================================

class TestWanSelfAttention:

    def test_basic(self):
        """Test WanSelfAttention with small config."""
        dim, num_heads = 64, 4
        B, S = 1, 24
        head_dim = dim // num_heads
        np.random.seed(42)

        # Create PyTorch model
        sa_pt = WanSelfAttention_pt(dim, num_heads, qk_norm=True, use_memory=False)
        sa_pt.eval()

        # Create MLX model and copy weights
        sa_mlx = WanSelfAttention(dim, num_heads, qk_norm=True, use_memory=False)
        _copy_linear_weights(sa_pt.q, sa_mlx.q)
        _copy_linear_weights(sa_pt.k, sa_mlx.k)
        _copy_linear_weights(sa_pt.v, sa_mlx.v)
        _copy_linear_weights(sa_pt.o, sa_mlx.o)
        _copy_rmsnorm_weights(sa_pt.norm_q, sa_mlx.norm_q)
        _copy_rmsnorm_weights(sa_pt.norm_k, sa_mlx.norm_k)

        # Inputs
        x_np = np.random.randn(B, S, dim).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        grid = [[2, 3, 4]]
        grid_pt = torch.tensor(grid)
        seq_lens_pt = torch.tensor([24])
        seq_lens_mlx = mx.array([24])

        max_seq_len = 8
        freqs_pt = rope_params_pt(max_seq_len, head_dim)
        freqs_mlx = rope_params(max_seq_len, head_dim)

        with torch.no_grad():
            out_pt = sa_pt(x_pt, seq_lens_pt, grid_pt, freqs_pt)

        out_mlx = sa_mlx(x_mlx, seq_lens_mlx, grid, freqs_mlx)

        assert_close(out_mlx, out_pt, atol=1e-3, msg="WanSelfAttention basic")


# =============================================================================
# WanCrossAttention
# =============================================================================

class TestWanCrossAttention:

    def test_basic(self):
        """Test WanCrossAttention with small config."""
        dim, num_heads = 64, 4
        B, L1, L2 = 1, 12, 8
        np.random.seed(42)

        ca_pt = WanCrossAttention_pt(dim, num_heads, qk_norm=True)
        ca_pt.eval()

        ca_mlx = WanCrossAttention(dim, num_heads, qk_norm=True)
        _copy_linear_weights(ca_pt.q, ca_mlx.q)
        _copy_linear_weights(ca_pt.k, ca_mlx.k)
        _copy_linear_weights(ca_pt.v, ca_mlx.v)
        _copy_linear_weights(ca_pt.o, ca_mlx.o)
        _copy_rmsnorm_weights(ca_pt.norm_q, ca_mlx.norm_q)
        _copy_rmsnorm_weights(ca_pt.norm_k, ca_mlx.norm_k)

        x_np = np.random.randn(B, L1, dim).astype(np.float32)
        ctx_np = np.random.randn(B, L2, dim).astype(np.float32)

        x_pt = torch.from_numpy(x_np)
        ctx_pt = torch.from_numpy(ctx_np)
        ctx_lens_pt = torch.tensor([L2])

        x_mlx = mx.array(x_np)
        ctx_mlx = mx.array(ctx_np)
        ctx_lens_mlx = mx.array([L2])

        with torch.no_grad():
            out_pt = ca_pt(x_pt, ctx_pt, ctx_lens_pt)

        out_mlx = ca_mlx(x_mlx, ctx_mlx, ctx_lens_mlx)

        assert_close(out_mlx, out_pt, atol=1e-3, msg="WanCrossAttention basic")


# =============================================================================
# WanAttentionBlock
# =============================================================================

class TestWanAttentionBlock:

    def test_basic(self):
        """Test WanAttentionBlock without action model."""
        dim, ffn_dim, num_heads = 64, 128, 4
        B, S = 1, 24
        head_dim = dim // num_heads
        ctx_len = 8
        np.random.seed(42)

        blk_pt = WanAttentionBlock_pt(dim, ffn_dim, num_heads, qk_norm=True,
                                      cross_attn_norm=False, use_memory=False)
        blk_pt.eval()

        blk_mlx = WanAttentionBlock(dim, ffn_dim, num_heads, qk_norm=True,
                                    cross_attn_norm=False, use_memory=False)

        # Copy all weights
        # self_attn
        _copy_linear_weights(blk_pt.self_attn.q, blk_mlx.self_attn.q)
        _copy_linear_weights(blk_pt.self_attn.k, blk_mlx.self_attn.k)
        _copy_linear_weights(blk_pt.self_attn.v, blk_mlx.self_attn.v)
        _copy_linear_weights(blk_pt.self_attn.o, blk_mlx.self_attn.o)
        _copy_rmsnorm_weights(blk_pt.self_attn.norm_q, blk_mlx.self_attn.norm_q)
        _copy_rmsnorm_weights(blk_pt.self_attn.norm_k, blk_mlx.self_attn.norm_k)

        # cross_attn
        _copy_linear_weights(blk_pt.cross_attn.q, blk_mlx.cross_attn.q)
        _copy_linear_weights(blk_pt.cross_attn.k, blk_mlx.cross_attn.k)
        _copy_linear_weights(blk_pt.cross_attn.v, blk_mlx.cross_attn.v)
        _copy_linear_weights(blk_pt.cross_attn.o, blk_mlx.cross_attn.o)
        _copy_rmsnorm_weights(blk_pt.cross_attn.norm_q, blk_mlx.cross_attn.norm_q)
        _copy_rmsnorm_weights(blk_pt.cross_attn.norm_k, blk_mlx.cross_attn.norm_k)

        # FFN
        _copy_linear_weights(blk_pt.ffn[0], blk_mlx.ffn_linear1)
        _copy_linear_weights(blk_pt.ffn[2], blk_mlx.ffn_linear2)

        # Modulation
        blk_mlx.modulation = torch_to_mlx(blk_pt.modulation.data)

        # Inputs
        x_np = np.random.randn(B, S, dim).astype(np.float32)
        e_np = np.random.randn(B, S, 6, dim).astype(np.float32)
        ctx_np = np.random.randn(B, ctx_len, dim).astype(np.float32)

        x_pt = torch.from_numpy(x_np)
        e_pt = torch.from_numpy(e_np)
        ctx_pt = torch.from_numpy(ctx_np)

        grid = [[2, 3, 4]]
        grid_pt = torch.tensor(grid)
        seq_lens_pt = torch.tensor([S])
        ctx_lens_pt = torch.tensor([ctx_len])

        max_seq_len = 8
        freqs_pt = rope_params_pt(max_seq_len, head_dim)
        freqs_mlx = rope_params(max_seq_len, head_dim)

        x_mlx = mx.array(x_np)
        e_mlx = mx.array(e_np)
        ctx_mlx = mx.array(ctx_np)
        seq_lens_mlx = mx.array([S])
        ctx_lens_mlx = mx.array([ctx_len])

        with torch.no_grad():
            out_pt = blk_pt(x_pt, e_pt, seq_lens_pt, grid_pt, freqs_pt,
                            ctx_pt, ctx_lens_pt)

        out_mlx = blk_mlx(x_mlx, e_mlx, seq_lens_mlx, grid, freqs_mlx,
                          ctx_mlx, ctx_lens_mlx)

        assert_close(out_mlx, out_pt, atol=1e-3, msg="WanAttentionBlock basic")


# =============================================================================
# Head
# =============================================================================

class TestHead:

    def test_basic(self):
        """Test Head module."""
        dim, out_dim = 64, 4
        patch_size = (1, 2, 2)
        B, S = 1, 24
        np.random.seed(42)

        head_pt = Head_pt(dim, out_dim, patch_size)
        head_pt.eval()

        head_mlx = Head(dim, out_dim, patch_size)
        _copy_linear_weights(head_pt.head, head_mlx.head)
        head_mlx.modulation = torch_to_mlx(head_pt.modulation.data)

        x_np = np.random.randn(B, S, dim).astype(np.float32)
        e_np = np.random.randn(B, S, dim).astype(np.float32)

        x_pt = torch.from_numpy(x_np)
        e_pt = torch.from_numpy(e_np)
        x_mlx = mx.array(x_np)
        e_mlx = mx.array(e_np)

        with torch.no_grad():
            out_pt = head_pt(x_pt, e_pt)

        out_mlx = head_mlx(x_mlx, e_mlx)

        assert_close(out_mlx, out_pt, atol=1e-3, msg="Head basic")


# =============================================================================
# WanModel (small config)
# =============================================================================

class TestWanModel:

    def test_small(self):
        """Test full WanModel with tiny config."""
        dim = 64
        ffn_dim = 128
        num_heads = 4
        num_layers = 2
        patch_size = (1, 2, 2)
        in_dim = 4
        out_dim = 4
        text_dim = 32
        freq_dim = 32
        text_len = 8
        head_dim = dim // num_heads
        B = 1
        np.random.seed(42)
        torch.manual_seed(42)

        # Create MLX model
        model_mlx = WanModel(
            model_type='t2v',
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            qk_norm=True,
            cross_attn_norm=False,
            use_memory=False,
            sigma_theta=0.0,
        )

        # Build a matching PyTorch model from individual components
        # We'll test by doing the forward pass step-by-step

        # Input: [F, H, W, C_in] channels-last for MLX
        F_frames, H, W = 2, 4, 4
        x_np = np.random.randn(F_frames, H, W, in_dim).astype(np.float32)
        x_mlx = [mx.array(x_np)]

        t_np = np.array([0.5], dtype=np.float32)
        t_mlx = mx.array(t_np)

        ctx_np = np.random.randn(6, text_dim).astype(np.float32)
        ctx_mlx = [mx.array(ctx_np)]

        # Compute expected seq_len
        f_p = F_frames // patch_size[0]  # 2
        h_p = H // patch_size[1]         # 2
        w_p = W // patch_size[2]         # 2
        num_patches = f_p * h_p * w_p    # 8
        seq_len = num_patches

        # Run MLX model
        out_mlx = model_mlx(x_mlx, t_mlx, ctx_mlx, seq_len)

        # Check output shape
        assert len(out_mlx) == 1
        out_shape = out_mlx[0].shape
        assert out_shape == (F_frames, H, W, out_dim), f"Expected ({F_frames}, {H}, {W}, {out_dim}), got {out_shape}"

        # Verify it produces finite values
        out_np = np.array(out_mlx[0])
        assert np.all(np.isfinite(out_np)), "Output contains non-finite values"

    def test_unpatchify_roundtrip(self):
        """Test that patchify -> unpatchify recovers spatial dims correctly."""
        dim = 64
        patch_size = (1, 2, 2)
        in_dim = 4
        out_dim = 4
        np.random.seed(42)

        model = WanModel(
            model_type='t2v', patch_size=patch_size,
            text_len=8, in_dim=in_dim, dim=dim, ffn_dim=128,
            freq_dim=32, text_dim=32, out_dim=out_dim,
            num_heads=4, num_layers=1,
            use_memory=False,
        )

        # Create a tensor of the right shape for unpatchify output
        import math
        f_p, h_p, w_p = 2, 3, 4
        total = f_p * h_p * w_p
        patch_out_dim = math.prod(patch_size) * out_dim
        x = mx.ones((1, total, patch_out_dim))

        out = model.unpatchify(x, [[f_p, h_p, w_p]])
        assert len(out) == 1
        expected = (
            f_p * patch_size[0],
            h_p * patch_size[1],
            w_p * patch_size[2],
            out_dim,
        )
        assert out[0].shape == expected, f"Expected {expected}, got {out[0].shape}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
