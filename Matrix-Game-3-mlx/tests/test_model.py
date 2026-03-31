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
)


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
