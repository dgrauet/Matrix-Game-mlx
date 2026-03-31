"""Validation tests for positional embeddings (RoPE) — MLX vs PyTorch."""

import sys
import os
import importlib
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import PyTorch version first (before MLX version pollutes sys.modules)
# Load directly from file path to avoid module name collision.
# ---------------------------------------------------------------------------
PYTORCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Matrix-Game-3'))
_torch_posemb_path = os.path.join(PYTORCH_ROOT, 'wan', 'modules', 'posemb_layers.py')

# Patch torch.cuda.current_device so the PyTorch reference works on CPU
_original_current_device = getattr(torch.cuda, 'current_device', None)
torch.cuda.current_device = lambda: 'cpu'

spec = importlib.util.spec_from_file_location('torch_posemb_layers', _torch_posemb_path)
torch_posemb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_posemb)

# Restore original
if _original_current_device is not None:
    torch.cuda.current_device = _original_current_device

torch_get_1d_rotary_pos_embed = torch_posemb.get_1d_rotary_pos_embed
torch_get_nd_rotary_pos_embed = torch_posemb.get_nd_rotary_pos_embed
torch_apply_rotary_emb = torch_posemb.apply_rotary_emb

# ---------------------------------------------------------------------------
# Import MLX version
# ---------------------------------------------------------------------------
MLX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MLX_ROOT)

import mlx.core as mx
from wan.modules.posemb_layers import (
    get_1d_rotary_pos_embed as mlx_get_1d_rotary_pos_embed,
    get_nd_rotary_pos_embed as mlx_get_nd_rotary_pos_embed,
    apply_rotary_emb as mlx_apply_rotary_emb,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _assert_close(mlx_arr, torch_tensor, atol=1e-5, msg=""):
    mlx_np = np.array(mlx_arr, copy=False).astype(np.float32)
    torch_np = torch_tensor.detach().cpu().float().numpy()
    np.testing.assert_allclose(mlx_np, torch_np, atol=atol, rtol=1e-5, err_msg=msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPositionalEmbeddings(unittest.TestCase):

    def test_get_1d_rotary_pos_embed(self):
        """Compare 1D rotary pos embed: MLX vs PyTorch (dim=64, pos=16, use_real=True)."""
        dim = 64
        pos = 16

        # PyTorch (on CPU via patch)
        with patch.object(torch.cuda, 'current_device', return_value='cpu'):
            torch_cos, torch_sin = torch_get_1d_rotary_pos_embed(
                dim, pos, use_real=True
            )

        # MLX
        mlx_cos, mlx_sin = mlx_get_1d_rotary_pos_embed(dim, pos, use_real=True)
        mx.eval(mlx_cos, mlx_sin)

        _assert_close(mlx_cos, torch_cos, atol=1e-5, msg="1d cos mismatch")
        _assert_close(mlx_sin, torch_sin, atol=1e-5, msg="1d sin mismatch")

        # Check shapes
        self.assertEqual(mlx_cos.shape, (pos, dim), f"cos shape: {mlx_cos.shape}")
        self.assertEqual(mlx_sin.shape, (pos, dim), f"sin shape: {mlx_sin.shape}")

    def test_get_nd_rotary_pos_embed(self):
        """Compare n-D rotary pos embed: MLX vs PyTorch."""
        rope_dim_list = [20, 44, 44]
        rope_sizes = [4, 22, 40]

        with patch.object(torch.cuda, 'current_device', return_value='cpu'):
            torch_cos, torch_sin = torch_get_nd_rotary_pos_embed(
                rope_dim_list, rope_sizes, use_real=True
            )

        mlx_cos, mlx_sin = mlx_get_nd_rotary_pos_embed(
            rope_dim_list, rope_sizes, use_real=True
        )
        mx.eval(mlx_cos, mlx_sin)

        _assert_close(mlx_cos, torch_cos, atol=1e-5, msg="nd cos mismatch")
        _assert_close(mlx_sin, torch_sin, atol=1e-5, msg="nd sin mismatch")

        total_dim = sum(rope_dim_list)
        total_seq = 4 * 22 * 40
        self.assertEqual(mlx_cos.shape, (total_seq, total_dim))
        self.assertEqual(mlx_sin.shape, (total_seq, total_dim))

    def test_apply_rotary_emb(self):
        """Compare apply_rotary_emb: MLX vs PyTorch (B=1, S=16, H=8, D=64)."""
        B, S, H, D = 1, 16, 8, 64

        # Create deterministic random inputs
        rng = np.random.RandomState(42)
        q_np = rng.randn(B, S, H, D).astype(np.float32)
        k_np = rng.randn(B, S, H, D).astype(np.float32)

        # Generate RoPE (PyTorch)
        with patch.object(torch.cuda, 'current_device', return_value='cpu'):
            torch_cos, torch_sin = torch_get_1d_rotary_pos_embed(
                D, S, use_real=True
            )

        # Generate RoPE (MLX)
        mlx_cos, mlx_sin = mlx_get_1d_rotary_pos_embed(D, S, use_real=True)
        mx.eval(mlx_cos, mlx_sin)

        # PyTorch apply
        torch_q = torch.from_numpy(q_np)
        torch_k = torch.from_numpy(k_np)
        torch_qo, torch_ko = torch_apply_rotary_emb(
            torch_q, torch_k, (torch_cos, torch_sin), head_first=False
        )

        # MLX apply
        mlx_q = mx.array(q_np)
        mlx_k = mx.array(k_np)
        mlx_qo, mlx_ko = mlx_apply_rotary_emb(
            mlx_q, mlx_k, (mlx_cos, mlx_sin), head_first=False
        )
        mx.eval(mlx_qo, mlx_ko)

        _assert_close(mlx_qo, torch_qo, atol=1e-4, msg="apply_rotary q mismatch")
        _assert_close(mlx_ko, torch_ko, atol=1e-4, msg="apply_rotary k mismatch")


if __name__ == '__main__':
    unittest.main()
