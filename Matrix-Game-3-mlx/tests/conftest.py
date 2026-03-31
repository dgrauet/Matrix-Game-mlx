"""Shared test fixtures for MLX vs PyTorch validation."""
import sys
import os
import numpy as np

MLX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PYTORCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Matrix-Game-3'))


def torch_to_mlx(tensor):
    """Convert PyTorch tensor to MLX array."""
    import mlx.core as mx
    return mx.array(tensor.detach().cpu().float().numpy())


def mlx_to_torch(array):
    """Convert MLX array to PyTorch tensor."""
    import torch
    return torch.from_numpy(np.array(array, copy=False).astype(np.float32))


def assert_close(mlx_out, torch_out, atol=1e-4, rtol=1e-5, msg=""):
    """Assert MLX and PyTorch outputs are numerically close."""
    mlx_np = np.array(mlx_out, copy=False).astype(np.float32)
    if hasattr(torch_out, 'numpy'):
        torch_np = torch_out.detach().cpu().float().numpy()
    else:
        torch_np = np.array(torch_out, dtype=np.float32)
    np.testing.assert_allclose(mlx_np, torch_np, atol=atol, rtol=rtol, err_msg=msg)
