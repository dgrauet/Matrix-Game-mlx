"""Validation tests for wan/distributed tensor parallelism.

Unit tests check the weight-slicing math single-process. The integration
test launches 2 real ranks over the ring backend on localhost via
``mlx.launch`` and asserts sharded vs unsharded model parity — no second
machine required.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wan.distributed.util import shard_linear_cols, shard_linear_rows

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_linear(out_dim: int, in_dim: int) -> nn.Linear:
    mx.random.seed(0)
    linear = nn.Linear(in_dim, out_dim)
    linear.weight = mx.random.normal((out_dim, in_dim))
    linear.bias = mx.random.normal((out_dim,))
    return linear


def test_shard_linear_rows_reconstructs_full_output():
    """Concatenated row-shard outputs equal the full layer output."""
    full = _make_linear(8, 6)
    x = mx.random.normal((2, 5, 6))
    expected = full(x)

    parts = []
    for rank in range(2):
        shard = _make_linear(8, 6)
        shard_linear_rows(shard, rank, 2)
        assert shard.weight.shape == (4, 6)
        parts.append(shard(x))

    out = mx.concatenate(parts, axis=-1)
    assert mx.abs(out - expected).max().item() < 1e-6


def test_shard_linear_cols_partials_sum_to_full_output():
    """Summed col-shard partial outputs equal the full layer output.

    Mirrors the runtime behavior where partials are combined by all_sum;
    the bias must be counted exactly once (kept on rank 0 only).
    """
    full = _make_linear(6, 8)
    x = mx.random.normal((2, 5, 8))
    expected = full(x)

    out = mx.zeros((2, 5, 6))
    for rank in range(2):
        shard = _make_linear(6, 8)
        shard_linear_cols(shard, rank, 2)
        assert shard.weight.shape == (6, 4)
        out = out + shard(x[..., rank * 4:(rank + 1) * 4])

    assert mx.abs(out - expected).max().item() < 1e-6


def test_shard_rejects_quantized_layers():
    """Slicing packed quantized weights would be silently wrong — reject."""
    linear = nn.QuantizedLinear(64, 64, group_size=32, bits=4)
    with pytest.raises(NotImplementedError):
        shard_linear_rows(linear, 0, 2)
    with pytest.raises(NotImplementedError):
        shard_linear_cols(linear, 0, 2)


def test_shard_model_singleton_group_is_noop():
    """A group of size 1 must leave the model untouched."""
    from wan.distributed.tensor_parallel import shard_model
    from wan.distributed.util import init_distributed_group
    from wan.modules.model import WanModel

    mx.random.seed(0)
    model = WanModel(
        model_type='t2v', patch_size=(1, 2, 2), in_dim=8, dim=96,
        ffn_dim=192, freq_dim=32, text_dim=64, out_dim=8, num_heads=4,
        num_layers=2, use_memory=True, sigma_theta=0.8, action_config={},
    )
    group = init_distributed_group()
    if group.size() != 1:
        pytest.skip("test requires a non-distributed run")

    shard_model(model, group)
    assert model.blocks[0].self_attn.num_heads == 4
    assert model.blocks[0].self_attn.tp_group is None


@pytest.mark.skipif(
    shutil.which("mlx.launch") is None, reason="mlx.launch not available"
)
def test_tensor_parallel_parity_2_ranks():
    """Sharded (2 ranks, ring backend on localhost) == unsharded output."""
    result = subprocess.run(
        [
            "mlx.launch", "-n", "2", "--backend", "ring",
            str(PROJECT_ROOT / "tests" / "_tp_parity_worker.py"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"TP parity failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "max_diff" in result.stdout
