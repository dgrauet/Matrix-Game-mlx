"""Distributed communication utilities for MLX tensor parallelism.

MLX counterpart of the reference ``wan/distributed/util.py``. The reference
uses ``torch.distributed`` with NCCL and exposes ``all_to_all`` for Ulysses
sequence parallelism; ``mx.distributed`` has no ``all_to_all`` primitive, so
this port uses head-wise tensor parallelism instead (see
``tensor_parallel.py`` for the rationale). This module owns process-group
initialization and the weight-slicing helpers shared by the sharding code.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

_GROUP: Optional[mx.distributed.Group] = None


def init_distributed_group(backend: str = "any") -> mx.distributed.Group:
    """Initialize (or return) the global distributed group.

    Args:
        backend: One of ``"ring"``, ``"mpi"``, ``"jaccl"`` or ``"any"``.
            With ``"any"``, MLX tries each available backend and falls back
            to a singleton group (size 1) when none is available, so this is
            always safe to call — including single-device runs.

    Returns:
        The initialized distributed group.
    """
    global _GROUP
    if _GROUP is None:
        _GROUP = mx.distributed.init(backend=backend)
    return _GROUP


def get_group() -> Optional[mx.distributed.Group]:
    """Return the global distributed group, or None if not initialized."""
    return _GROUP


def get_rank() -> int:
    """Return this process's rank (0 if not distributed)."""
    return _GROUP.rank() if _GROUP is not None else 0


def get_world_size() -> int:
    """Return the number of processes (1 if not distributed)."""
    return _GROUP.size() if _GROUP is not None else 1


def _check_not_quantized(linear: nn.Linear) -> None:
    """Reject quantized layers — slicing packed weights would be silently wrong."""
    if isinstance(linear, nn.QuantizedLinear):
        raise NotImplementedError(
            "Tensor parallelism does not support quantized weights; "
            "load fp16/bf16 weights for distributed inference."
        )


def shard_linear_rows(linear: nn.Linear, rank: int, size: int) -> None:
    """Shard a Linear along its output dimension (rows of ``weight``).

    Used for projections whose *outputs* are partitioned across ranks
    (q/k/v, ffn_linear1). Each rank keeps its contiguous slice of output
    features, bias included.

    Args:
        linear: Layer to shard in place.
        rank: This process's rank.
        size: World size. Must divide the output dimension.
    """
    _check_not_quantized(linear)
    out_dim = linear.weight.shape[0]
    assert out_dim % size == 0, (
        f"output dim {out_dim} not divisible by world size {size}"
    )
    k = out_dim // size
    linear.weight = linear.weight[rank * k:(rank + 1) * k, :]
    if "bias" in linear:
        linear.bias = linear.bias[rank * k:(rank + 1) * k]


def shard_linear_cols(linear: nn.Linear, rank: int, size: int) -> None:
    """Shard a Linear along its input dimension (columns of ``weight``).

    Used for projections whose *inputs* are partitioned across ranks (o,
    ffn_linear2). Partial outputs are later combined with ``all_sum``, so
    only rank 0 keeps the bias — otherwise it would be added ``size`` times.

    Args:
        linear: Layer to shard in place.
        rank: This process's rank.
        size: World size. Must divide the input dimension.
    """
    _check_not_quantized(linear)
    in_dim = linear.weight.shape[1]
    assert in_dim % size == 0, (
        f"input dim {in_dim} not divisible by world size {size}"
    )
    k = in_dim // size
    linear.weight = linear.weight[:, rank * k:(rank + 1) * k]
    if "bias" in linear and rank != 0:
        linear.bias = mx.zeros_like(linear.bias)
