"""Head-wise tensor parallelism for the WanModel DiT.

Why tensor parallelism instead of the reference's Ulysses sequence
parallelism (``Matrix-Game-3/wan/distributed/ulysses.py``):

- Ulysses requires an ``all_to_all`` collective, which ``mx.distributed``
  does not provide (available ops: ``all_sum``, ``all_gather``, ``send``,
  ``recv``).
- Communication analysis at 720p (13,200 patches x 3,072 dim, bf16 ~= 77 MB
  per activation): tensor parallelism needs 2 ``all_sum`` per block (after
  the attention output projection and after the FFN), i.e. ~4.6 GB per
  denoising step — negligible next to per-step compute time over a
  Thunderbolt ring or JACCL (RDMA) link.

Sharding scheme (mlx-lm style, per transformer block):

- ``q``/``k``/``v``: output-sharded by heads (24 heads -> 24/N per rank).
- ``o``: input-sharded; partial outputs combined with ``all_sum``.
- ``norm_q``/``norm_k``: weight sliced per rank; the RMS statistic is
  computed globally via an ``all_sum`` of local sums of squares (these
  norms operate over the full hidden dim, which is now sharded).
- ``ffn_linear1``: output-sharded; ``ffn_linear2``: input-sharded +
  ``all_sum``.
- Per-head RoPE frequency tables (``sigma_theta > 0``) sliced to each
  rank's head range.

Everything else (patch/text/time embeddings, LayerNorms, modulation,
ActionModule, camera-injection layers, head) is replicated: every rank
computes them identically on the replicated hidden states.
"""

import mlx.core as mx

from ..modules.model import WanModel, WanSelfAttention
from .util import shard_linear_cols, shard_linear_rows


def _shard_attention(attn: WanSelfAttention, group: mx.distributed.Group) -> None:
    """Shard one attention module (self- or cross-) across the group."""
    rank, size = group.rank(), group.size()
    assert attn.num_heads % size == 0, (
        f"num_heads {attn.num_heads} not divisible by world size {size}"
    )

    shard_linear_rows(attn.q, rank, size)
    shard_linear_rows(attn.k, rank, size)
    shard_linear_rows(attn.v, rank, size)
    shard_linear_cols(attn.o, rank, size)

    # norm_q / norm_k normalize over the full hidden dim, which is now
    # sharded: slice the affine weight and switch the RMS statistic to the
    # distributed (global) path. ``norm.dim`` keeps the full dimension.
    local_dim = attn.dim // size
    for norm in (attn.norm_q, attn.norm_k):
        if norm is not None:
            norm.weight = norm.weight[rank * local_dim:(rank + 1) * local_dim]
            norm.tp_group = group

    attn.num_heads = attn.num_heads // size
    attn.tp_group = group


def shard_model(model: WanModel, group: mx.distributed.Group) -> WanModel:
    """Shard a WanModel in place for tensor-parallel inference.

    Must be called after weights are loaded and before the first forward
    pass. Slicing lazy arrays is cheap; ``mx.eval`` at the end materializes
    only each rank's shard.

    Args:
        model: The (fully loaded) DiT to shard.
        group: Distributed group. A group of size 1 makes this a no-op.

    Returns:
        The same model, sharded across ``group``.
    """
    size = group.size()
    if size == 1:
        return model
    rank = group.rank()

    assert model.num_heads % size == 0, (
        f"num_heads {model.num_heads} not divisible by world size {size}"
    )
    assert model.ffn_dim % size == 0, (
        f"ffn_dim {model.ffn_dim} not divisible by world size {size}"
    )

    for block in model.blocks:
        _shard_attention(block.self_attn, group)
        _shard_attention(block.cross_attn, group)
        shard_linear_rows(block.ffn_linear1, rank, size)
        shard_linear_cols(block.ffn_linear2, rank, size)
        block.tp_group = group

    # Per-head RoPE tables [num_heads, max_seq_len, C//2] (sigma_theta > 0):
    # each rank keeps its contiguous head range. Shared 2-D tables need no
    # slicing — all heads use the same frequencies.
    cos_all, sin_all = model.freqs
    if cos_all.ndim == 3:
        n_local = model.num_heads // size
        model.freqs = (
            cos_all[rank * n_local:(rank + 1) * n_local],
            sin_all[rank * n_local:(rank + 1) * n_local],
        )

    mx.eval(model.parameters())
    return model
