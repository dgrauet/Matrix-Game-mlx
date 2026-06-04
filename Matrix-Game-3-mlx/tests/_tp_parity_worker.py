"""Worker script for distributed tensor-parallel parity testing.

Launched by tests/test_tensor_parallel.py via ``mlx.launch -n 2``. Each rank
builds the same tiny WanModel (fixed seed), computes the unsharded reference
output locally, then shards the model across the group and compares the
distributed output against the reference. Exits non-zero on divergence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from wan.distributed.tensor_parallel import shard_model
from wan.distributed.util import init_distributed_group
from wan.modules.model import WanModel

TOL = 5e-5


def build_model() -> WanModel:
    mx.random.seed(0)
    return WanModel(
        model_type='t2v',
        patch_size=(1, 2, 2),
        in_dim=8,
        dim=96,
        ffn_dim=192,
        freq_dim=32,
        text_dim=64,
        out_dim=8,
        num_heads=4,
        num_layers=2,
        use_memory=True,
        sigma_theta=0.8,
        action_config={},
    )


def main() -> int:
    group = init_distributed_group()
    rank, size = group.rank(), group.size()
    if size < 2:
        print("expected world size >= 2", flush=True)
        return 1

    mx.random.seed(42)
    x = [mx.random.normal((4, 8, 8, 8))]
    context = [mx.random.normal((10, 64))]
    t = mx.full((1,), 500.0)

    # Unsharded reference, computed identically on every rank.
    model = build_model()
    ref = model(x, t, context, seq_len=64)
    mx.eval(ref)

    # Sharded model: fresh instance with identical weights (same seed).
    model_tp = shard_model(build_model(), group)
    out = model_tp(x, t, context, seq_len=64)
    mx.eval(out)

    max_diff = mx.abs(ref[0] - out[0]).max().item()
    print(f"rank {rank}/{size}: max_diff={max_diff:.3e}", flush=True)
    return 0 if max_diff < TOL else 1


if __name__ == "__main__":
    sys.exit(main())
