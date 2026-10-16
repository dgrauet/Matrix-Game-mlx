"""Inference schedule helpers (MLX port of matrix_game/inference/rollout.py, inference subset)."""
from typing import Optional

import numpy as np


def warp_steps(scheduler, steps) -> Optional[list]:
    """Official warp_denoising_step: 1000/750/500/250 -> scheduler.timesteps[1000 - s] (shift-5 grid), trailing 0 dropped."""
    if steps is None:
        return None
    timesteps = np.concatenate([np.array(scheduler.timesteps, dtype=np.float32), np.array([0.0], np.float32)])
    out = [float(timesteps[1000 - int(s)]) for s in steps]
    if out[-1] == 0:
        out = out[:-1]
    return out


def set_attention_window(generator, local_attn_size: int = -1, sink_size: int = 0) -> None:
    """Set the sliding window (W) and sink size (S) on every causal self-attention and on the model.
    Duck-typed: `generator.model` (or `generator` itself) must expose `.blocks`, each with `.self_attn`."""
    model = getattr(generator, "model", generator)
    model.local_attn_size = local_attn_size
    for block in model.blocks:
        block.self_attn.local_attn_size = local_attn_size
        block.self_attn.sink_size = sink_size
