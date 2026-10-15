"""Strict DiT loading from the mlx-forge pack (MLX port of matrix_game/inference/loading.py)."""
import json
from pathlib import Path

import mlx.core as mx

from matrix_game_mlx.inference.wan.modules.causal_model import CausalWanModel
from matrix_game_mlx.inference.wrappers import WanDiffusionWrapper

_ARCH_KEYS = ("model_type", "patch_size", "text_len", "in_dim", "dim", "ffn_dim", "freq_dim", "text_dim",
              "out_dim", "num_heads", "num_layers", "qk_norm", "cross_attn_norm", "eps")


def build_dit(pack_dir: Path, component: str, dtype=mx.bfloat16, timestep_shift: float = 5.0) -> WanDiffusionWrapper:
    """Build the DiT from `config.json` and load `<component>.safetensors` with strict key matching."""
    cfg = json.loads((Path(pack_dir) / "config.json").read_text())
    model = CausalWanModel(**{k: cfg[k] for k in _ARCH_KEYS})
    weights = mx.load(str(Path(pack_dir) / f"{component}.safetensors"))
    prefix = component + "."
    clean = {k[len(prefix):] if k.startswith(prefix) else k: v.astype(dtype) for k, v in weights.items()}
    model.load_weights(list(clean.items()), strict=True)
    mx.eval(model.parameters())
    return WanDiffusionWrapper(model, timestep_shift=timestep_shift)
