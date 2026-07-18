"""VAE configuration and loading utilities for MLX.

Simplified version of the PyTorch reference — no async VAE worker,
no torch.compile. LightVAE variants are supported (pruned student decoder
+ full-VAE teacher encoder, weights pre-converted by mlx-forge).
"""

import logging
import os
from typing import Any, Dict, Optional

import mlx.core as mx

from wan.modules.vae2_2 import Wan2_2_VAE

logger = logging.getLogger(__name__)


def get_vae_config(args: Any = None) -> Dict[str, Any]:
    """Get VAE configuration from args or defaults.

    Args:
        args: Optional namespace with ``vae_type`` attribute.

    Returns:
        Dict with keys ``z_dim``, ``c_dim``, ``dim_mult``,
        ``temperal_downsample``, and ``vae_type``.
    """
    config: Dict[str, Any] = {
        "z_dim": 48,
        "c_dim": 160,
        "dec_dim": 256,
        "dim_mult": [1, 2, 4, 4],
        "temperal_downsample": [False, True, True],
        "vae_type": "wan",
    }
    if args is not None:
        config["vae_type"] = getattr(args, "vae_type", "wan")
    return config


# Mirrors the reference vae_config: LightVAE variants map to their pruned
# student checkpoint (plus the full VAE as teacher encoder).
_VAE_FILES = {
    "wan": ("vae.safetensors", None),
    "mg_lightvae": ("vae_lightvae.safetensors", 0.5),
    "mg_lightvae_v2": ("vae_lightvae_v2.safetensors", 0.75),
}


def load_vae(
    model_path: Optional[str] = None,
    vae_type: str = "wan",
    dtype: mx.Dtype = mx.float32,
    **kwargs: Any,
) -> Wan2_2_VAE:
    """Load a VAE model from mlx-forge converted weights.

    Args:
        model_path: Path to the converted model directory, or directly to a
            VAE safetensors file (``vae_type='wan'`` only).
        vae_type: VAE variant: ``'wan'`` (full Wan2.2), ``'mg_lightvae'``
            (pruning 0.5) or ``'mg_lightvae_v2'`` (pruning 0.75). LightVAE
            variants decode through the pruned student and encode through
            the full-VAE teacher, like the reference.
        dtype: Model dtype (e.g. ``mx.float32``, ``mx.bfloat16``).
        **kwargs: Reserved for future use.

    Returns:
        A :class:`Wan2_2_VAE` instance ready for encode/decode.
    """
    if vae_type not in _VAE_FILES:
        raise ValueError(f"Unsupported vae_type: {vae_type}")
    vae_file, pruning_rate = _VAE_FILES[vae_type]

    config = get_vae_config()
    config["vae_type"] = vae_type

    encoder_path = None
    if model_path is not None and os.path.isdir(model_path):
        vae_path = os.path.join(model_path, vae_file)
        encoder_path = os.path.join(model_path, "vae.safetensors")
    else:
        vae_path = model_path

    logger.info("Loading VAE (type=%s, dtype=%s, path=%s)", vae_type, dtype, vae_path)

    vae = Wan2_2_VAE(
        z_dim=config["z_dim"],
        c_dim=config["c_dim"],
        dec_dim=config["dec_dim"],
        dim_mult=config["dim_mult"],
        temperal_downsample=config["temperal_downsample"],
        dtype=dtype,
        vae_pth=vae_path,
        vae_type="wan2.2" if pruning_rate is None else "mg_lightvae",
        lightvae_pruning_rate=pruning_rate,
        lightvae_encoder_vae_pth=encoder_path,
    )

    return vae
