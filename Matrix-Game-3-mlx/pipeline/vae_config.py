"""VAE configuration and loading utilities for MLX.

Simplified version of the PyTorch reference — no async VAE worker,
no torch.compile, no LightVAE pruning (weights pre-converted by mlx-forge).
"""

import logging
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


def load_vae(
    model_path: Optional[str] = None,
    vae_type: str = "wan",
    dtype: mx.Dtype = mx.float32,
    **kwargs: Any,
) -> Wan2_2_VAE:
    """Load a VAE model from mlx-forge converted weights.

    Args:
        model_path: Path to VAE safetensors weights file or directory.
        vae_type: VAE variant (``'wan'`` for standard Wan2.2).
        dtype: Model dtype (e.g. ``mx.float32``, ``mx.float16``).
        **kwargs: Reserved for future use.

    Returns:
        A :class:`Wan2_2_VAE` instance ready for encode/decode.
    """
    config = get_vae_config()
    config["vae_type"] = vae_type

    logger.info("Loading VAE (type=%s, dtype=%s, path=%s)", vae_type, dtype, model_path)

    vae = Wan2_2_VAE(
        z_dim=config["z_dim"],
        c_dim=config["c_dim"],
        dec_dim=config["dec_dim"],
        dim_mult=config["dim_mult"],
        temperal_downsample=config["temperal_downsample"],
        dtype=dtype,
        vae_pth=model_path,
    )

    return vae
