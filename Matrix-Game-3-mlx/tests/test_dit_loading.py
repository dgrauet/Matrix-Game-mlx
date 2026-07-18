"""Validation for DiT weight loading dtype handling.

The official distilled checkpoint is distributed in float32, but the PyTorch
reference casts the model to bfloat16 at load time
(``Matrix-Game-3/pipeline/inference_pipeline.py:245``). The MLX port must
mirror that cast: loading float32 weights must yield bfloat16 parameters.
"""

import os
import tempfile

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from wan.modules.model import WanModel


def _tiny_model() -> WanModel:
    return WanModel(
        model_type='t2v', patch_size=(1, 2, 2), in_dim=8, dim=96,
        ffn_dim=192, freq_dim=32, text_dim=64, out_dim=8, num_heads=4,
        num_layers=2, use_memory=True, sigma_theta=0.8, action_config={},
    )


@pytest.fixture()
def fp32_checkpoint(tmp_path):
    """A safetensors checkpoint with float32 weights and a dit prefix."""
    model = _tiny_model()
    weights = {
        f"dit_distilled.{k}": v.astype(mx.float32)
        for k, v in tree_flatten(model.parameters())
    }
    path = os.path.join(tmp_path, "dit_distilled.safetensors")
    mx.save_safetensors(path, weights)
    return path


class TestLoadDitWeights:
    def test_fp32_checkpoint_loads_as_bf16(self, fp32_checkpoint):
        from pipeline.inference_pipeline import load_dit_weights

        model = _tiny_model()
        load_dit_weights(
            model, fp32_checkpoint, "dit_distilled.", mx.bfloat16
        )

        for name, param in tree_flatten(model.parameters()):
            assert param.dtype == mx.bfloat16, (
                f"{name} is {param.dtype}, expected bfloat16"
            )
