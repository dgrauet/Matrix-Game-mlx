import pytest
import mlx.core as mx
from tests.conftest import weights_dir


def test_copied_t5_modules_import():
    from matrix_game_mlx.inference.wan.modules.t5 import T5EncoderModel, umt5_xxl  # noqa: F401
    from matrix_game_mlx.inference.wan.modules.tokenizers import HuggingfaceTokenizer  # noqa: F401


@pytest.mark.needs_weights
def test_prompt_embeds_shape_and_padding():
    from matrix_game_mlx.inference.loader import resolve_shared_files
    from matrix_game_mlx.inference.wrappers import WanTextEncoder
    shared = resolve_shared_files(weights_dir())
    enc = WanTextEncoder(str(shared["t5_encoder.safetensors"]), str(shared["google/umt5-xxl"]))
    out = enc(["a short prompt"])["prompt_embeds"]
    assert out.shape == (1, 512, 4096) and out.dtype == mx.bfloat16
    assert float(mx.abs(out[0, 20:]).max()) == 0.0
