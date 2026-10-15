import pytest
import mlx.core as mx
from tests.conftest import weights_dir

# MLX vs upstream umT5 (bf16, emulated fp32-accumulate GEMMs) on the ancient-egypt prompt: 1.5x the largest rel L2
# between that reference and its torch variants (float64 accumulation 3.21e-2, fp32 model 1.75e-2), rounded up.
# Measured: 2.80e-2.
T5_REL_L2 = 5e-2


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


def _torch_umt5_prompt_embeds(t5_path: str, tokenizer_path: str, prompt: str):
    """Upstream WanTextEncoder.forward on the upstream umT5 in bf16, as upstream's loader runs it
    (`WanTextEncoder(...).to(device, torch.bfloat16)`). The weights come from the shared MG3 safetensors (the
    bf16 values of `models_t5_umt5-xxl-enc-bf16.pth`, keys prefixed `t5_encoder.`), loaded with `assign=True`
    onto a meta-built model so only one bf16 copy is resident. CPU bf16 GEMMs are emulated with fp32
    accumulation (CUDA semantics), as in the parity fixtures."""
    import torch
    from safetensors.torch import load_file
    from matrix_game.inference.wan.modules.t5 import umt5_xxl
    from matrix_game.inference.wan.modules.tokenizers import HuggingfaceTokenizer
    from matrix_game.inference.wrappers import WanTextEncoder as TorchTextEncoder
    from tests.parity.gen_fixtures import cpu_bf16_emulation

    model = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device="meta")
    state = {k.replace("t5_encoder.", "", 1): v for k, v in load_file(t5_path).items()}
    model.load_state_dict(state, assign=True)
    del state
    enc = TorchTextEncoder.__new__(TorchTextEncoder)
    torch.nn.Module.__init__(enc)
    enc.text_encoder = model.eval().requires_grad_(False)
    enc.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean="whitespace")
    with torch.no_grad(), cpu_bf16_emulation(torch):
        out = enc(text_prompts=[prompt])["prompt_embeds"].float().numpy()
    del enc, model
    return out


@pytest.mark.needs_weights
def test_prompt_embeds_match_upstream_umt5():
    """MLX umT5 vs upstream's umT5 in bf16 on the ancient-egypt scene prompt; gate from the torch-variant floor."""
    import gc
    import numpy as np
    from matrix_game_mlx.inference.loader import resolve_shared_files
    from matrix_game_mlx.inference.wrappers import WanTextEncoder
    from matrix_game_mlx.scenes import SCENES
    from tests.conftest import rel_l2, to_np
    shared = resolve_shared_files(weights_dir())
    t5_path, tok_path = str(shared["t5_encoder.safetensors"]), str(shared["google/umt5-xxl"])
    prompt = SCENES["ancient-egypt"]["prompt"]
    enc = WanTextEncoder(t5_path, tok_path)
    ours = to_np(enc([prompt])["prompt_embeds"].astype(mx.float32))
    del enc
    gc.collect()
    mx.clear_cache()
    ref = _torch_umt5_prompt_embeds(t5_path, tok_path, prompt)
    n = int((np.abs(ref[0]).max(axis=-1) > 0).sum())
    diff = np.abs(ours - ref)
    print(f"umT5 parity: tokens={n} rel_l2={rel_l2(ours, ref):.3e} max_abs={diff.max():.3e} "
          f"mean_abs={diff.mean():.3e} ref_absmax={np.abs(ref).max():.3e} exact={float((diff == 0).mean()):.4f}")
    assert np.all(ours[0, n:] == 0) and np.all(ref[0, n:] == 0)
    assert rel_l2(ours, ref) < T5_REL_L2


def test_t5_attention_bf16_matches_upstream():
    """bf16 T5Attention keeps bf16 scores (upstream `x.new_zeros` bias) and matches upstream's bf16 forward."""
    import numpy as np
    import torch
    from matrix_game.inference.wan.modules.t5 import T5Attention as TorchAttention
    from matrix_game_mlx.inference.wan.modules.t5 import T5Attention
    from tests.conftest import rel_l2, to_np
    torch.manual_seed(0)
    ref = TorchAttention(64, 64, 4).eval().to(torch.bfloat16)
    ours = T5Attention(64, 64, 4)
    ours.load_weights([(f"{n}.weight", mx.array(getattr(ref, n).weight.detach().float().numpy()).astype(mx.bfloat16))
                       for n in ("q", "k", "v", "o")])
    x = torch.randn(1, 16, 64).to(torch.bfloat16)
    pos_bias = torch.randn(1, 4, 16, 16).to(torch.bfloat16)
    mask = torch.ones(1, 16, dtype=torch.long)
    mask[:, 12:] = 0
    with torch.no_grad():
        want = ref(x, mask=mask, pos_bias=pos_bias).float().numpy()
    got = ours(mx.array(x.float().numpy()).astype(mx.bfloat16), mask=mx.array(mask.numpy()),
               pos_bias=mx.array(pos_bias.float().numpy()).astype(mx.bfloat16))
    assert got.dtype == mx.bfloat16
    assert rel_l2(to_np(got.astype(mx.float32)), want) < 1e-2, rel_l2(to_np(got.astype(mx.float32)), want)
    assert np.isfinite(to_np(got.astype(mx.float32))).all()
