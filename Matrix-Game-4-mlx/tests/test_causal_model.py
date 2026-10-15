import numpy as np
import torch
import mlx.core as mx
from tests.conftest import import_upstream, assert_close

from matrix_game_mlx.inference.wan.modules.causal_model import CausalWanModel

up_c = import_upstream("matrix_game.inference.wan.modules.causal_model")

CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), text_len=8, in_dim=12, dim=128, ffn_dim=256, freq_dim=256,
           text_dim=32, out_dim=4, num_heads=2, num_layers=2, local_attn_size=4, sink_size=1, qk_norm=True,
           cross_attn_norm=True, eps=1e-6)
RENAMES = {"ffn.0.": "ffn_linear1.", "ffn.2.": "ffn_linear2.", "text_embedding.0.": "text_embedding_linear1.",
           "text_embedding.2.": "text_embedding_linear2.", "time_embedding.0.": "time_embedding_linear1.",
           "time_embedding.2.": "time_embedding_linear2.", "time_projection.1.": "time_projection_linear1."}


def mlx_weights_from_torch(sd):
    """Same transform as the mlx-forge matrix-game-4.0 recipe: renames + patch_embedding conv -> linear."""
    out = {}
    for k, v in sd.items():
        for old, new in RENAMES.items():
            k = k.replace(old, new)
        v = v.detach().float().numpy()
        if k == "patch_embedding.weight":
            v = v.reshape(v.shape[0], -1)
        out[k] = mx.array(v)
    return out


def build_pair():
    torch.manual_seed(0)
    ref = up_c.CausalWanModel(**CFG).eval()
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.05)
    model = CausalWanModel(**CFG)
    model.load_weights(list(mlx_weights_from_torch(ref.state_dict()).items()), strict=True)
    return model, ref


def test_rollout_25_chunks_fixed_sink():
    model, ref = build_pair()
    fsl, L = 2, 25
    kv_b = [{"k": torch.zeros(1, 4 * fsl, 2, 64), "v": torch.zeros(1, 4 * fsl, 2, 64),
             "global_end_index": torch.tensor([0]), "local_end_index": torch.tensor([0]),
             "k_sink_raw": torch.zeros(1, fsl, 2, 64)} for _ in range(2)]
    kv_a = [{"k": mx.zeros((1, 4 * fsl, 2, 64)), "v": mx.zeros((1, 4 * fsl, 2, 64)), "global_end_index": 0,
             "local_end_index": 0, "k_sink_raw": mx.zeros((1, fsl, 2, 64))} for _ in range(2)]
    xc_b = [{"k": None, "v": None, "is_init": False} for _ in range(2)]
    xc_a = [{"is_init": False} for _ in range(2)]
    ref.set_rope_max_len(8192); model.set_rope_max_len(8192)
    rng = np.random.default_rng(0)
    ctx = rng.standard_normal((1, 8, 32)).astype(np.float32)
    for i in range(L):
        x = rng.standard_normal((12, 1, 2, 4)).astype(np.float32)          # torch [C, F, H, W]
        for t in (937.5, 0.0):
            with torch.no_grad():
                out_b = ref(torch.from_numpy(x)[None], t=torch.tensor([[t]]), context=torch.from_numpy(ctx),
                            seq_len=fsl, kv_cache=kv_b, crossattn_cache=xc_b, current_start=i * fsl,
                            sink_rope_mode="fixed")
            out_a = model([mx.array(x.transpose(1, 2, 3, 0))], t=mx.array([[t]]), context=mx.array(ctx),
                          seq_len=fsl, kv_cache=kv_a, crossattn_cache=xc_a, current_start=i * fsl,
                          sink_rope_mode="fixed")
            assert_close(out_a, out_b.permute(0, 2, 3, 4, 1), atol=5e-4, msg=f"chunk {i} t={t}")
