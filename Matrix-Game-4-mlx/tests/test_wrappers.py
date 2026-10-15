import numpy as np
import pytest
import torch
import mlx.core as mx
from tests.conftest import import_upstream, assert_close, weights_dir
from tests.test_causal_model import CFG, build_pair

from matrix_game_mlx.inference.wrappers import WanDiffusionWrapper, init_kv_cache, init_crossattn_cache, frame_seq_length_of

up_w = import_upstream("matrix_game.inference.wrappers")


def test_with_condition_layout():
    model, _ = build_pair()
    w = WanDiffusionWrapper(model, timestep_shift=5.0)
    assert w.control_channels == 8
    x = mx.ones((1, 1, 4, 2, 4)); ctrl = mx.full((1, 1, w.control_channels, 2, 4), 2.0)
    out = w._with_condition(x, ctrl)[0]            # [F, h, w, in_dim]
    assert out.shape == (1, 2, 4, 12)
    assert float(out[..., :4].mean()) == 1.0 and float(out[..., 4:12].mean()) == 2.0
    out = w._with_condition(x, None)[0]
    assert float(out[..., :4].mean()) == 1.0 and float(out[..., 4:].sum()) == 0.0


def test_flow_to_x0_matches_upstream():
    model, _ = build_pair()
    w = WanDiffusionWrapper(model, timestep_shift=5.0)
    sched = up_w.FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(1000, training=True)
    rng = np.random.default_rng(0)
    fp, xt = (rng.standard_normal((1, 48, 30, 54)).astype(np.float32) for _ in range(2))
    for t in (float(sched.timesteps[0]), float(sched.timesteps[500]), 0.0):
        ref_wrapper = type("W", (), {"scheduler": sched})()
        ref = up_w.WanDiffusionWrapper._convert_flow_pred_to_x0(ref_wrapper, torch.from_numpy(fp), torch.from_numpy(xt),
                                                                torch.tensor([t]))
        out = w._convert_flow_pred_to_x0(mx.array(fp), mx.array(xt), mx.array([t]))
        np.testing.assert_array_equal(np.array(out), ref.numpy(), err_msg=f"t={t}")        # float64 as upstream
        bf = lambda a: torch.from_numpy(a).bfloat16()
        ref = up_w.WanDiffusionWrapper._convert_flow_pred_to_x0(ref_wrapper, bf(fp), bf(xt), torch.tensor([t]))
        out = w._convert_flow_pred_to_x0(mx.array(fp).astype(mx.bfloat16), mx.array(xt).astype(mx.bfloat16), mx.array([t]))
        np.testing.assert_array_equal(np.array(out.astype(mx.float32)), ref.float().numpy(), err_msg=f"bf16 t={t}")


def test_caches_shapes():
    model, _ = build_pair()
    kv = init_kv_cache(model, 1, 21 * 2, mx.bfloat16)
    assert len(kv) == 2 and kv[0]["k"].shape == (1, 42, 2, 64) and kv[0]["local_end_index"] == 0
    assert init_crossattn_cache(model, 1, mx.bfloat16)[0] == {"is_init": False}
    assert frame_seq_length_of(model, 30, 54) == 405


@pytest.mark.needs_weights
def test_build_dit_real_pack():
    from matrix_game_mlx.inference.loading import build_dit
    w = build_dit(weights_dir(), "dit_ancient_egypt")
    m = w.model
    assert (m.in_dim, m.dim, m.num_layers, m.num_heads) == (148, 3072, 30, 24)
    assert m.patch_embedding.weight.shape == (3072, 592) and m.patch_embedding.weight.dtype == mx.bfloat16
