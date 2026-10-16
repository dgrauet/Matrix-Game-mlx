import numpy as np
import torch
import mlx.core as mx
from tests.conftest import import_upstream, assert_close

from matrix_game_mlx.inference.scheduler import FlowMatchScheduler
from matrix_game_mlx.inference.rollout import warp_steps

up_s = import_upstream("matrix_game.inference.scheduler")
up_r = import_upstream("matrix_game.inference.rollout")


def _pair():
    a = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    a.set_timesteps(1000, training=True)
    b = up_s.FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    b.set_timesteps(1000, training=True)
    return a, b


def test_sigmas_timesteps_match():
    """Bitwise: torch.linspace's float32 FMA halves are reproduced (np.linspace differed on ~220 entries)."""
    a, b = _pair()
    np.testing.assert_array_equal(np.array(a.sigmas), b.sigmas.numpy())
    np.testing.assert_array_equal(np.array(a.timesteps), b.timesteps.numpy())


def test_warp_steps_match():
    a, b = _pair()
    for steps in ([1000, 500], [1000], [1000, 750, 500, 250]):
        np.testing.assert_array_equal(np.array(warp_steps(a, steps)), up_r.warp_steps(b, steps).numpy())


def test_add_noise_match():
    a, b = _pair()
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((1, 48, 30, 54)).astype(np.float32)
    n = rng.standard_normal((1, 48, 30, 54)).astype(np.float32)
    t = float(up_r.warp_steps(b, [1000, 500])[1])
    out_a = a.add_noise(mx.array(x0), mx.array(n), mx.array([t]))
    out_b = b.add_noise(torch.from_numpy(x0), torch.from_numpy(n), torch.tensor([t]))
    np.testing.assert_array_equal(np.array(out_a), out_b.numpy())
