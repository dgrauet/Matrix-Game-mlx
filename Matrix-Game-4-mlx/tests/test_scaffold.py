from tests.conftest import import_upstream, rel_l2, psnr
import numpy as np


def test_package_imports():
    import matrix_game_mlx.inference  # noqa: F401


def test_upstream_importable():
    mod = import_upstream("matrix_game.inference.preprocessing")
    assert hasattr(mod, "crop_and_resize")


def test_metrics():
    a = np.ones((4, 4), np.float32)
    assert rel_l2(a, a) == 0.0
    assert psnr(a, a) == float("inf")
