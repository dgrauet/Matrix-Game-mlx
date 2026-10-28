"""Shared helpers for MLX vs upstream PyTorch tests."""
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from tests import racing_game

MLX_ROOT = str(Path(__file__).resolve().parents[1])
UPSTREAM_ROOT = str(Path(__file__).resolve().parents[2] / "Matrix-Game-4")
for p in (MLX_ROOT, UPSTREAM_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def import_upstream(name: str):
    """Import an upstream module (package `matrix_game`) from Matrix-Game-4/."""
    return importlib.import_module(name)


def to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    import mlx.core as mx
    if isinstance(x, mx.array):
        return np.array(x) if x.dtype == mx.float64 else np.array(x.astype(mx.float32))
    return np.asarray(x, dtype=np.float32)


def assert_close(a, b, atol=1e-4, rtol=1e-5, msg=""):
    np.testing.assert_allclose(to_np(a), to_np(b), atol=atol, rtol=rtol, err_msg=msg)


def rel_l2(a, b) -> float:
    a, b = to_np(a).astype(np.float64), to_np(b).astype(np.float64)
    denom = np.linalg.norm(b)
    return 0.0 if denom == 0 and np.linalg.norm(a) == 0 else float(np.linalg.norm(a - b) / denom)


def psnr(a, b, peak: float = 2.0) -> float:
    """PSNR for [-1, 1] images (peak-to-peak 2.0)."""
    mse = float(np.mean((to_np(a).astype(np.float64) - to_np(b).astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak ** 2 / mse)


def fixtures_dir() -> Path:
    return Path(os.environ.get("MG4_FIXTURES_DIR", Path.home() / ".cache/matrix-game-4-mlx/fixtures"))


def weights_dir() -> Optional[Path]:
    d = os.environ.get("MATRIX_GAME_4_MLX_WEIGHTS_DIR")
    return Path(d) if d and Path(d, "config.json").is_file() else None


@pytest.hookimpl(trylast=True)                  # after pytest's own fixture-based reordering
def pytest_collection_modifyitems(config, items):
    # Group the fixture-parity items of each module by their "scene" parameter (first-appearance order, stable
    # otherwise), so tests/parity/test_real_weights.py, which keeps one scene's DiT in memory at a time, loads each
    # scene once. Items without the needs_fixtures marker keep their order.
    def scene_key(item):
        scene = getattr(getattr(item, "callspec", None), "params", {}).get("scene")
        return item.path, scene if "needs_fixtures" in item.keywords else None

    modules, scenes = {}, {}
    for item in items:
        modules.setdefault(item.path, len(modules))
        scenes.setdefault(scene_key(item), len(scenes))
    items.sort(key=lambda item: (modules[item.path], scenes[scene_key(item)]))
    for item in items:
        if "needs_weights" in item.keywords and weights_dir() is None:
            item.add_marker(pytest.mark.skip(reason="MATRIX_GAME_4_MLX_WEIGHTS_DIR not set"))
        if "needs_fixtures" in item.keywords and not fixtures_dir().is_dir():
            item.add_marker(pytest.mark.skip(reason="no torch fixtures in MG4_FIXTURES_DIR"))
        if "needs_racing_game" in item.keywords and racing_game.racing_game_missing():
            item.add_marker(pytest.mark.skip(reason=racing_game.racing_game_missing()))
