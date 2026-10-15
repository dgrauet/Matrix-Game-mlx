import numpy as np
import pytest
from PIL import Image
from tests.conftest import import_upstream, assert_close, to_np

from matrix_game_mlx.inference.preprocessing import crop_and_resize, frames_to_tensor, tensor_to_uint8
from matrix_game_mlx.scenes import SCENES, scene_sink_policy

up = import_upstream("matrix_game.inference.preprocessing")


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (720, 1280), (864, 480)])
def test_crop_and_resize_matches_upstream(size):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    a = np.asarray(crop_and_resize(img, 480, 864))
    b = np.asarray(up.crop_and_resize(img, 480, 864))
    assert a.shape == (480, 864, 3)
    np.testing.assert_array_equal(a, b)


def test_frames_to_tensor_and_back():
    rng = np.random.default_rng(1)
    frames = [Image.fromarray(rng.integers(0, 256, (480, 864, 3), dtype=np.uint8)) for _ in range(4)]
    x = frames_to_tensor(frames)
    assert_close(x, up.frames_to_tensor(frames), atol=0, rtol=0)
    np.testing.assert_array_equal(tensor_to_uint8(x), up.tensor_to_uint8(up.frames_to_tensor(frames)))


def test_scenes():
    assert set(SCENES) == {"castle", "ancient-egypt", "racing"}
    assert scene_sink_policy("racing") == ("first_frame", "absolute")
    assert scene_sink_policy("ancient-egypt") == ("kv_only", "fixed")
