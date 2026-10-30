"""Real native game processes (shooter and racing reference servers) driven through the IPC wrappers."""
import threading

import numpy as np
import pytest
from PIL import Image

from matrix_game_mlx.inference.preprocessing import crop_and_resize, frames_to_tensor
from matrix_game_mlx.server.game import CastleGame
from matrix_game_mlx.server.racing import RacingGame
from matrix_game_mlx.server.reference_input import PreparedReference, ReferencePreprocessor
from tests.racing_game import UPSTREAM, resources_dir

GAME_ROOT = UPSTREAM / "game"


@pytest.mark.needs_castle_game
@pytest.mark.parametrize("scene", ["castle", "ancient-egypt"])
def test_castle_reference_server_handshake(scene, tmp_path):
    game = CastleGame(GAME_ROOT, fps=40, scene=scene, asset_root=resources_dir() / "game",
                      log_path=tmp_path / "game.log")
    try:
        assert game.hello["audio_schema"] == 1
        assert game.hello["mouse_input_schema"] == 2
        assert game.hello["renderer"]
        cancel = threading.Event()
        first, states = game.next(cancel)
        assert len(first) == 1 and [s["frame"] for s in states] == [0]
        game.send(dict(type="controls", sequence=1, forward=1.0, right=0.0, look_x=0.0, look_y=0.0))
        second, states = game.next(cancel)
        assert len(second) == 4 and [s["frame"] for s in states] == [1, 2, 3, 4]
    finally:
        game.close()
    assert game.process.poll() is not None


@pytest.mark.needs_racing_server
def test_racing_reference_server_captions(tmp_path):
    game = RacingGame(GAME_ROOT, fps=24, asset_root=resources_dir() / "racing", options={},
                      log_path=tmp_path / "race.log", prepare=ReferencePreprocessor(model=True))
    try:
        cancel = threading.Event()
        for count in (1, 4, 4):
            reference, states = game.next(cancel)
            assert isinstance(reference, PreparedReference)
            assert tuple(reference.pixels.shape) == (count, 3, 480, 864)
            assert len(states) == count
            prompt = states[-1]["chunk_prompt"]
            assert isinstance(prompt, str) and prompt
    finally:
        game.close()
    assert game.process.poll() is not None


@pytest.mark.needs_racing_server
def test_stream_and_serial_pipelines_match(tmp_path):
    """Both preprocessing pipelines see the same raw RGBA frames (one game run), and must agree exactly."""
    game = RacingGame(GAME_ROOT, fps=24, asset_root=resources_dir() / "racing", options={},
                      log_path=tmp_path / "race.log")
    try:
        cancel = threading.Event()
        chunks = [game.next(cancel)[0] for _ in range(3)]
    finally:
        game.close()
    assert [len(c) for c in chunks] == [1, 4, 4]
    for rgba in chunks:
        stream = ReferencePreprocessor(model=True)
        stream.begin()
        streamed = stream.finish([stream.frame(frame) for frame in rgba]).pixels
        serial = frames_to_tensor([crop_and_resize(Image.fromarray(frame).convert("RGB"), 480, 864)
                                   for frame in rgba])
        assert np.array_equal(np.array(streamed), np.array(serial))
