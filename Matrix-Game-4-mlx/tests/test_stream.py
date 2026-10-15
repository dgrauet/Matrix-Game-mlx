import time

import mlx.core as mx
import pytest

from matrix_game_mlx.inference.stream import CastleStream, LATENT_SHAPE


def _bare_stream(index=0, frame=0, max_latents=8):
    s = CastleStream.__new__(CastleStream)
    s.closed, s.failed, s.index, s.frame, s.max_latents, s.shape = False, False, index, frame, max_latents, LATENT_SHAPE
    return s


def test_validate_cadence_range_budget():
    s = _bare_stream()
    with pytest.raises(ValueError, match="consecutive"):
        s._validate(mx.zeros((4, 3, 480, 864)), frame_start=0)          # first chunk must be 1 frame
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        s._validate(mx.full((1, 3, 480, 864), 2.0), frame_start=0)
    assert s._validate(mx.zeros((1, 3, 480, 864)), frame_start=0) == 1
    s = _bare_stream(index=1, frame=1)
    with pytest.raises(ValueError, match="consecutive"):
        s._validate(mx.zeros((4, 3, 480, 864)), frame_start=2)          # frame_start must follow
    assert s._validate(mx.zeros((4, 3, 480, 864)), frame_start=1) == 4
    with pytest.raises(ValueError, match="budget"):
        _bare_stream(index=8, frame=29)._validate(mx.zeros((4, 3, 480, 864)), frame_start=29)
    s.failed = True
    with pytest.raises(RuntimeError, match="closed or failed"):
        s._validate(mx.zeros((4, 3, 480, 864)), frame_start=1)


@pytest.mark.needs_weights
def test_model_components_stream_smoke():
    import numpy as np
    from matrix_game_mlx.inference.loader import ModelComponents

    components = ModelComponents("ancient-egypt")
    components.load()
    stream = components.stream()
    frame = 0
    for count in (1, 4):
        t0 = time.perf_counter()
        chunk = stream.push(mx.zeros((count, 3, 480, 864)), frame_start=frame)
        mx.eval(chunk.pixels)
        print(f"chunk {chunk.index}: {time.perf_counter() - t0:.1f}s, peak {mx.get_peak_memory() / 2**30:.1f} GiB")
        assert tuple(chunk.pixels.shape) == (count, 3, 480, 864)
        assert np.isfinite(np.array(chunk.pixels)).all()
        frame += count
    stream.close()
