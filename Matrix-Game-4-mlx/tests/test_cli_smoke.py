import json
from pathlib import Path

import pytest

CAPTIONS = Path(__file__).parent / "data" / "racing_captions.jsonl"


@pytest.mark.needs_weights
def test_dynamic_render_smoke(tmp_path):
    """Realtime + dynamic prompts end to end through the CLI (T5 stays resident; must fit 32 GB)."""
    import imageio_ffmpeg
    import numpy as np
    from matrix_game_mlx import cli

    src = tmp_path / "in.mp4"
    w = imageio_ffmpeg.write_frames(str(src), (864, 480), fps=24, macro_block_size=1)
    w.send(None)
    for i in range(9):
        w.send(np.full((480, 864, 3), 20 * i, np.uint8).tobytes())
    w.close()
    out = tmp_path / "out"
    cli.main(["render", "--scene", "racing", "--preset", "realtime", "--prompt-mode", "dynamic",
              "--captions", str(CAPTIONS), "--input", str(src), "--output", str(out)])
    rep = json.loads((out / "render.json").read_text())
    assert rep["prompt_mode"] == "dynamic" and rep["status"] == "complete"
    assert rep["frames"] == 9
    assert rep["peak_memory_gb"] < 30
    frames = sum(1 for f in imageio_ffmpeg.read_frames(str(out / "rendered.mp4")) if not isinstance(f, dict))
    assert frames == 9
    print("peak_memory_gb", rep["peak_memory_gb"])
