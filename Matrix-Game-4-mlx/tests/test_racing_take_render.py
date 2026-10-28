"""MLX render of a real recorded racing take, then upstream finalize."""
import json
import shutil
import subprocess
import sys

import pytest
from matrix_game_mlx.inference import offline
from matrix_game_mlx.scenes import SCENES

from tests import racing_game

pytestmark = [pytest.mark.needs_weights, pytest.mark.needs_racing_game]


def test_realtime_render_of_take_finalizes(tmp_path):
    canonical = racing_game.default_take()
    if not canonical.is_dir():
        pytest.skip(f"no recorded take at {canonical}: record one first (Task 2 Step 5)")
    take = tmp_path / "take"
    shutil.copytree(canonical, take)  # finalize writes demo.mp4 into the take; keep the canonical one pristine
    out = tmp_path / "rendered"
    offline.main(["--scene", "racing", "--take", str(take), "--preset", "realtime", "--output", str(out)])
    report = json.loads((out / "render.json").read_text())
    session = json.loads((take / "session.json").read_text())
    assert report["status"] == "complete" and report["frames"] == session["frames"]
    assert report["scene"] == "racing" and report["scene_name"] == SCENES["racing"]["name"]
    assert report["sink_source"] != "generated_first_frame"
    final = subprocess.run([sys.executable, str(racing_game.finalize_tool()), str(take),
                            "--rendered", str(out / "rendered.mp4")])
    assert final.returncode == 0
