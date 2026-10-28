"""MLX render of a real recorded racing take, then upstream finalize."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from matrix_game_mlx.inference import offline
from matrix_game_mlx.scenes import SCENES

from tests import racing_game

CAPTIONS = Path(__file__).parent / "data" / "racing" / "take_captions.jsonl"

pytestmark = [pytest.mark.needs_weights, pytest.mark.needs_racing_game]


def _render_and_finalize(tmp_path, *extra):
    """Render a copy of the canonical take with `extra` offline args, then run upstream finalize."""
    canonical = racing_game.default_take()
    if not canonical.is_dir():
        pytest.skip(f"no recorded take at {canonical}: record one first (Task 2 Step 5)")
    take = tmp_path / "take"
    shutil.copytree(canonical, take)  # finalize writes demo.mp4 into the take; keep the canonical one pristine
    out = tmp_path / "rendered"
    offline.main(["--scene", "racing", "--take", str(take), "--preset", "realtime", "--output", str(out), *extra])
    report = json.loads((out / "render.json").read_text())
    session = json.loads((take / "session.json").read_text())
    final = subprocess.run([sys.executable, str(racing_game.finalize_tool()), str(take),
                            "--rendered", str(out / "rendered.mp4")], capture_output=True, text=True)
    assert final.returncode == 0, final.stdout + final.stderr
    return report, session


def test_realtime_render_of_take_finalizes(tmp_path):
    report, session = _render_and_finalize(tmp_path)
    assert report["status"] == "complete" and report["frames"] == session["frames"]
    assert report["scene"] == "racing" and report["scene_name"] == SCENES["racing"]["name"]
    assert report["sink_source"] != "generated_first_frame"


def test_captions_file_matches_take():
    manifest = racing_game.default_take() / "session.json"
    if not manifest.is_file():
        pytest.skip(f"no recorded take at {manifest.parent}")
    session = json.loads(manifest.read_text())
    rows = [json.loads(line) for line in CAPTIONS.read_text().splitlines()]
    assert rows[0]["frame_start"] == 0 and rows[0]["frame_end"] == 0
    assert all(r["frame_start"] == p["frame_end"] + 1 for p, r in zip(rows, rows[1:]))
    assert rows[-1]["frame_end"] == session["frames"] - 1
    assert rows[0]["prompt"] == SCENES["racing"]["prompt"]
    assert len({r["prompt"] for r in rows}) >= 2


def test_dynamic_realtime_render_of_take_finalizes(tmp_path):
    report, session = _render_and_finalize(tmp_path, "--prompt-mode", "dynamic", "--captions", str(CAPTIONS))
    assert report["status"] == "complete" and report["prompt_mode"] == "dynamic"
    assert report["frames"] == session["frames"]
