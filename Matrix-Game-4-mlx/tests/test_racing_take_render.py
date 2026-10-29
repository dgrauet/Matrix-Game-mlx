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


def _render_and_finalize(tmp_path, *extra):
    """Render a copy of the canonical take with `extra` offline args, then run upstream finalize."""
    canonical = racing_game.default_take()
    if not canonical.is_dir():
        pytest.skip(racing_game.take_hint(canonical))
    take = tmp_path / "take"
    shutil.copytree(canonical, take)  # finalize writes demo.mp4 into the take; keep the canonical one pristine
    out = tmp_path / "rendered"
    offline.main(["--scene", "racing", "--take", str(take), "--preset", "realtime", "--output", str(out), *extra])
    report = json.loads((out / "render.json").read_text())
    session = json.loads((take / "session.json").read_text())
    final = subprocess.run([sys.executable, str(racing_game.finalize_tool()), str(take),
                            "--rendered", str(out / "rendered.mp4")], capture_output=True, text=True)
    assert final.returncode == 0, final.stdout + final.stderr
    return report, session, take, out


@pytest.mark.needs_weights
@pytest.mark.needs_racing_game
def test_realtime_render_of_take_finalizes(tmp_path):
    report, session, take, out = _render_and_finalize(tmp_path)
    assert report["status"] == "complete" and report["frames"] == session["frames"]
    assert report["scene"] == "racing" and report["scene_name"] == SCENES["racing"]["name"]
    assert report["sink_source"] != "generated_first_frame"
    # Upstream's optional HUD/input composition (README "Finalize and compose"), on the first 48 frames only
    composed = tmp_path / "composed.mp4"
    compose = subprocess.run([sys.executable, str(racing_game.compose_tool()), str(take), "--rendered",
                              str(out / "rendered.mp4"), "--output", str(composed), "--max-frames", "48"],
                             capture_output=True, text=True)
    assert compose.returncode == 0, compose.stdout + compose.stderr
    assert composed.is_file() and composed.stat().st_size > 0


def _caption_rows():
    return [json.loads(line) for line in CAPTIONS.read_text().splitlines()]


def test_captions_file_structure():
    """Chunking of the 720-frame take: frame 0 alone, groups of four, a final group of three (181 records)."""
    rows = _caption_rows()
    assert len(rows) == 181 and rows[-1]["frame_end"] == 719
    assert (rows[0]["frame_start"], rows[0]["frame_end"]) == (0, 0)
    assert all(r["frame_start"] == p["frame_end"] + 1 for p, r in zip(rows, rows[1:]))
    assert all(r["frame_end"] - r["frame_start"] + 1 == 4 for r in rows[1:-1])
    assert 1 <= rows[-1]["frame_end"] - rows[-1]["frame_start"] + 1 <= 4
    assert rows[0]["prompt"] == SCENES["racing"]["prompt"]
    assert len({r["prompt"] for r in rows}) >= 2


def test_captions_file_covers_take():
    manifest = racing_game.default_take() / "session.json"
    if not manifest.is_file():
        pytest.skip(racing_game.take_hint(manifest.parent))
    session = json.loads(manifest.read_text())
    assert _caption_rows()[-1]["frame_end"] == session["frames"] - 1


@pytest.mark.needs_weights
@pytest.mark.needs_racing_game
def test_dynamic_realtime_render_of_take_finalizes(tmp_path):
    report, session, _, _ = _render_and_finalize(tmp_path, "--prompt-mode", "dynamic", "--captions", str(CAPTIONS))
    assert report["status"] == "complete" and report["prompt_mode"] == "dynamic"
    assert report["frames"] == session["frames"]
