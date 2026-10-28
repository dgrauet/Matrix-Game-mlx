"""Scripted, reproducible take recorded from the upstream racing game."""
import json
import subprocess
import sys

import pytest
from matrix_game_mlx.scenes import SCENES

from tests import racing_game

pytestmark = pytest.mark.needs_racing_game


def _key_events(take):
    """Applied key events of a take as (frame, key, down), from inputs.jsonl."""
    events = (json.loads(line) for line in (take / "inputs.jsonl").read_text().splitlines())
    return [(e["next_frame"], e["key"], e["down"]) for e in events if e["event"] == "key"]


def test_script_presses_every_control_and_releases_them():
    script = json.loads(racing_game.TAKE_SCRIPT.read_text())
    frames = [e["frame"] for e in script]
    assert frames == sorted(frames)
    pressed = {e["key"] for e in script if e["down"]}
    assert {"W", "A", "D", "S", "Space"} <= pressed
    for key in pressed:
        downs = sum(e["down"] for e in script if e["key"] == key)
        assert downs == sum(not e["down"] for e in script if e["key"] == key)


def test_scripted_take_is_complete_and_finalizable(tmp_path):
    take = racing_game.record_scripted_take(tmp_path)
    session = json.loads((take / "session.json").read_text())
    assert session["status"] == "complete" and session["fps"] == 48
    assert session["frames"] >= 48 * 10
    assert session["prompt"] == SCENES["racing"]["prompt"]
    check = subprocess.run([sys.executable, str(racing_game.finalize_tool()), str(take), "--check-only"])
    assert check.returncode == 0


def test_scripted_take_is_reproducible(tmp_path):
    first = racing_game.record_scripted_take(tmp_path)
    second = racing_game.record_scripted_take(tmp_path)
    assert first != second                              # take-001 then take-002 in one directory
    a, b = (json.loads((t / "session.json").read_text()) for t in (first, second))
    assert a["frames"] == b["frames"]
    assert _key_events(first) == _key_events(second)    # applied input events from inputs.jsonl, per frame
