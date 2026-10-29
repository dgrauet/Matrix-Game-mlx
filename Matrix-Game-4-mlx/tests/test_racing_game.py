"""Headless smoke of the upstream racing game (C++/SDL2/OpenGL) built on macOS."""
import pytest

from tests import racing_game


def test_skip_reason_names_download_and_build_commands(monkeypatch, tmp_path):
    monkeypatch.setenv("MG4_RESOURCES", str(tmp_path))
    monkeypatch.setattr(racing_game, "game_binary", lambda: tmp_path / "missing-binary")
    reason = racing_game.racing_game_missing()
    assert "matrix-game-mlx download --manifest releases/game-assets.json" in reason
    assert f"--scene racing --resources {tmp_path}" in reason
    assert f'bash scripts/build_racing.sh -DRACING_ASSET_ROOT="{tmp_path}/racing"' in reason
    assert "..." not in reason and "$MG4_RESOURCES" not in reason


def test_take_hint_points_at_readme(tmp_path):
    hint = racing_game.take_hint(tmp_path / "take-001")
    assert "Racing tests" in hint and "MG4_RACING_TAKE" in hint and str(tmp_path / "take-001") in hint


@pytest.mark.needs_racing_game
@pytest.mark.parametrize("mode", [[], ["--race", "--route", "downtown"], ["--race", "--route", "avenue"],
                                  ["--race", "--route", "skyway"], ["--pursuit"]])
def test_mode_runs_headless(mode):
    result = racing_game.run_game(mode + ["--duration", "3", "--no-audio"], timeout=120)
    assert result.returncode == 0, result.stderr
