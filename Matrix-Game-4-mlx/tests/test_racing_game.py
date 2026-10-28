"""Headless smoke of the upstream racing game (C++/SDL2/OpenGL) built on macOS."""
import pytest

from tests import racing_game


def test_skip_reason_names_build_command(monkeypatch, tmp_path):
    monkeypatch.setenv("MG4_RESOURCES", str(tmp_path))
    reason = racing_game.racing_game_missing()
    assert "matrix-game-mlx download --scene racing" in reason


@pytest.mark.needs_racing_game
@pytest.mark.parametrize("mode", [[], ["--race", "--route", "downtown"], ["--race", "--route", "avenue"],
                                  ["--race", "--route", "skyway"], ["--pursuit"]])
def test_mode_runs_headless(mode):
    result = racing_game.run_game(mode + ["--duration", "3", "--no-audio"], timeout=120)
    assert result.returncode == 0, result.stderr
