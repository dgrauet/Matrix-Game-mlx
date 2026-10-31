import hashlib
from pathlib import Path

import pytest

from matrix_game_mlx.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,flag", [("serve", "--reference-only"), ("render", "--take"), ("download", "--manifest"),
                                       ("verify", "--manifest")])
def test_subcommand_help(name, flag, capsys):
    with pytest.raises(SystemExit) as stopped:
        main([name, "--help"])
    assert stopped.value.code == 0
    assert flag in capsys.readouterr().out


def test_serve_dispatches_to_server_app(monkeypatch):
    received = []
    monkeypatch.setattr("matrix_game_mlx.server.app.main", received.append)
    main(["serve", "--scene", "castle", "--fps", "24"])
    assert received == [["--scene", "castle", "--fps", "24"]]


def test_console_script_declared():
    assert 'matrix-game-mlx = "matrix_game_mlx.cli:main"' in (ROOT / "pyproject.toml").read_text()


def test_game_assets_manifest_is_upstream():
    mine = ROOT / "releases" / "game-assets.json"
    upstream = ROOT.parent / "Matrix-Game-4" / "releases" / "game-assets.json"
    assert hashlib.sha256(mine.read_bytes()).hexdigest() == hashlib.sha256(upstream.read_bytes()).hexdigest()
