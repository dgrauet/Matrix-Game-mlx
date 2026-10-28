"""Locate, build-check and run the upstream racing game (Matrix-Game-4/game/racing) for tests."""
import os
import subprocess
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "Matrix-Game-4"
RESOURCES_ENV = "MG4_RESOURCES"
BUILD_COMMAND = 'cd Matrix-Game-4 && bash scripts/build_racing.sh -DRACING_ASSET_ROOT="$MG4_RESOURCES/racing"'


def resources_dir() -> Path:
    return Path(os.environ.get(RESOURCES_ENV) or UPSTREAM / "resources")


def racing_data_dir() -> Path:
    return resources_dir() / "racing"


def game_binary() -> Path:
    return UPSTREAM / "game/racing/bin/CitySampleBBoxGame"


def racing_game_missing() -> Optional[str]:
    """None when the game binary and its assets are present, else what to run to get them."""
    todo = []
    if not (racing_data_dir() / "assets/bbox-assets.bin").is_file():
        todo.append("matrix-game-mlx download --scene racing --manifest releases/game-assets.json "
                    "--resources $MG4_RESOURCES ...")
    if not game_binary().is_file():
        todo.append(BUILD_COMMAND)
    return "racing game not ready, run: " + " ; ".join(todo) if todo else None


def run_game(args: list, timeout: float) -> subprocess.CompletedProcess:
    """Run the game hidden (no window) with the racing data dir; raises TimeoutExpired past `timeout` seconds."""
    command = [str(game_binary()), "--data-dir", str(racing_data_dir()), "--hidden", *args]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
