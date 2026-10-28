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


# Scripted take. The script's frames count from game start (fixed 1/fps step offline); the 3 s countdown
# ends and recording auto-starts at frame 144, and the script's F9 press at frame 864 stops it, so the
# take holds exactly TAKE_DURATION_S * 48 frames regardless of wall-clock speed.
TAKE_SCRIPT = Path(__file__).parent / "data/racing/take_input_script.json"
TAKE_ROUTE = "downtown"
TAKE_DURATION_S = 15.0
TAKE_ENV = "MG4_RACING_TAKE"
_TAKE_GAME_TIMEOUT_S = 120.0
# --duration is a wall-clock safety net: countdown + take + margin for the slower-than-realtime startup.
_TAKE_GAME_DURATION_S = 3 + TAKE_DURATION_S + 30


def finalize_tool() -> Path:
    return UPSTREAM / "game/racing/tools/finalize-offline-demo.py"


def compose_tool() -> Path:
    return UPSTREAM / "game/racing/tools/compose-demo.py"


def default_take() -> Path:
    return Path(os.environ.get(TAKE_ENV) or resources_dir() / "racing/captures/take-001")


def record_scripted_take(directory: Path) -> Path:
    """Record TAKE_SCRIPT into `directory` and return the new take-NNN directory it created."""
    before = set(directory.glob("take-*")) if directory.is_dir() else set()
    result = run_game(["--race", "--route", TAKE_ROUTE, "--fps", "48", "--physics-hz", "480",
                       "--offline-record", str(directory), "--offline-auto-record",
                       "--offline-input-script", str(TAKE_SCRIPT), "--duration", str(_TAKE_GAME_DURATION_S)],
                      timeout=_TAKE_GAME_TIMEOUT_S)
    assert result.returncode == 0, result.stderr
    new = set(directory.glob("take-*")) - before
    assert len(new) == 1, f"expected one new take in {directory}, got {sorted(new)}"
    return new.pop()
