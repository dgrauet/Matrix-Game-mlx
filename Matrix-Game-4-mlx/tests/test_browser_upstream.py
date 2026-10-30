"""Upstream browser tests (Matrix-Game-4/tests/browser/*.cjs), unmodified, against the port's serve and web client.

The scripts are configured only through their environment knobs (TEST_URL, TEST_WEBKIT, REFERENCE_ONLY, ...), with
short durations. The pure-node ones (HUD, audio scheduling) and the synthetic Web Audio one load the web files from
the upstream tree, which the port's `matrix_game_mlx/web/` copies byte for byte (tests/test_serve_app.py).

Two settings, both through the scripts' own knobs (docs/testing.md), make them run on macOS:
- CHROME_PATH is Playwright's full Chromium. Its default headless shell rejects the canvas pointer lock
  (`WrongDocumentError: The root document of this element is not valid for pointer lock`), so every input check
  times out; the full binary runs in Chromium's new headless mode, which grants it.
- The WebKit run sets DISPLAY_ONLY=1: Playwright's WebKit never grants pointer lock here, headless or headed
  (`Pointer lock requires the window to have focus`), the case docs/testing.md prescribes DISPLAY_ONLY for. It still
  checks WHEP video, frame-matched HUD/metadata, Stop and the layouts; its report records
  `displayOnly`.
"""
import functools
import os
import subprocess

import pytest

from tests.racing_game import resources_dir
from tests.serve_process import PORT_ROOT, playwright_module, serve

BROWSER_TESTS = "../Matrix-Game-4/tests/browser"      # relative to PORT_ROOT


@functools.lru_cache(maxsize=None)
def chrome_path() -> str:
    """CHROME_PATH, else the full Chromium Playwright installed (not its headless shell; see the module doc)."""
    if os.environ.get("CHROME_PATH"):
        return os.environ["CHROME_PATH"]
    script = "console.log(require(process.env.PLAYWRIGHT_MODULE).chromium.executablePath())"
    return subprocess.run(["node", "-e", script], env=dict(os.environ, PLAYWRIGHT_MODULE=playwright_module()),
                          capture_output=True, text=True, check=True).stdout.strip()


def run_node(script: str, argument, passed: str, timeout: float, browser: bool = True, **env):
    """Run one upstream script from the port root; it must exit 0 and print its pass marker."""
    environment = dict(os.environ, **env)
    if browser:
        environment.update(PLAYWRIGHT_MODULE=playwright_module(), CHROME_PATH=chrome_path())
    result = subprocess.run(["node", f"{BROWSER_TESTS}/{script}", *([str(argument)] if argument else [])],
                            cwd=PORT_ROOT, env=environment, capture_output=True, text=True, timeout=timeout)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{script} exited {result.returncode}:\n{output[-6000:]}"
    assert passed in result.stdout, output[-6000:]


def test_hud():
    run_node("test_hud.cjs", None, "HUD_TEST_PASSED", 60, browser=False)


def test_audio_scheduling():
    run_node("test_audio.cjs", None, "AUDIO_SCHEDULING_PASS", 60, browser=False)


@pytest.mark.needs_playwright
def test_audio_browser():
    run_node("test_audio_browser.cjs", resources_dir() / "game", "AUDIO_BROWSER_PASS", 180)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.needs_castle_game
@pytest.mark.needs_playwright
@pytest.mark.needs_media
def test_browser(engine, tmp_path):
    env = dict(REFERENCE_ONLY="1", TEST_SCENE="castle")
    if engine == "webkit":
        env.update(TEST_WEBKIT="1", DISPLAY_ONLY="1")
    with serve(["--scene", "castle", "--reference-only", "--fps", "24"], tmp_path / "runtime") as base:
        run_node("test_browser.cjs", tmp_path / "out", "BROWSER_PASS", 600, TEST_URL=base, **env)


@pytest.mark.needs_racing_server
@pytest.mark.needs_playwright
@pytest.mark.needs_media
def test_racing(tmp_path):
    with serve(["--scene", "racing", "--reference-only", "--fps", "24"], tmp_path / "runtime") as base:
        run_node("test_racing.cjs", tmp_path / "out", "RACING_PASS", 600, TEST_URL=base, TEST_MODE="reference",
                 TEST_SECONDS="20")


@pytest.mark.needs_weights
@pytest.mark.needs_castle_game
@pytest.mark.needs_playwright
@pytest.mark.needs_media
def test_stability(tmp_path):
    with serve(["--scene", "castle", "--preset", "realtime", "--fps", "24"], tmp_path / "runtime") as base:
        run_node("test_stability.cjs", tmp_path / "out", "STABILITY_PASS", 900, TEST_URL=base, STABILITY_SECONDS="20")
