"""End-to-end online sessions: the serve CLI with the native game, the media sidecar and MediaMTX, over HTTP and ws.

Each session is driven like the browser client drives it: same-origin `POST /api/start`, controls over
`/api/{id}/ws`, `/api/status` polling, `POST /api/{id}/stop`; the reports the server writes are then checked.
Model cases print their time to first encoded frame and the peak RSS of all processes involved.
"""
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import pytest

from tests.serve_process import memory_monitor, serve

SHOOTER = [pytest.mark.needs_media, pytest.mark.needs_castle_game]
RACING = [pytest.mark.needs_media, pytest.mark.needs_racing_server]
MODEL = [pytest.mark.needs_weights]
MIN_FRAMES = 9                    # the first chunk (1 frame) and two 4-frame chunks
KEEPALIVE_S = 2.0                 # the browser sends controls continuously; the server ends a session silent for 15 s


def shooter_controls(sequence: int) -> dict:
    variants = ({"forward": 1}, {"look_x": 40, "look_y": -5}, {"fire": True}, {"weapon": 1},
                {"right": -1, "actions": ["jump"]})
    return dict({"sequence": sequence, "forward": 0, "right": 0, "look_x": 0, "look_y": 0, "look_reset": False,
                 "fire": False, "ads": False, "shield": False, "crouch": False, "weapon": -1, "actions": []},
                **variants[sequence % len(variants)])


def racing_controls(sequence: int) -> dict:
    drives = ([1, 0, 0, 0], [1, 1, 0, 0], [1, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1])
    return {"sequence": sequence, "drive": drives[sequence % len(drives)]}


async def _drain(ws):
    async for _ in ws:            # reading also answers the server's heartbeat pings
        pass


async def _start(http, base: str, body: dict):
    started = time.monotonic()
    async with http.post(base + "/api/start", json=body, headers={"Origin": base}) as response:
        assert response.status == 200, await response.text()
        info = await response.json()
    return info, time.monotonic() - started


async def _stop(http, base: str, session: str):
    async with http.post(f"{base}/api/{session}/stop", data="{}",
                         headers={"Origin": base, "Content-Type": "application/json"}) as response:
        assert response.status == 200, await response.text()
        assert await response.json() == {"stopped": True}


async def _session(base: str, body: dict, controls, min_frames: int, frames_timeout: float):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http:
        info, first_frame_s = await _start(http, base, body)
        session = info["id"]
        async with http.ws_connect(f"{base.replace('http', 'ws', 1)}/api/{session}/ws",
                                   headers={"Origin": base}) as ws:
            reader = asyncio.create_task(_drain(ws))
            sequence = 0
            for _ in range(5):
                sequence += 1
                await ws.send_json(controls(sequence))
            deadline, sent = time.monotonic() + frames_timeout, time.monotonic()
            while True:
                async with http.get(base + "/api/status") as response:
                    status = await response.json()
                assert status["active"] == session, status
                if status["report"]["frames"] >= min_frames:
                    break
                assert time.monotonic() < deadline, f"fewer than {min_frames} frames: {status['report']}"
                if time.monotonic() - sent > KEEPALIVE_S:
                    sequence += 1
                    await ws.send_json(controls(sequence))
                    sent = time.monotonic()
                await asyncio.sleep(0.25)
            await _stop(http, base, session)
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
    return session, first_frame_s


def run_session(base: str, runtime: Path, body: dict, controls, min_frames: int = MIN_FRAMES,
                frames_timeout: float = 300.0):
    """Start, control, poll to `min_frames`, stop; return (session report, profile or None, seconds to first frame)."""
    session, first_frame_s = asyncio.run(_session(base, body, controls, min_frames, frames_timeout))
    report = json.loads((runtime / f"session-{session}.json").read_text())
    assert report["error"] is None, report
    assert report["frames"] >= min_frames, report
    profile_path = runtime / f"profile-{session}.json"
    profile = json.loads(profile_path.read_text()) if profile_path.is_file() else None
    return report, profile, first_frame_s


@pytest.fixture
def log_line(capsys):
    def log(text: str):
        with capsys.disabled():
            print(f"\n[e2e] {text}", flush=True)
    return log


def model_session(args, body, controls, tmp_path, log_line, label, **kwargs):
    with serve(args, tmp_path, monitor_memory=True) as base:
        memory = memory_monitor(base)
        report, profile, first_frame_s = run_session(base, tmp_path, body, controls, **kwargs)
    log_line(f"{label}: first encoded frame after {first_frame_s:.1f} s, {report['frames']} frames, "
             f"model_fps {report.get('model_fps', 0):.2f}, peak memory (all processes): "
             f"RSS {memory.peak_rss_mb / 1024:.2f} GB, footprint {memory.peak_footprint_mb / 1024:.2f} GB")
    assert report["mode"] == "model"
    return report, profile, first_frame_s


@pytest.mark.parametrize("scene", [pytest.param("castle", marks=SHOOTER), pytest.param("ancient-egypt", marks=SHOOTER),
                                   pytest.param("racing", marks=RACING)])
def test_reference_only(scene, tmp_path):
    controls = racing_controls if scene == "racing" else shooter_controls
    with serve(["--scene", scene, "--reference-only", "--fps", "24"], tmp_path) as base:
        report, _, _ = run_session(base, tmp_path, {"mode": "reference"}, controls)
    assert report["mode"] == "reference"


@pytest.mark.parametrize("scene", [pytest.param("castle", marks=SHOOTER + MODEL),
                                   pytest.param("racing", marks=RACING + MODEL)])
def test_model_realtime(scene, tmp_path, log_line):
    controls = racing_controls if scene == "racing" else shooter_controls
    args = ["--scene", scene, "--preset", "realtime", "--fps", "24", "--profile-chunks"]
    _, profile, first_frame_s = model_session(args, {"mode": "model"}, controls, tmp_path, log_line,
                                              f"realtime {scene}")
    assert first_frame_s < 45
    assert profile is not None and profile["rows"]
    for row in profile["rows"]:
        assert "stream_wall_ms" in row and "gpu_decode_ms" in row, sorted(row)


@pytest.mark.needs_media
@pytest.mark.needs_racing_server
@pytest.mark.needs_weights
def test_model_realtime_racing_dynamic(tmp_path, log_line):
    args = ["--scene", "racing", "--prompt-mode", "dynamic", "--prompt-interval-chunks", "1", "--fps", "24"]
    report, _, _ = model_session(args, {"mode": "model"}, racing_controls, tmp_path, log_line,
                                 "realtime racing dynamic")
    assert report["prompt_updates"] >= 1, report


@pytest.mark.slow
@pytest.mark.needs_media
@pytest.mark.needs_castle_game
@pytest.mark.needs_weights
def test_model_quality_castle(tmp_path, log_line):
    args = ["--scene", "castle", "--preset", "quality", "--fps", "24"]
    _, _, first_frame_s = model_session(args, {"mode": "model"}, shooter_controls, tmp_path, log_line,
                                        "quality castle", frames_timeout=900.0)
    assert first_frame_s < 300                    # first_frame_timeout for quality


@pytest.mark.needs_media
@pytest.mark.needs_castle_game
@pytest.mark.needs_weights
def test_no_early_publish_and_serial_pipeline(tmp_path, log_line):
    args = ["--scene", "castle", "--preset", "realtime", "--fps", "24", "--no-early-publish",
            "--reference-pipeline", "serial"]
    report, _, _ = model_session(args, {"mode": "model"}, shooter_controls, tmp_path, log_line,
                                 "realtime castle, no early publish, serial pipeline")
    assert report["early_publish"] is False


async def _disconnect_then_restart(base: str) -> tuple:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http:
        info, _ = await _start(http, base, {"mode": "reference"})
        async with http.ws_connect(f"{base.replace('http', 'ws', 1)}/api/{info['id']}/ws",
                                   headers={"Origin": base}) as ws:
            for sequence in range(1, 6):
                await ws.send_json(shooter_controls(sequence))
        # The ws is closed and nothing polls the session any more: the server must reclaim it on its own.
        deadline, active = time.monotonic() + 20, info["id"]
        while active is not None and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            async with http.get(base + "/api/status") as response:
                active = (await response.json())["active"]
        assert active is None, f"session {active} still active 20 s after the browser left"
        again, _ = await _start(http, base, {"mode": "reference"})
        await _stop(http, base, again["id"])
    return info["id"], again["id"]


@pytest.mark.needs_media
@pytest.mark.needs_castle_game
def test_browser_disconnect_cleans_up(tmp_path):
    with serve(["--scene", "castle", "--reference-only", "--fps", "24"], tmp_path) as base:
        first, second = asyncio.run(_disconnect_then_restart(base))
    assert first != second
    for session in (first, second):
        assert (tmp_path / f"session-{session}.json").is_file()
    assert json.loads((tmp_path / f"session-{second}.json").read_text())["error"] is None
