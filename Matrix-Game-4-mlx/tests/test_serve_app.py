"""macOS changes of the online app (renderer names, startup deadline, media environment), its argument validation,
the copied web client, and the session lifecycle through the real aiohttp app with a fake game and a fake media
output (no model, GPU, native game, GStreamer or MediaMTX)."""
import asyncio
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from matrix_game_mlx import scenes
from matrix_game_mlx.server import app as serve_app
from matrix_game_mlx.server.app import first_frame_timeout, validate_renderer

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_WEB = ROOT.parent / "Matrix-Game-4" / "matrix_game" / "web"
PORT_WEB = ROOT / "matrix_game_mlx" / "web"
RACING_SOUNDS = ("engine-loop", "tire-loop", "tire-front-loop", "impact-medium", "impact-heavy")


@pytest.fixture(autouse=True)
def fake_pack(tmp_path, monkeypatch):
    """Offline pack holding only config.json; pack_dir never hits the network."""
    pack = tmp_path / "pack"
    pack.mkdir()
    cfg = {"scenes": {s: {"component": f"dit_{s}", "inference": f"{s}.json", "sink_image": f"{s}_sink.png"}
                      for s in scenes.SCENES}}
    (pack / "config.json").write_text(json.dumps(cfg))
    monkeypatch.delenv("MATRIX_GAME_4_MLX_WEIGHTS_DIR", raising=False)
    monkeypatch.setattr("matrix_game_mlx.inference.loader.pack_dir", lambda components=None: pack)
    return pack


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Run `main` without loading a model, touching the process environment or serving: returns the record of
    calls and the argv prefix of a complete (fake) game root, resources and runtime dir."""
    game_root, resources = tmp_path / "game", tmp_path / "resources"
    for binary in (game_root / "bin/CastleReferenceServer", game_root / "racing/bin/CitySampleReferenceServer"):
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"")
    sounds = [resources / "game/assets/combat-audio" / f"{i}.wav" for i in range(7)]
    sounds += [resources / "racing/assets/audio" / f"{name}.wav" for name in RACING_SOUNDS]
    for sound in sounds:
        sound.parent.mkdir(parents=True, exist_ok=True)
        sound.write_bytes(b"RIFF")
    calls = SimpleNamespace(order=[], app=None, runtime=None)
    monkeypatch.setattr("matrix_game_mlx.server.media_env.configure_media_environment",
                        lambda *a, **k: calls.order.append("media"))
    monkeypatch.setattr(serve_app.Runtime, "load", lambda self: calls.order.append("load"))

    def run_app(app, **kwargs):
        calls.order.append("run_app")
        calls.app = app
    monkeypatch.setattr(serve_app.web, "run_app", run_app)
    original = serve_app.Runtime.__init__

    def init(self, args):
        original(self, args)
        calls.runtime = self
    monkeypatch.setattr(serve_app.Runtime, "__init__", init)
    calls.game_root = game_root
    calls.argv = ["--game-root", str(game_root), "--resources", str(resources), "--runtime", str(tmp_path / "runtime")]
    return calls


class FakeGame:
    """CastleGame stand-in: same handshake fields and chunk cadence (1 then 4 frames), RGBA frames prepared by the
    session's real ReferencePreprocessor."""

    def __init__(self, root, *, fps, log_path, scene="castle", expected_prompt=None, asset_root=None, prepare=None):
        self.fps, self.prepare = fps, prepare
        self.hello = dict(type="hello", scene=scene, renderer="Apple M2 Pro", audio_schema=1, mouse_input_schema=2)
        self.ended = None
        self.frame = 0
        self.sent = []
        self.closed = False

    def next(self, cancel):
        if cancel.wait(.01) or self.closed:
            return None
        count = 1 if self.frame == 0 else 4
        frames = [np.full((72, 128, 4), 255, np.uint8) for _ in range(count)]
        states = [dict(frame=self.frame + i, simulation_time=(self.frame + i) / self.fps) for i in range(count)]
        self.frame += count
        self.prepare.begin()
        return self.prepare.finish([self.prepare.frame(frame) for frame in frames]), states

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


class FakeOutput:
    """WebRTCModelOutput stand-in: counts submitted frames as encoded and yields one metadata record per chunk."""
    encodes = True

    def __init__(self, width, height, fps, generation_id, cancel, report, *, queue_size=8):
        self.report, self.width = report, width
        self.records, self.pending = deque(), 0
        self.closed = False
        report.update(media_encoded_frames=0)

    async def start(self):
        pass

    def check(self):
        pass

    def inference_done(self, future):
        pass

    def note_frames(self, count, ready_wall_seconds=None, chunk_index=None, *, game_frames=None):
        assert len(game_frames) == count
        self.pending = count
        self.records.append(json.dumps(dict(type="chunk", chunk=chunk_index, frames=count)))

    def __call__(self, packed):
        assert len(packed) == self.pending * 480 * self.width * 3
        if self.encodes:
            self.report["media_encoded_frames"] += self.pending

    async def stream(self):
        while not self.closed:
            while self.records:
                yield self.records.popleft()
            await asyncio.sleep(.01)

    async def close(self):
        self.closed = True


class SilentOutput(FakeOutput):
    encodes = False


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr("matrix_game_mlx.server.game.CastleGame", FakeGame)
    monkeypatch.setattr("matrix_game_mlx.server.transport.output.WebRTCModelOutput", FakeOutput)


async def _status(client):
    return await (await client.get("/api/status")).json()


async def _wait_frames(client, frames, timeout=20):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        report = (await _status(client))["report"]
        if report and report["frames"] >= frames:
            return report
        await asyncio.sleep(.05)
    raise AssertionError(f"session did not reach {frames} frames")


def test_apple_software_renderer_rejected():
    with pytest.raises(RuntimeError, match="software OpenGL"):
        validate_renderer({"renderer": "Apple Software Renderer"})
    assert validate_renderer({"renderer": "Apple Software Renderer"}, allow_software=True) == "Apple Software Renderer"
    assert validate_renderer({"renderer": "Apple M2 Pro"}) == "Apple M2 Pro"


def test_first_frame_timeout():
    assert first_frame_timeout(SimpleNamespace(preset="realtime")) == 45
    assert first_frame_timeout(SimpleNamespace(preset="quality")) == 300
    assert first_frame_timeout(SimpleNamespace(preset="baseline")) == 300


def test_serve_argument_validation(served, capsys):
    with pytest.raises(SystemExit) as stopped:
        serve_app.main(served.argv + ["--fps", "7"])
    assert stopped.value.code == 2
    assert "FPS must be 1..60, and divide 120 for shooters" in capsys.readouterr().err
    assert served.order == []

    serve_app.main(served.argv + ["--scene", "racing", "--fps", "7"])
    assert "FPS must be" not in capsys.readouterr().err
    # The media environment is configured after validation and before the model and the server start.
    assert served.order == ["media", "load", "run_app"]

    with pytest.raises(SystemExit) as stopped:
        serve_app.main(served.argv + ["--preset", "rtx5090"])
    assert stopped.value.code == 2
    assert "--preset rtx5090 is CUDA-only" in capsys.readouterr().err

    with pytest.raises(SystemExit) as stopped:
        serve_app.main(served.argv[2:] + ["--game-root", str(served.game_root.parent / "unbuilt")])
    assert stopped.value.code == 2
    assert "Build the scene's native game first; see docs/installation.md" in capsys.readouterr().err


def test_web_assets_identical():
    upstream = {p.relative_to(UPSTREAM_WEB): p.read_bytes() for p in UPSTREAM_WEB.rglob("*") if p.is_file()}
    port = {p.relative_to(PORT_WEB): p.read_bytes() for p in PORT_WEB.rglob("*") if p.is_file()}
    assert len(upstream) == 6
    assert sorted(port) == sorted(upstream)
    for name, data in upstream.items():
        assert port[name] == data, name


def test_session_report_keys(served, fakes, tmp_path):
    serve_app.main(served.argv + ["--scene", "castle", "--reference-only", "--profile-chunks"])
    assert served.order == ["media", "run_app"]

    async def play():
        async with TestClient(TestServer(served.app)) as client:
            response = await client.post("/api/start", json={"mode": "reference"})
            assert response.status == 200, await response.text()
            session = await response.json()
            assert session["whep"] == f"http://127.0.0.1:8890/native-{session['id']}/whep"
            await _wait_frames(client, 9)
            response = await client.post(f"/api/{session['id']}/stop", json={})
            assert await response.json() == {"stopped": True}
            assert (await _status(client))["active"] is None
            return session["id"]
    session_id = asyncio.run(play())

    report = json.loads((tmp_path / "runtime" / f"session-{session_id}.json").read_text())
    for key in ("frames", "chunk", "mode", "monitor", "display_width", "game_renderer", "error"):
        assert key in report, key
    assert report["frames"] >= 9 and report["chunk"] >= 2
    assert (report["mode"], report["monitor"], report["display_width"]) == ("reference", False, 864)
    assert (report["game_renderer"], report["error"]) == ("Apple M2 Pro", None)
    profile = json.loads((tmp_path / "runtime" / f"profile-{session_id}.json").read_text())
    assert set(profile) == {"config", "summary", "rows"}
    assert profile["config"]["profile_chunks"] is True
    assert [row["chunk"] for row in profile["rows"]][:3] == [0, 1, 2]
    assert all("publish_enqueue_ms" in row for row in profile["rows"])


def test_second_start_conflicts_then_restarts(served, fakes, tmp_path):
    serve_app.main(served.argv + ["--scene", "castle", "--reference-only"])

    async def play():
        async with TestClient(TestServer(served.app)) as client:
            first = await (await client.post("/api/start", json={"mode": "reference"})).json()
            response = await client.post("/api/start", json={"mode": "reference"})
            assert response.status == 409
            assert "already active" in await response.text()
            await _wait_frames(client, 5)
            assert (await client.post(f"/api/{first['id']}/stop", json={})).status == 200
            response = await client.post("/api/start", json={"mode": "reference"})
            assert response.status == 200, await response.text()
            second = await response.json()
            assert second["id"] != first["id"]
            await _wait_frames(client, 5)
            assert (await client.post(f"/api/{second['id']}/stop", json={})).status == 200
            return first["id"], second["id"]
    ids = asyncio.run(play())
    for session_id in ids:
        assert json.loads((tmp_path / "runtime" / f"session-{session_id}.json").read_text())["error"] is None


def test_session_start_uses_first_frame_timeout(served, monkeypatch, tmp_path):
    """A session whose first frame is never encoded fails at the scene's deadline, with upstream's message."""
    monkeypatch.setattr("matrix_game_mlx.server.game.CastleGame", FakeGame)
    monkeypatch.setattr("matrix_game_mlx.server.transport.output.WebRTCModelOutput", SilentOutput)
    monkeypatch.setattr(serve_app, "first_frame_timeout", lambda args: .3)
    serve_app.main(served.argv + ["--scene", "castle", "--reference-only"])
    runtime = served.runtime

    async def start():
        with pytest.raises(TimeoutError, match=r"^No encoded first frame after 0\.3 seconds$"):
            await runtime.start("reference")
        assert runtime.active is None
    asyncio.run(start())
    assert len(list((tmp_path / "runtime").glob("session-*.json"))) == 1


class FakeStream:
    """CastleStream stand-in with the MLX push contract: on_rgb gets an owned uint8 THWC array, then Chunk.rgb is
    that array and Chunk.pixels is None; without on_rgb, Chunk.pixels is the decoded [T, 3, H, W] mx.array."""

    def __init__(self, profile=False):
        self.profile, self.closed, self.published = profile, False, []

    def push(self, pixels, *, frame_start, on_rgb=None):
        import mlx.core as mx
        count = pixels.shape[0]
        names = ("encode", "denoise", "clean_kv", "decode") if on_rgb is None else (
            "encode", "denoise", "decode", "pack_copy", "clean_kv")
        timings = {name: 1. for name in names} if self.profile else {}
        if on_rgb is None:
            return SimpleNamespace(pixels=mx.zeros((count, 3, 480, 864)), rgb=None, timings_ms=timings)
        rgb = np.full((count, 480, 864, 3), 128, np.uint8)
        on_rgb(rgb)
        self.published.append(rgb)
        return SimpleNamespace(pixels=None, rgb=rgb, timings_ms=timings)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("early", [True, False])
def test_model_session_gpu_timings(served, fakes, monkeypatch, tmp_path, early):
    streams = []

    def stream(self, **kwargs):
        streams.append(FakeStream(**kwargs))
        return streams[-1]
    monkeypatch.setattr(serve_app.Runtime, "stream", stream)
    argv = ["--scene", "castle", "--reference-only", "--profile-chunks"] + ([] if early else ["--no-early-publish"])
    serve_app.main(served.argv + argv)
    served.runtime.gen = object()  # a loaded model, as far as /api/start is concerned

    async def play():
        async with TestClient(TestServer(served.app)) as client:
            response = await client.post("/api/start", json={"mode": "model"})
            assert response.status == 200, await response.text()
            session = await response.json()
            await _wait_frames(client, 9)
            assert (await client.post(f"/api/{session['id']}/stop", json={})).status == 200
            return session["id"]
    session_id = asyncio.run(play())

    assert len(streams) == 1 and streams[0].closed and streams[0].profile
    assert bool(streams[0].published) is early
    report = json.loads((tmp_path / "runtime" / f"session-{session_id}.json").read_text())
    assert (report["mode"], report["early_publish"], report["error"]) == ("model", early, None)
    rows = json.loads((tmp_path / "runtime" / f"profile-{session_id}.json").read_text())["rows"]
    stages = ("encode", "denoise", "decode", "pack_copy", "clean_kv") if early else (
        "encode", "denoise", "clean_kv", "decode")
    for row in rows:
        assert {"gpu_" + stage + "_ms" for stage in stages} <= set(row)
        assert not any(key.startswith("cuda_") for key in row)
        assert {"stream_wall_ms", "materialize_rgb_ms", "publish_enqueue_ms"} <= set(row)
