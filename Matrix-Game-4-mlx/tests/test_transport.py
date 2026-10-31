import asyncio
import os
import socket
import subprocess
import threading
import time
import unittest
import unittest.mock
import uuid

import numpy as np
import pytest

from matrix_game_mlx.server.media_env import configure_media_environment
from matrix_game_mlx.server.transport.output import WebRTCModelOutput
from tests.media import mediamtx_server

W, H, FPS = 864, 480, 24


def rtsp_describe(path, port=8555):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(f"DESCRIBE rtsp://127.0.0.1:{port}/{path} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\n\r\n".encode())
        return s.recv(4096).decode(errors="replace").split("\r\n", 1)[0]


def gradient(i):
    x = np.linspace(0, 255, W, dtype=np.uint8)[None, :, None]
    y = np.linspace(0, 255, H, dtype=np.uint8)[:, None, None]
    frame = np.empty((H, W, 3), dtype=np.uint8)
    frame[..., 0] = x[..., 0] + np.uint8(i)
    frame[..., 1] = y[..., 0]
    frame[..., 2] = x[..., 0] // 2 + y[..., 0] // 2
    return frame.tobytes()


def sidecars():
    out = subprocess.run(["pgrep", "-f", "transport.publisher"], capture_output=True, text=True).stdout
    return [p for p in out.split() if p]


@pytest.mark.needs_media
class TransportTests(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch.dict(os.environ, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        configure_media_environment()

    def output(self, report):
        return WebRTCModelOutput(W, H, FPS, uuid.uuid4().hex, threading.Event(), report, queue_size=4)

    def feed(self, out, chunks, on_chunk=None):
        """Async driver: start, submit `chunks` x 12 paced frames, always close."""
        async def run():
            result = None
            try:
                await out.start()
                for chunk in range(chunks):
                    out.note_frames(12, chunk_index=chunk, game_frames=[
                        {"frame": chunk * 12 + k, "capture_monotonic": time.monotonic()} for k in range(12)])
                    for k in range(12):
                        out(gradient(chunk * 12 + k))
                        await asyncio.sleep(1 / FPS)
                    if on_chunk:
                        result = on_chunk(chunk) or result
                return result
            finally:
                await out.close()
        return asyncio.run(run())

    def test_rgb_reaches_mediamtx(self):
        report = {"_started_monotonic": time.perf_counter()}    # set by the session in production
        out = self.output(report)
        path = f"native-{out.command[-1]}"
        with mediamtx_server():
            described = self.feed(out, 4, lambda chunk: rtsp_describe(path) if chunk == 2 else None)
        self.assertGreaterEqual(report["media_encoded_frames"], 1)
        self.assertIn("200", described)
        self.assertEqual(sidecars(), [])

    def test_missing_mediamtx_fails_fast(self):
        out = self.output({"_started_monotonic": time.perf_counter()})
        began = time.monotonic()
        with self.assertRaises(Exception):
            self.feed(out, 10)
        self.assertLess(time.monotonic() - began, 30)
        self.assertEqual(sidecars(), [])


if __name__ == "__main__":
    unittest.main()
