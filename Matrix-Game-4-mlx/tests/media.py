"""MediaMTX helper for the media tests (online play transport)."""
import contextlib
import shutil
import socket
import subprocess
import time
from typing import Optional

from matrix_game_mlx.server.media_env import default_sidecar_python
from tests.racing_game import UPSTREAM

MEDIAMTX_CONFIG = UPSTREAM / "configs/mediamtx.yml"
PORTS = (8555, 8890, 8190)          # RTSP, WebRTC HTTP, WebRTC ICE/TCP (configs/mediamtx.yml)


def media_missing() -> Optional[str]:
    """None when mediamtx and a GI-enabled sidecar Python are available, else what to install."""
    if not shutil.which("mediamtx"):
        return "mediamtx not found, run: brew install mediamtx"
    try:
        default_sidecar_python()
    except RuntimeError as e:
        return str(e)
    return None


def _accepts(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


@contextlib.contextmanager
def mediamtx_server(timeout: float = 15.0):
    """Run MediaMTX with the upstream config; terminated on exit. Fails clearly if a port is already taken."""
    taken = [p for p in PORTS if _accepts(p)]
    if taken:
        raise RuntimeError(f"port(s) {taken} already in use (another MediaMTX?); free them before running the media tests")
    proc = subprocess.Popen(["mediamtx", str(MEDIAMTX_CONFIG)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + timeout
        while not (_accepts(8555) and _accepts(8890)):
            if proc.poll() is not None:
                raise RuntimeError(f"mediamtx exited with code {proc.returncode}")
            if time.monotonic() > deadline:
                raise RuntimeError("mediamtx did not open 127.0.0.1:8555 and :8890 in time")
            time.sleep(0.1)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
