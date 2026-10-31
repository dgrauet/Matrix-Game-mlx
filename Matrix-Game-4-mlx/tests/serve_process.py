"""Run `python3 -m matrix_game_mlx.cli serve` as a child process for the end-to-end tests (and measurements)."""
import contextlib
import functools
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

from tests.media import mediamtx_server
from tests.racing_game import resources_dir

PORT_ROOT = Path(__file__).resolve().parents[1]
GAME_ROOT = "../Matrix-Game-4/game"          # relative to PORT_ROOT, the serve child's working directory
PORT = 8025
# Child processes a session starts (they stay in the server's process group); none may outlive the server.
CHILDREN = ("CastleReferenceServer", "CitySampleReferenceServer", "transport.publisher")
DEFAULT_PLAYWRIGHT = Path.home() / ".cache/matrix-game-4-mlx/playwright/node_modules/playwright"

_monitors = {}                                # base URL -> MemoryMonitor of a serve(monitor_memory=True)


def playwright_module() -> str:
    """PLAYWRIGHT_MODULE, else the out-of-tree install under ~/.cache/matrix-game-4-mlx/playwright."""
    return os.environ.get("PLAYWRIGHT_MODULE") or str(DEFAULT_PLAYWRIGHT)


@functools.lru_cache(maxsize=None)
def playwright_missing() -> Optional[str]:
    """None when node can require the Playwright module, else what to do."""
    env = dict(os.environ, PLAYWRIGHT_MODULE=playwright_module())
    try:
        result = subprocess.run(["node", "-e", "require(process.env.PLAYWRIGHT_MODULE)"], env=env,
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"node is not usable: {exc}"
    if result.returncode:
        return (f"Playwright not found at {env['PLAYWRIGHT_MODULE']}; install it outside the repo: "
                f"cd {DEFAULT_PLAYWRIGHT.parents[1]} && npm init -y && npm install playwright "
                "&& npx playwright install chromium webkit (or set PLAYWRIGHT_MODULE)")
    return None


def get_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _group(pgid: int) -> list:
    """`pid command` lines of the processes still in process group `pgid`."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,command="], capture_output=True, text=True).stdout
    rows = (line.split(None, 2) for line in out.splitlines())
    return [f"{row[0]} {row[2]}" for row in rows if len(row) == 3 and row[1] == str(pgid)]


def _log_tail(path: Path, lines: int = 40) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) if path.is_file() else ""


@contextlib.contextmanager
def serve(args: list, runtime: Path, ready_timeout: float = 900.0, monitor_memory: bool = False) -> Iterator[str]:
    """Start MediaMTX, then the serve CLI with `args`; yield its base URL once /api/status answers ready.

    On exit the server gets SIGTERM (aiohttp stops the active session in its cleanup) and is waited on; the context
    fails if any process of its group (game, media sidecar) survives it. The log is `runtime/serve.log`. With
    `monitor_memory`, `memory_monitor(base)` samples memory from the server's start to its exit.
    """
    runtime = Path(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    log_path = runtime / "serve.log"
    command = [sys.executable, "-m", "matrix_game_mlx.cli", "serve", "--game-root", GAME_ROOT,
               "--resources", str(resources_dir()), "--runtime", str(runtime), "--port", str(PORT), *args]
    base = f"http://127.0.0.1:{PORT}"
    with mediamtx_server(), open(log_path, "w") as log:
        # Its own process group, so the game and the sidecar it starts can be found (and reaped) afterwards.
        proc = subprocess.Popen(command, cwd=PORT_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"))
        monitor = MemoryMonitor(proc.pid) if monitor_memory else contextlib.nullcontext()
        try:
            monitor.__enter__()
            deadline = time.monotonic() + ready_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"serve exited with code {proc.returncode}:\n{_log_tail(log_path)}")
                try:
                    if get_json(base + "/api/status", timeout=2).get("ready"):
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"serve not ready after {ready_timeout:g} s:\n{_log_tail(log_path)}")
                time.sleep(0.5)
            if monitor_memory:
                _monitors[base] = monitor
            yield base
        finally:
            _monitors.pop(base, None)
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            monitor.__exit__(None, None, None)
            survivors = []
            for _ in range(20):                       # children finishing their own shutdown get a few seconds
                survivors = _group(proc.pid)
                if not survivors:
                    break
                time.sleep(0.25)
            if survivors:
                os.killpg(proc.pid, signal.SIGKILL)
                named = [line for line in survivors if any(name in line for name in CHILDREN)]
                raise AssertionError("processes survived the server (session children: "
                                     f"{named or 'none'}): " + "; ".join(survivors))


class MemoryMonitor:
    """Peak summed memory of the server's process group and MediaMTX, sampled each second.

    `peak_rss_mb` sums `ps -o rss`; `peak_footprint_mb` sums top's MEM (the physical footprint), which also counts
    the Metal buffers MLX allocates, which RSS does not show.
    """

    def __init__(self, pgid: int):
        self.pgid, self.peak_rss_mb, self.peak_footprint_mb = pgid, 0.0, 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        out = subprocess.run(["ps", "-A", "-o", "pid=,rss=,pgid=,comm="], capture_output=True, text=True).stdout
        pids, rss = [], 0
        for line in out.splitlines():
            pid, kb, pgid, comm = (line.split(None, 3) + ["", "", ""])[:4]
            if pgid == str(self.pgid) or comm.endswith("mediamtx"):
                pids.append(pid)
                rss += int(kb)
        self.peak_rss_mb = max(self.peak_rss_mb, rss / 1024)
        if pids:
            top = subprocess.run(["top", "-l", "1", "-stats", "pid,mem", *[a for pid in pids for a in ("-pid", pid)]],
                                 capture_output=True, text=True).stdout
            footprint = sum(_megabytes(row.split()[1]) for row in top.splitlines()
                            if len(row.split()) == 2 and row.split()[0] in pids)
            self.peak_footprint_mb = max(self.peak_footprint_mb, footprint)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:                         # one failed ps/top sample must not end the sampling
                pass                                  # (the peaks of the last good samples are kept)
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def _megabytes(value: str) -> float:
    """top's MEM column ("512K", "1408M", "15G", optional trailing +/-) in MB."""
    value = value.rstrip("+-")
    scale = {"B": 1 / 2**20, "K": 1 / 1024, "M": 1, "G": 1024, "T": 2**20}
    return float(value[:-1]) * scale[value[-1]] if value[-1] in scale else float(value) / 2**20


def memory_monitor(base: str) -> MemoryMonitor:
    """The MemoryMonitor of the server `serve(..., monitor_memory=True)` yielded `base` for; final once it exits."""
    return _monitors[base]
