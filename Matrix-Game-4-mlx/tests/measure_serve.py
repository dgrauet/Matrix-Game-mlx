"""Measure one online configuration (not collected by pytest): chunk time, model_fps, online_fps, memory, swap.

    python3 -m tests.measure_serve --scene castle --preset realtime --fps 24 --chunks 30 [--browser] [--prompt-mode dynamic]
        [--dit-runtime native|ring-fused]

Starts MediaMTX and `serve` through tests.serve_process.serve, drives a session until `--chunks` chunks were produced
(an HTTP client as in the e2e tests, or with `--browser` a headless Chromium page that also reports what a player
sees: presented and game frames per second from `window.castleDebug()`), then prints one JSON line. Memory is the
summed physical footprint (top's MEM, which counts Metal buffers) of server, game, sidecar and MediaMTX; RSS is shown
for comparison only. Needs the env vars of the e2e tests (weights, MG4_RESOURCES, PLAYWRIGHT_MODULE for --browser).
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import aiohttp

from tests.serve_process import PORT_ROOT, memory_monitor, playwright_module, serve
from tests.test_serve_e2e import KEEPALIVE_S, _start, _stop, racing_controls, shooter_controls

BROWSER_SCRIPT = r"""
const {chromium} = require(process.env.PLAYWRIGHT_MODULE);
const fs = require('node:fs');
let finishing = false;
(async () => {
  const [base, out, scene] = process.argv.slice(2);
  const browser = await chromium.launch({headless: true, executablePath: chromium.executablePath()});
  const page = await browser.newPage({viewport: {width: 1280, height: 800}});
  const samples = [];
  const finish = async () => {
    finishing = true;
    await browser.close().catch(() => {});
    process.exit(0);
  };
  process.on('SIGTERM', finish);
  await page.goto(base);
  await page.selectOption('#mode', 'model');
  if (scene === 'racing') await page.selectOption('#race-mode', 'time_trial');
  await page.click('#play');
  await page.waitForFunction(() => window.castleDebug().presented > 12, {}, {timeout: 300000});
  await page.click('#surface');
  const started = Date.now();
  for (let tick = 0; ; tick++) {
    await page.keyboard.down(tick % 10 < 7 ? 'w' : 'd');
    await page.waitForTimeout(2000);
    await page.keyboard.up('w'); await page.keyboard.up('d');
    const value = await page.evaluate(() => ({presented: window.castleDebug().presented,
      gameFrame: window.castleDebug().gameFrame, frames: window.castleDebug().server?.frames}));
    samples.push({t: (Date.now() - started) / 1000, ...value});
    fs.writeFileSync(out + '/samples.json', JSON.stringify(samples));
    if (tick % 5 === 4) await page.screenshot({path: out + '/last.png'}).catch(() => {});
  }
})().catch(e => { if (!finishing) { console.error(e); process.exit(1); } });
"""


class Sampler:
    """Peak system swap and peak summed RSS of the Playwright browser processes, sampled every 2 s."""

    def __init__(self):
        self.swap_start, self.peak_swap, self.peak_browser_mb = swap_used_mb(), 0.0, 0.0
        self._stop, self._thread = threading.Event(), threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_swap = max(self.peak_swap, swap_used_mb())
            out = subprocess.run(["ps", "-A", "-o", "rss=,command="], capture_output=True, text=True).stdout
            kb = sum(int(line.split(None, 1)[0]) for line in out.splitlines() if "ms-playwright" in line)
            self.peak_browser_mb = max(self.peak_browser_mb, kb / 1024)
            self._stop.wait(2.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used =")[1].split("M")[0])


async def drive(base: str, scene: str, controls, chunks: int, timeout: float, browser_dir) -> dict:
    """Run a session until `chunks` chunks exist; HTTP-driven, or only polled while a browser plays."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http:
        node = None
        if browser_dir:
            script = Path(browser_dir) / "player.cjs"
            script.write_text(BROWSER_SCRIPT)
            env = dict(__import__("os").environ, PLAYWRIGHT_MODULE=playwright_module())
            node = subprocess.Popen(["node", str(script), base, str(browser_dir), scene], env=env)
            session, first_frame_s, ws, reader = None, None, None, None
            while session is None:                                  # the page starts the session itself
                async with http.get(base + "/api/status") as response:
                    session = (await response.json())["active"]
                await asyncio.sleep(0.5)
            first_frame_s = float("nan")
        else:
            info, first_frame_s = await _start(http, base, {"mode": "model"})
            session = info["id"]
            ws = await http.ws_connect(f"{base.replace('http', 'ws', 1)}/api/{session}/ws", headers={"Origin": base})
            reader = asyncio.create_task(_drain(ws))
        sequence, sent, deadline = 0, 0.0, time.monotonic() + timeout
        try:
            while True:
                async with http.get(base + "/api/status") as response:
                    status = await response.json()
                assert status["active"] == session, status
                if status["report"]["chunk"] >= chunks - 1:
                    break
                assert time.monotonic() < deadline, f"fewer than {chunks} chunks: {status['report']}"
                if ws is not None and time.monotonic() - sent > KEEPALIVE_S:
                    sequence += 1
                    await ws.send_json(controls(sequence))
                    sent = time.monotonic()
                await asyncio.sleep(0.5)
            if node is None:
                await _stop(http, base, session)
        finally:
            if node is not None:
                node.terminate()
                node.wait(60)
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                await ws.close()
    return {"session": session, "first_frame_s": first_frame_s}


async def _drain(ws):
    async for _ in ws:
        pass


def player_view(browser_dir: Path, window_s: float = 20.0) -> dict:
    samples = json.loads((browser_dir / "samples.json").read_text())
    last = samples[-1]
    first = next(s for s in samples if s["t"] >= last["t"] - window_s)
    span = last["t"] - first["t"]
    return {"window_s": round(span, 1), "presented_fps": round((last["presented"] - first["presented"]) / span, 2),
            "game_fps": round((last["gameFrame"] - first["gameFrame"]) / span, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="castle")
    parser.add_argument("--preset", default="realtime")
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--chunks", type=int, default=30)
    parser.add_argument("--prompt-mode")
    parser.add_argument("--dit-runtime", choices=("native", "ring-fused"), help="passed to serve (default: the preset's)")
    parser.add_argument("--browser", action="store_true", help="a headless Chromium page plays and is sampled")
    parser.add_argument("--timeout", type=float, default=1800.0)
    options = parser.parse_args()
    args = ["--scene", options.scene, "--preset", options.preset, "--fps", str(options.fps), "--profile-chunks"]
    if options.prompt_mode == "dynamic":
        args += ["--prompt-mode", "dynamic", "--prompt-interval-chunks", "1"]
    if options.dit_runtime:
        args += ["--dit-runtime", options.dit_runtime]
    controls = racing_controls if options.scene == "racing" else shooter_controls
    runtime = Path(tempfile.mkdtemp(prefix="measure-serve-"))
    failure, result = None, {"session": None, "first_frame_s": float("nan")}
    with Sampler() as sampler, serve(args, runtime, monitor_memory=True) as base:
        memory = memory_monitor(base)
        try:
            result = asyncio.run(drive(base, options.scene, controls, options.chunks, options.timeout,
                                       runtime if options.browser else None))
        except AssertionError as exc:               # keep the memory figures of a session that did not finish
            failure = str(exc)[:300]
            result["session"] = next(iter(Path(runtime).glob("session-*.json")), Path("session-.json")).stem[8:]
    report = json.loads((runtime / f"session-{result['session']}.json").read_text())
    rows = json.loads((runtime / f"profile-{result['session']}.json").read_text())["rows"]
    times = [row["processing_ms"] for row in rows[2:]]
    late = [row["processing_ms"] for row in rows[-4:]]
    out = {"scene": options.scene, "preset": options.preset, "fps": options.fps, "prompt_mode": options.prompt_mode,
           "dit_runtime": report.get("dit_runtime"),
           "browser": options.browser, "chunks": report["chunk"] + 1, "frames": report["frames"],
           "first_frame_s": round(result["first_frame_s"], 1),
           "chunk_ms_median": round(statistics.median(times)), "chunk_ms_max": round(max(times)), "chunk_ms_last4": round(statistics.mean(late)),
           "model_fps": round(report["model_fps"], 2), "online_fps": round(report["online_fps"], 2)
           if "online_fps" in report else None,
           "peak_footprint_gb": round(memory.peak_footprint_mb / 1024, 2), "peak_rss_gb": round(memory.peak_rss_mb / 1024, 2),
           "swap_delta_mb": round(sampler.peak_swap - sampler.swap_start), "peak_browser_rss_gb": round(sampler.peak_browser_mb / 1024, 2),
           "error": report["error"], "did_not_finish": failure, "runtime": str(runtime)}
    if options.browser:
        out["player_view"] = player_view(runtime)
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
