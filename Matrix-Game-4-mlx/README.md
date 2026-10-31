# Matrix-Game-4-mlx

MLX inference port of [Matrix-Game 4.0](https://github.com/SkyworkAI/Matrix-Game) for Apple Silicon.
It targets offline rendering and an online browser demo (`serve`): a causal video-to-video renderer that turns a clip into a stylized one
using the quality preset (Wan VAE) or the realtime preset (TAEHV codec, one denoising step), for the
ancient-egypt, castle and racing scenes.

## Status

| Milestone | Scope | Status |
| --- | --- | --- |
| M1 | ancient-egypt, quality preset (Wan VAE, steps `1000,500`), offline CLI, real-weight parity | Done: parity-validated; a 19 s recorded take (909 frames, 48 FPS) rendered in 1 h 52 min (29.4 s/chunk, peak 24.4 GB) |
| M2 | castle and racing (first-frame sink, absolute RoPE, racing RNG role), fixtures for both | Done: parity-validated, same methodology as ancient-egypt (racing RNG role transcribed from upstream, not parity-validated: parity injects noise on both sides); acceptance renders: 97-frame pan renders of castle (37.7 s/chunk) and racing (27.3 s/chunk), peak 24.4 GB |
| M3 | realtime preset: TAEHV encode/decode (`taehv.py`), one-step schedule, upstream `native` DiT runtime | Done: parity-validated on the three scenes (`<scene>_realtime` fixtures, torch bf16 CPU reference); acceptance renders: 97-frame pans of egypt (3.57 s/chunk), castle (3.58 s/chunk) and racing (3.51 s/chunk), peak 14.96 GB (~1.1 processed frames/s, M2 Pro 32 GB) |

## Quick start

```bash
cd Matrix-Game-4-mlx
pip install -e .
matrix-game-mlx render --scene ancient-egypt --input lowpoly.mp4 --output output/egypt
```

`matrix-game-mlx` is the port of upstream's `matrix-game` CLI. Subcommands: `render`, `download`, `verify`, `serve`
(the online browser demo, see "Online play on macOS").

To render a game take recorded with the upstream game (`bash scripts/play.sh --scene ancient-egypt` in
`../Matrix-Game-4/`, F9 to start and stop), pass the take directory; the scene comes from its `session.json`:

```bash
matrix-game-mlx render --take /path/to/take-1 --output output/take-1
```

The output directory gets `rendered.mp4` (same frame count and FPS as the input; the last incomplete
four-frame block is padded internally, then trimmed) and `render.json` (configuration, frames, seconds,
`seconds_per_chunk`, `peak_memory_gb`, `prompt_mode`). Realtime example:

```bash
matrix-game-mlx render --scene castle --preset realtime --input lowpoly.mp4 --output output/castle-realtime
```

### `render` flags

The offline default preset is `quality`, as upstream's offline renderer. Weights come only from the MLX pack
(see Weights), so upstream's weight-path flags and CUDA-only options are rejected with an error.

| Flag | Status |
| --- | --- |
| `--scene {ancient-egypt,castle,racing}` | Supported |
| `--preset {realtime,quality,baseline}` | Supported (default `quality`) |
| `--preset {rtx5090,rtx5090-sage}` | Rejected: CUDA-only presets |
| `--vae-backend {wan,taehv}` | Supported (default from the preset: `taehv` for realtime, `wan` otherwise) |
| `--dit-runtime native` | Supported (the only runtime) |
| `--dit-runtime` other values (`kernels`, `cudagraph`, `ring`, `ring-graph`, `ring-fused`, `rtx5090`, `rtx5090-sage`) | Rejected: CUDA-only; the MLX port runs `native` only |
| `--steps`, `--first-steps` | Supported (comma-separated timesteps, strictly decreasing from 1000) |
| `--vae-graph` | Rejected: CUDA graphs, not available on MLX |
| `--no-vae-graph` | Accepted, no effect (VAE graphs are never used on MLX) |
| `--prompt-mode {fixed,dynamic}`, `--prompt`, `--prompt-interval-chunks` | Supported; dynamic is racing only (see below) |
| `--captions FILE` | Supported, dynamic mode only (required there) |
| `--t5-runtime {native,graph}` | Accepted; `graph` behaves as `native` (no CUDA graphs) |
| `--sink-image PATH`, `--no-sink-image` | Supported (mutually exclusive) |
| `--resources DIR` | Accepted; `render` does not read weights or the sink image from it (those come from the MLX pack). It sets the game-asset root and is the default `download --resources` target |
| `--asset-root DIR` | Accepted: overrides the game-asset root derived from `--resources`; `render` does not read it |
| `--input CLIP` / `--take DIR`, `--output DIR`, `--seed N` | Supported |
| `--model-dir`, `--checkpoint`, `--taehv-path` | Rejected: weights come only from the MLX pack (`MATRIX_GAME_4_MLX_WEIGHTS_DIR` selects a local pack) |

### Dynamic captions (racing)

`--prompt-mode dynamic --captions FILE` switches the text prompt during the render. The file is JSONL with one
record per decoded chunk: `{"frame_start": int, "frame_end": int, "prompt": str}`. Chunk 0 is one frame, then
every chunk is four frames, so a 9-frame clip has three records (`0..0`, `1..4`, `5..8`). The ranges must match
the clip's chunks. A complete example is `tests/data/racing_captions.jsonl`:

```bash
matrix-game-mlx render --scene racing --preset realtime --prompt-mode dynamic \
  --captions tests/data/racing_captions.jsonl --input clip.mp4 --output output/racing-dynamic
```

`--prompt-interval-chunks N` (1 to 1024) only re-reads the caption every N chunks. Dynamic mode keeps the umT5
encoder resident (~11 GB) instead of freeing it after encoding the fixed prompt. The realtime preset fits a 32 GB
machine (measured peak 26.3 GB on a 9-frame 864x480 clip, M2 Pro 32 GB); the quality preset with dynamic captions
needs more than 32 GB.

### Game assets: `download` and `verify`

`releases/game-assets.json` is a byte copy of upstream's asset manifest. Scene resources (not weights) are fetched
and checked with:

```bash
matrix-game-mlx download --manifest releases/game-assets.json --repo-id Skywork/Matrix-Game-4.0 \
  --repo-type model --revision 4419accdb7bb7881234c5914de8cd915266e0325 --scene ancient-egypt --resources resources
matrix-game-mlx verify --manifest releases/game-assets.json --scene ancient-egypt --resources resources
```

Unlike upstream, there is no `models.json` step: the model weights come from the MLX pack (see Weights).
Archives are never extracted.

## City racing on macOS

The upstream racing game (`../Matrix-Game-4/game/racing`, a native C++20/SDL2/OpenGL program) builds and runs
on macOS unchanged. You play or script a drive, the game records a take (clean reference video, alpha UI,
audio, inputs, telemetry), and `matrix-game-mlx render --take` turns the take into a stylized video. Online
play is described under "Online play on macOS". Game, finalize and compose commands run in `Matrix-Game-4/`
(the upstream checkout), `matrix-game-mlx` commands in `Matrix-Game-4-mlx/`.

### Dependencies

Homebrew packages (cmake, SDL2, pkg-config, FFmpeg development libraries; the C++20 compiler comes from the
Xcode command line tools):

```bash
brew install cmake pkgconf sdl2 ffmpeg
```

### Racing assets

The racing scene needs 13 racing files plus 7 shared wav files. Upstream's `download`/`verify` form applies,
with `matrix-game` renamed to `matrix-game-mlx` and `--resources` pointing at the upstream checkout's
`resources` directory, where the game looks (`resources/racing`):

```bash
matrix-game-mlx download --manifest releases/game-assets.json --repo-id Skywork/Matrix-Game-4.0 \
  --repo-type model --revision 4419accdb7bb7881234c5914de8cd915266e0325 --scene racing \
  --resources ../Matrix-Game-4/resources
matrix-game-mlx verify --manifest releases/game-assets.json --scene racing \
  --resources ../Matrix-Game-4/resources
```

If you keep the flags in a shell variable, use an array under zsh (`opts=(--manifest ...)` then
`"${opts[@]}"`): an unquoted string variable is not word-split there.

### Build

Download the assets above first: `build_racing.sh` runs ctest under `set -e`, and 4 of its asset tests fail
without `resources/racing/assets`. Verified on macOS with `sdl2` (sdl2-compat 2.32) and `ffmpeg` 9.0.2. Build from
`Matrix-Game-4/`:

```bash
bash scripts/build_racing.sh -DRACING_ASSET_ROOT="$PWD/resources/racing"
```

The binary is `game/racing/bin/CitySampleBBoxGame`; the tests look for it in the `Matrix-Game-4` checkout next to
`Matrix-Game-4-mlx` (`racing_game.game_binary()`), so in a git worktree build inside that worktree's
`Matrix-Game-4`. The build succeeded on the first try and the 9 ctest targets pass. No upstream patch was needed
(see "Upstream patches" below).

### Play

From `Matrix-Game-4/`:

```bash
game/racing/bin/CitySampleBBoxGame --data-dir resources/racing --hud              # free drive
game/racing/bin/CitySampleBBoxGame --data-dir resources/racing --race --hud       # time trial (default route)
game/racing/bin/CitySampleBBoxGame --data-dir resources/racing --pursuit --hud    # single-player pursuit
```

`--race --route downtown|avenue|skyway` picks the time-trial route.

W/S accelerate/reverse, A/D steer, Space applies the handbrake. P pauses a race, Enter resumes, and R
requests recovery in a time trial (pursuit recovery is disabled).

### Recording a take

Add `--fps 48 --physics-hz 480 --offline-record captures` (a directory of your choice) to any play command
and press F9 to start and stop recording; each recording becomes `captures/take-NNN`. The take holds the
clean reference video, an alpha UI track, audio, input events and per-frame telemetry. Do not feed the UI
track to the model.

For a reproducible take, the test suite replays an input script (`tests/data/racing/take_input_script.json`,
a JSON array of `{frame, key, down}` with SDL scancode names) in a hidden window. From `Matrix-Game-4/`:

```bash
game/racing/bin/CitySampleBBoxGame --data-dir resources/racing --hidden --race --route downtown \
  --fps 48 --physics-hz 480 --offline-record captures --offline-auto-record \
  --offline-input-script ../Matrix-Game-4-mlx/tests/data/racing/take_input_script.json --duration 48
```

Recording starts by itself at game frame 144 (after the 3 s countdown) and the scripted F9 at frame 864
stops it at exactly 720 frames (15 s at 48 FPS). `--duration` is a wall-clock safety net (the tests use
3 + 15 + 30 s).

Check a take without rendering (upstream's finalizer validates its own racing schema; do not use the shooter
48 FPS / 2x validator):

```bash
python game/racing/tools/finalize-offline-demo.py captures/take-001 --check-only
```

### Render

Racing takes must be rendered with `--scene racing` (a racing take's `session.json` carries no scene key, so only shooter takes can omit it):

```bash
matrix-game-mlx render --scene racing --take ../Matrix-Game-4/captures/take-001 --resources ../Matrix-Game-4/resources --preset realtime --output output/racing-take
matrix-game-mlx render --scene racing --take ../Matrix-Game-4/captures/take-001 --resources ../Matrix-Game-4/resources --preset quality --output output/racing-take-quality
```

Renders need the converted pack's `dit_racing.safetensors` (the Hub pack has it; a partial local cache may not).
Measured on an M2 Pro 32 GB with a 720-frame (15 s) take:

| Mode | Seconds per chunk | Peak memory | Total |
| --- | --- | --- | --- |
| realtime | 3.9 s | 15.0 GB | ~12 min |
| quality | 34.6 s | 24.4 GB | ~104 min |
| realtime, dynamic captions | 4.6 s | 26.3 GB | not timed |
| quality, dynamic captions | not run | needs more than 32 GB | not run |

Dynamic captions use `--prompt-mode dynamic --captions FILE` (see "Dynamic captions (racing)" above). The
JSONL has one record per chunk, `{"frame_start", "frame_end", "prompt"}` with `frame_end` inclusive and
zero-based: frame 0 alone, then consecutive groups of four, the last range possibly shorter. For a
720-frame take that is frame 0, 179 groups of four, then a final group of three (181 records). A complete example (181 records, one
per chunk of the 720-frame scripted take) is `tests/data/racing/take_captions.jsonl`:

```bash
matrix-game-mlx render --scene racing --take ../Matrix-Game-4/captures/take-001 --resources ../Matrix-Game-4/resources --preset realtime --prompt-mode dynamic \
  --captions captions.jsonl --output output/racing-take-dynamic
```

A plain RGB video does not supply engine captions. Real engine-derived captions need the engine, so they
come from `serve --prompt-mode dynamic` (see "Online play on macOS").

### Finalize and compose

Pack the rendered video with the take's audio and telemetry (upstream tool, from `Matrix-Game-4/`):

```bash
python game/racing/tools/finalize-offline-demo.py captures/take-001 \
  --rendered ../Matrix-Game-4-mlx/output/racing-take/rendered.mp4
```

`game/racing/tools/compose-demo.py` provides optional HUD relayout and input visualization (`--help` for
options). Both tools, and the video handling of the finalizer, need `ffmpeg` on the PATH.

### Online racing

`matrix-game-mlx serve --scene racing` (MediaMTX, browser, engine captions at runtime) is described under "Online play on macOS".

### Upstream patches

None. The upstream game code builds and runs on macOS as is. One upstream test fails with ffmpeg 9.0.2:
`game/racing/tools/test_demo_composition.py::test_ffmpeg_preserves_overlay_frame_indices` decodes 199 frames
where it expects 200 (an rgba to rgb24 rounding difference in that ffmpeg; the frame indices themselves are
correct). It is not patched.

## Online play on macOS

`matrix-game-mlx serve` is the port of upstream's online demo (`../Matrix-Game-4/docs/online.md`, and "Online" in
`docs/racing.md`): the native game renders the scene, the MLX model restyles each chunk, GStreamer encodes it and
MediaMTX serves it to the browser over WHEP, with upstream's web client (`matrix_game_mlx/web/`, byte-identical).
Same options, validations, routes and messages as upstream's `serve`, except the CUDA-only options (see "Known
deviations"). **It is not real time on a laptop**: the realtime preset generates about 1.1 frames per second and the
quality preset about 0.15 (table below), so a session is a slow-motion game. Commands run from `Matrix-Game-4-mlx/`
unless noted; `matrix-game-mlx serve` and `python3 -m matrix_game_mlx.cli serve` are the same entry point.

### Setup

```bash
brew install cmake pkgconf sdl2 ffmpeg mediamtx gstreamer pygobject3        # build tools, media stack
pip install 'aiohttp>=3.10,<4'                                              # or: pip install -e '.[serve]'
```

The GStreamer sidecar needs a Python with `gi`, NumPy and the GStreamer 1.0 typelibs; `serve` picks the first of
the current interpreter, `/opt/homebrew/bin/python3`, `python3` on the `PATH` that imports them, or uses
`STUDIO_WEBRTC_PYTHON`. Without any, it exits with the install command. GStreamer needs `x264enc` and
`rtspclientsink` (Homebrew's `gstreamer` includes both). Build the games from `Matrix-Game-4/` (the shooter
build needs no asset flag; the racing build is described under "City racing on macOS"), and fetch the scene
assets (the games and `serve` read `../Matrix-Game-4/resources`; use `--resources` for another place):

```bash
cd ../Matrix-Game-4 && bash scripts/build_game.sh && bash scripts/build_racing.sh -DRACING_ASSET_ROOT="$PWD/resources/racing"; cd -
for scene in castle ancient-egypt racing; do
  matrix-game-mlx download --manifest releases/game-assets.json --repo-id Skywork/Matrix-Game-4.0 --repo-type model \
    --revision 4419accdb7bb7881234c5914de8cd915266e0325 --scene $scene --resources ../Matrix-Game-4/resources
done
```

(The castle game does not start without its `download --scene castle` assets (`castle-map.bin`); racing assets must
exist before `build_racing.sh`, as above. `verify --scene castle|ancient-egypt|racing` passes on the machine used for the
measurements.)

### Run

Terminal 1, MediaMTX with the loopback-only configuration (no binary is downloaded for you):

```bash
mediamtx ../Matrix-Game-4/configs/mediamtx.yml
```

Terminal 2, one scene at a time (`--game-root` and `--resources` default to `game` and `resources` in the working
directory, so pass them when running from `Matrix-Game-4-mlx/`):

```bash
S="--game-root ../Matrix-Game-4/game --resources ../Matrix-Game-4/resources"
matrix-game-mlx serve --scene castle --preset realtime --fps 4 $S            # or --scene ancient-egypt
matrix-game-mlx serve --scene racing --preset realtime --fps 4 $S            # fixed caption
matrix-game-mlx serve --scene racing --prompt-mode dynamic --prompt-interval-chunks 1 --fps 4 $S   # see "Memory"
matrix-game-mlx serve --scene castle --preset quality --fps 4 $S             # Wan VAE, two denoising steps
matrix-game-mlx serve --scene castle --reference-only --fps 24 $S            # no model: game, media and browser only
matrix-game-mlx serve --scene castle --fps 4 --profile-chunks $S             # per-chunk timings in runtime/
```

(Under zsh use an array for `$S`, as for `opts` above.) Open `http://127.0.0.1:8025/`, press Play, click the video for pointer lock.
`--reference-only` is the no-model transport check (the browser's Mode menu then offers `reference`). Only one
session is active at a time; Stop before starting another. Ports: HTTP 8025, WHEP 8890, ICE/TCP 8190, internal RTSP
8555 (change both the server options and `configs/mediamtx.yml` together). This is for one trusted local user:
no accounts, TLS or authentication. For a remote browser, use upstream's tunnel (substitute your own values):

```bash
ssh -N -p SSH_PORT -L 8025:127.0.0.1:8025 -L 8890:127.0.0.1:8890 -L 8190:127.0.0.1:8190 USER@HOST
```

The browser's `window.castleDebug()` shows frame matching, presented frames and audio counters. `--profile-chunks`
writes `runtime/profile-<session>.json`; upstream's help text still says "CPU/CUDA timings", but on this port the
stage keys are `gpu_<stage>_ms` (`gpu_encode_ms`, `gpu_denoise_ms`, `gpu_clean_kv_ms`, `gpu_decode_ms`, ...).
`model_fps` counts generated frames over inference time (from the third chunk); `online_fps` also counts waiting
for the live game and is reported only after 25 chunks.

### Measured rates

Apple M2 Pro, 32 GB, macOS 27.0.1, MLX 0.31.1, one headless Chromium page playing (`tests/measure_serve.py`, one session per
row, alone on the machine, 32 chunks; chunk = 4 frames; times are `processing_ms` medians). "Footprint" is the summed
physical footprint (`top` MEM, which includes MLX's Metal buffers) of server, game, sidecar and MediaMTX, excluding
the browser (1.1 GB). `ps` RSS undercounts MLX by 5 to 14 GB, so it is not used. Swap is the growth of
`vm.swapusage` during the session.

| Configuration | `--fps` | Chunk | `model_fps` | `online_fps` | Footprint | Swap |
| --- | --- | --- | --- | --- | --- | --- |
| castle realtime | 40 | 3.75 s | 1.09 | 0.99 | 16.2 GB | 0 |
| castle realtime | 4 | 3.83 s | 1.07 | 0.97 | 16.2 GB | 0 |
| racing realtime, fixed caption | 24 | 3.77 s | 1.09 | 1.01 | 16.5 GB | 0 |
| racing realtime, fixed caption | 4 | 3.75 s | 1.09 | 1.01 | 16.6 GB | 0 |
| racing realtime, dynamic captions, no browser | 24 | 4.0 s, then 31 to 35 s | 0.34 | 0.14 | 25.5 GB | +3.8 GB |
| racing realtime, dynamic captions, browser | 24 | 5.4 s, then 14 to 34 s | 0.55 (21 chunks) | n/a | 25.5 GB | +3.1 GB |
| castle quality (6 chunks, no browser) | 4 | 27.8 s | 0.15 | n/a | 24.2 GB | +0.4 GB |

Time to the first encoded frame: realtime 3.6 to 7.8 s (13.5 s with dynamic captions), quality 12.4 s (13.3 s in
the end-to-end test). Realtime chunk medians are about 3.3 s over a 14-chunk session and 3.8 s over 32 (the KV window fills at chunk 19).
The `--fps` flag does not change the generation rate: the model sets the pace and the game waits for it, so
`--fps` sets how much game time one generated frame covers.

What a player sees (Chromium counters over the last 18 s, `window.castleDebug()`: frames presented by the browser
and game frames, per second): about 0.7 to 1.1 presented and 0.9 to 1.2 game frames per second for every realtime row,
whatever `--fps`: **a slideshow of about one image per second**, with the 4-frame chunks arriving in bursts. Only the
game clock differs. The 125 frames of a 32-chunk session (about two minutes) cover 3.1 s of game time at `--fps 40`, 5.2 s at 24 and 31 s
at 4: the world runs about 40, 24 and 4 times slower than real time. Image quality also differs, from one screenshot of each (castle 14 chunks, racing 14 chunks, a
keyboard-driven drive): at 40 fps the castle had turned into a brown smear after 53 frames (plausibly because
the model sees almost no change between frames at 1/40 s steps; not tested further); at 4 fps the castle stayed coherent, with a wider spacing than the training
data but readable. Racing at 24 fps stayed sharp (the car had barely left the start line); at 4 fps the racing
frames show visible texture smearing on walls, and the car had driven 10 s of the route (its HUD timer). These are single sessions, not
a quality metric.

**Recommendation: `--fps 4` for both scenes and both presets**, which is the lowest rate that keeps the game
playable (a time trial then lasts minutes instead of tens of minutes), with the caveat that the image is rougher
than at upstream's 24 to 40 fps (`--fps 8`, untested here, trades game speed for a tighter frame spacing; 8 divides
120 for the shooters). The quality preset delivers one chunk per 28 s, so even at `--fps 4` the game runs about 28
times slower than real time: usable to look at the picture, not to play. Do not read `--fps` as a
speed-up (upstream says the same).

### Memory

Realtime fixed-caption sessions peak at 16.2 to 16.6 GB footprint, quality at 24.2 GB: both fit in 32 GB with the
browser. **Racing with dynamic captions does not fit sustainably on a 32 GB Mac.** T5 stays resident (no offload),
the footprint reaches 25.5 GB (24.5 GB in the end-to-end test, which stops after 9 frames), and the machine swaps:
swap grew by 3.8 GB without a browser and 3.1 GB with one, later chunks took 31 to 35 s instead of 4 s,
and with the browser the session was lost ("Browser disconnected": the page stopped answering the server's
15 s watchdog). These runs were at `--fps 24`; `--fps` changes neither the generation rate nor the memory, so the
same holds at `--fps 4`. Use `--prompt-mode fixed` (the default) on 32 GB; dynamic captions need a Mac with more memory.
(The swap file already held 4 to 5 GB from earlier work when these runs started, so a freshly booted Mac
may do better; that was not tested.)

### Quality preset in a browser

A browser keeps an idle `/api/start` request open for a limited time (WebKit about 60 s), while the server waits for
the first encoded frame. On this Mac quality reached it in 12 to 14 s, well inside that, but a slower Mac may not.
`serve` allows 300 s for the Wan-VAE presets (45 s for `realtime`, upstream's value). The 15 s silence watchdog
still applies: the browser sends controls continuously, but a client that does not (a script) must send one at
least every 15 s, as `tests/test_serve_e2e.py` does.

### macOS port changes

Upstream's server modules are copied with `matrix_game.` renamed to `matrix_game_mlx.`; the changes in
`server/app.py` each carry a `# macOS port:` comment:

- The sidecar interpreter is found by `media_env.default_sidecar_python()` (upstream's `serve.sh` uses `/usr/bin/python3`, which has no GStreamer here), and `serve` applies `serve.sh`'s environment defaults itself
  (`STUDIO_WEBRTC_RTSP_PORT=8555`, `STUDIO_MEDIA_MAXRATE_KBPS=2500`, `STUDIO_WEBRTC_FRAME_PACING=clock`). A machine without a
  GI-enabled Python is a usage error with the install command, not a traceback. `OMP_NUM_THREADS=8` and
  `OPENBLAS_NUM_THREADS=1` are set by `serve` after NumPy and MLX are imported, so unlike upstream's `serve.sh` they reach
  only child processes (the game and the sidecar), not the server itself.
- `validate_renderer` also rejects "Apple Software Renderer" (Apple's software OpenGL) unless `--allow-software-rendering`.
- The first-frame deadline is 45 s for `--preset realtime` (upstream's value) and 300 s for the Wan-VAE presets, which need
  longer on Apple Silicon; the error text keeps upstream's shape.
- `Runtime.__init__` no longer resets `attention_backend` (the MLX loader sets `mlx-sdpa`).
- No torch: the session report drops `profile_torch_threads`; the stream hands an owned `uint8` array to `on_rgb` (no `.numpy()`); profile keys are
  `gpu_<stage>_ms` instead of `cuda_<stage>_ms`.
- CUDA-only model options keep the treatment `render` gives them (for example `--preset rtx5090` exits with "--preset rtx5090 is CUDA-only"); the realtime DiT runtime is `native`.
- The web client is copied byte-for-byte, and `matrix_game_mlx/web/mediamtx-reader.LICENSE` ships in the wheel (`license-files`).
- Game patches: none. Both `CastleReferenceServer` and `CitySampleReferenceServer` create their hidden SDL/OpenGL window from a
  subprocess on macOS unchanged.

### Online tests

Online tests need MediaMTX and a GI-enabled Python (`needs_media`), the built games (`needs_castle_game`, `needs_racing_server`), the
weights (`needs_weights`) and, for the browser tests, Playwright (`needs_playwright`). Playwright lives outside the repo:

```bash
mkdir -p ~/.cache/matrix-game-4-mlx/playwright && cd ~/.cache/matrix-game-4-mlx/playwright && npm init -y && npm install playwright && npx playwright install chromium webkit
```

Environment: `MG4_RESOURCES` (game assets, default `../Matrix-Game-4/resources`), `MATRIX_GAME_4_MLX_WEIGHTS_DIR`,
`MG4_FIXTURES_DIR`, `PLAYWRIGHT_MODULE` (default the path above). `tests/test_serve_app.py`, `test_display.py`,
`test_media_environment.py` and `test_cli.py` need none of them beyond Python; `test_online_games.py`, `test_transport.py`, `test_serve_e2e.py` and
`test_browser_upstream.py` run the real stack; `slow` marks the quality session (several minutes). The browser tests run upstream's
`tests/browser/*.cjs` unmodified: Chromium uses Playwright's full Chromium (`CHROME_PATH`, set by the tests), because
the default headless shell rejects pointer lock; WebKit runs with `DISPLAY_ONLY=1`, since Playwright's WebKit never grants
it (Safari itself is not automated; do upstream's manual Safari check). If WebKit receives no video on a fresh machine, check System Settings, Privacy & Security, Local Network for the
browser: after a new install WebKit got no WHEP video for about an hour, then worked without any change. Measure with
`python3 -m tests.measure_serve --scene castle --fps 4 --chunks 32 [--browser] [--prompt-mode dynamic]`.

## Weights

The weight pack is produced by the [mlx-forge](https://github.com/dgrauet/mlx-forge) conversion of
`Skywork/Matrix-Game-4.0` and published as [`dgrauet/matrix-game-4.0-mlx`](https://huggingface.co/dgrauet/matrix-game-4.0-mlx)
(bf16). It is downloaded automatically on first use, pinned to a fixed revision (`PACK_REVISION` in
`inference/loader.py`), and only the selected scene's DiT (~10 GB) plus `taehv.safetensors` for the realtime preset
are fetched. `MATRIX_GAME_4_MLX_WEIGHTS_DIR` points at a local pack directory instead (no download). Tests marked
`needs_weights` use the local pack and are skipped when that variable is unset.

```text
matrix-game-4.0-mlx/
  config.json                    # DiT config, codecs, per-scene entries (component, inference, sink_image)
  split_model.json               # pinned shared files: repo, revision, sha256
  dit_ancient_egypt.safetensors  # also dit_castle, dit_racing
  taehv.safetensors              # realtime codec (TAEHV)
  fps-egypt/                     # inference.json, sink.png (also fps-castle, driving_game)
  licenses/                      # third-party licenses (TAEHV)
```

The umT5 encoder, the Wan 2.2 VAE and the `google/umt5-xxl` tokenizer are the same files as Matrix-Game 3.0.
They are not duplicated: `split_model.json` pins them to `dgrauet/matrix-game-3.0-mlx` at a fixed revision,
and every file is sha256-checked after download. `MATRIX_GAME_4_MLX_SHARED_DIR` points at a local copy of
these files (still sha256-checked).

## Performance

Apple M2 Pro, 32 GB, MLX 0.31.1, ancient-egypt, quality preset, frames at 480x864. A chunk is one latent
frame (four video frames, one for the first chunk), with two denoising calls and one cache-update call.

Long run: 27 chunks (105 frames) through `CastleStream.push` with real weights, timed per chunk (VAE encode, the
three DiT calls and VAE decode; no video I/O; model loading ~23 s excluded). The KV window (21 latents, sink
included) fills at chunk 19, and chunks 20 to 26 run with rolling eviction.

| Chunks | s/chunk mean | min | max |
| --- | --- | --- | --- |
| 0 (one frame) | 12.0 | | |
| 1 (first four-frame chunk) | 30.2 | | |
| 2 to 26, steady state (25 chunks) | 33.2 | 26.9 | 42.4 |

Memory over the same run: peak 22.74 GiB, active 17.24 GiB flat from chunk 2 through the window fill and the
eviction, MLX buffer cache (`mx.get_cache_memory`) 6.14 GiB. At the steady-state mean, a 10 s game take (recorded at 48 FPS:
480 frames, 121 chunks) renders in about 67 min, roughly 8.3 s per video frame, excluding model load and video I/O.

Short CLI runs (`render.json`, 1280x720 24 FPS input, 6 frames = 3 chunks): these averages include the
one-frame chunk 0, so they understate the steady-state cost per chunk. `seconds` excludes model loading;
`seconds_per_chunk` is `seconds / chunks` and includes video decode, VAE encode/decode and MP4 encoding.

| Short run (3 chunks) | Frames | Chunks | seconds | s/chunk | Peak memory |
| --- | --- | --- | --- | --- | --- |
| Before Task 15 numeric fixes (float64 islands, fused fp32-opmath activations) | 6 | 3 | 77.5 | 25.8 | 23.9 GB |
| After them (float64 islands on the CPU stream) | 6 | 3 | 73.6 | 24.5 | 24.4 GB |

Kernel dtype audit (smeltr, 5-frame render; shares come from smeltr's sampled per-op GPU timing, which was
partly degraded during the run, so they are approximate): all DiT GEMMs and attention kernels are bf16
(`steel_gemm_fused_*_bfloat16_bfloat16`, `steel_attention_bfloat16_*`), and the VAE convolutions are bf16
(the Conv kind is 42 % of GPU time, its top kernel `implicit_gemm_conv_3d_bfloat16` alone 40 %). No float32
GEMM or attention kernel remains. The float32 elementwise and reduce kernels are the norm statistics and the
torch-opmath activations listed under Numerics. The float64 CPU-stream work costs about 0.54 s per DiT forward (RoPE ~0.49 s over 30 layers,
timestep MLP ~0.05 s, flow-to-x0 negligible), so 1.1 to 1.6 s per chunk (3 to 5 % of the 33.2 s steady-state chunk).

Realtime preset (97-frame pans, 24 fps, seed 0, same machine and MLX version, peak 14.96 GB), against the
quality preset on the same inputs (castle 37.7 s/chunk, racing 27.3 s/chunk, peak 24.4 GB):

| Scene | realtime s/chunk |
| --- | --- |
| ancient-egypt | 3.57 |
| castle | 3.58 |
| racing | 3.51 |

That is about 1.1 processed frames per second. Outputs are in `output/{egypt,castle,racing}-pan-realtime/`.

## Numerics

Upstream's float64 islands (RoPE complex product and tables, flow-to-x0, timestep sinusoid) run in float64,
on the MLX CPU stream where MLX has no GPU float64. bf16 rounding points follow torch's opmath rather than
MLX's per-op defaults: SiLU / tanh-GELU and the VAE `RMS_norm` compute in fp32 and round once, and the
timestep-MLP linears (one row per timestep entry, B x F) accumulate in float64, bias included, and round once
(MLX's single-row bf16 addmm rounds the product before adding the bias, and fp32 summation order flips a few
outputs at any row count). The scheduler's sigma grid reproduces `torch.linspace` bitwise. The umT5 text
encoder runs in bf16, as upstream's loader runs it (`WanTextEncoder(...).to(device, torch.bfloat16)`),
attention bias included (`x.new_zeros`).

TAEHV (realtime codec): MLX bf16 `conv2d` on Metal loses precision on some kernel paths (decoder.8, the
256 to 128 3x3 conv: 3.06e-2 rel L2 vs fp64, where torch CPU bf16 gets 1.66e-3). All TAEHV convs therefore
accumulate in float32 and round once to bf16 (torch semantics), which brings that conv to 1.3e-4 vs torch bf16.
Parameters stay bf16.

## Scenes

All three scenes (`--scene {ancient-egypt,castle,racing}`) are parity-validated in the quality preset against
upstream PyTorch with the same methodology (see below): per-block and per-call teacher-forced gates at 1.5x the
torch kernel-variant floor, and a strict end-to-end check on chunk 0 (racing: chunks 0 and 1, chunk 1 pred gated at 1.5x the torch floor).

| Scene | Sink policy | End-to-end result |
| --- | --- | --- |
| ancient-egypt | fixed sink | chunk 0 pred rel L2 1.71e-2, decoded 44.9 dB |
| castle | fixed sink | chunk 0 pred 1.41e-2, 43.7 dB; blocks at most 0.56x gate. Chunk 1 (reported) 0.455 vs torch-vs-torch 0.478 |
| racing | first_frame sink, absolute sink RoPE | chunk 0 = sink latent, bitwise, 56.4 dB; chunk 1 pred 2.47e-2 (gate 4.6e-2 = 1.5x torch floor 3.07e-2), 45.8 dB; blocks at most 0.56x gate |

- Castle amplifies bf16 rounding faster than ancient-egypt: at chunk 1 MLX stays within torch's own spread.
- Racing's chunk 0 is the sink latent itself (no denoising). Its chunk 1 pred is gated at 1.5x the torch floor
  because torch's own spread exceeds 2e-2 there. Chunk 2 (reported) is 4.5e-2, 39.6 dB. The `--long` chunk 22,
  past the 21-latent window (eviction with absolute RoPE), passes its per-call gates.

### Realtime preset

The torch reference is the bf16 CPU emulation (`cpu_bf16_emulation`: fp32 compute with bf16-rounded outputs, see the parity methodology section); the decoder.8 regression test compares against native torch bf16 conv.

Gates use the `<scene>_realtime` fixtures (torch bf16 CPU reference, upstream native DiT runtime). TAEHV gates
are MLX rel L2 vs torch, with tolerance ceil_1sig(1.5 x the max over the torch variants, fp32 CPU and bf16 MPS):

| Scene | Sink | Control | Decoded | Decoded PSNR (torch variants) |
| --- | --- | --- | --- | --- |
| ancient-egypt | 8.3e-3 (tol 2e-2) | 7.5-8.3e-3 (tol 2e-2) | 4.1-5.3e-3 (tol 1e-2/2e-2) | 56.5-57.6 dB (53.5-54.5 dB) |
| castle | 7.6e-3 | 6.9-7.5e-3 | 3.5-5.4e-3 | 55.1-56.9 dB (51.5-52.5 dB) |
| racing | 8.0e-3 | 7.4-8.1e-3 | 3.4-5.0e-3 | 56.6-59.6 dB (53.7-56.2 dB) |

End to end (gated chunk 0; racing chunks 0 and 1; pred gate = max(2e-2, 1.5 x the torch floor of the chunk)):

| Scene | End-to-end result |
| --- | --- |
| ancient-egypt | chunk 0: control 8.3e-3, pred 2.75e-2 (gate 3.5e-2; torch floor 2.31e-2), KV < 6e-3, decoded 41.8 dB |
| castle | chunk 0: control 7.5e-3, pred 2.03e-2 (gate 2.96e-2; torch floor 1.97e-2), decoded 42.7 dB |
| racing | chunk 0: pred = sink bitwise, decoded 59.6 dB; chunk 1: pred 1.51e-2 (gate 2.9e-2), decoded 51.1 dB |

The pred gate follows the torch floor because, with one denoising step, torch against its own SDPA variant
already exceeds 2e-2 (egypt chunk 0: 2.31e-2). Injecting torch's exact control into the MLX stream gives 2.80e-2,
so the gap is the DiT's alone, which is preset-independent and gated per call and per block by the quality fixtures.

## Known deviations from upstream

- float64 runs on the MLX CPU stream (RoPE, flow-to-x0, timestep MLP accumulation) because MLX has no float64
  on the GPU; the values are upstream's, the device is not.
- KV-cache end indices are Python ints instead of 1-element tensors.
- `CastleStream.push` has a test-only `renoise` argument.
- Only the inference path (`kv_cache is not None`) is ported; training branches raise `NotImplementedError`.
- Realtime DiT runtime is upstream's `native`. Upstream's realtime default is the CUDA-only `ring-fused`, out of
  scope (upstream logs that ring changes the reduction order and that native is the baseline).
- No CUDA VAE graph (`vae_graph`): it only captures the same TAEHV ops.
- TAEHV convs accumulate in float32 and round once to bf16 (see Numerics).
- `serve`: the macOS port changes listed under "Online play on macOS" (sidecar interpreter, Apple software renderer, startup deadline, `gpu_` timing keys, no torch).

## Parity methodology

The upstream PyTorch code in `../Matrix-Game-4/` is the oracle. Unit and module tests compare against it
directly (single module fp32 `max_abs < 1e-3`, block `< 5e-3`). Real-weight parity
(`tests/parity/test_real_weights.py`) compares the MLX stream with torch fixtures on three levels:

- **Per call, teacher forced.** At each chunk boundary, torch's exact cache state and inputs are injected and
  one MLX generator call runs. Its pred and the layer 0 / 29 KV it writes are compared with torch. Each gate is
  1.5x the largest rel L2 between torch and three torch kernel variants of the same call (native bf16 SDPA,
  native CPU bf16 without emulation, float64 accumulation), so MLX must stay within the noise torch shows
  against itself. The rest of the cache (history, sink, rolled window), the end indices and `k_sink_raw`
  must match exactly (except the `k_sink_raw` rows that racing's chunk-0 clean pass writes: gated at 1.5x the floor
  on layers 0/29, only checked to have changed on the others). A `--long` chunk 22 runs past the 21-frame window (rolling eviction, and re-RoPE of the
  fixed sink or eviction with absolute RoPE for racing).
- **Per block.** For the first denoising call of chunk 0 (chunk 1 for racing, whose chunk 0 has none), each of the 30 blocks gets torch's exact input, `e0`,
  context and sink KV; its output must be within 1.5x the largest torch-variant rel L2 on that block. This
  catches a systematic divergence that per-call noise would hide.
- **End to end, chunk 0 strict.** The full MLX stream runs from the same inputs. Chunk 0 is gated with the
  spec thresholds (control, pred and KV rel L2 `< 2e-2`, decoded PSNR `> 35 dB`). The autoregressive stream
  amplifies bf16 rounding chunk over chunk (torch against itself reaches ~0.3 rel L2 by chunk 2), so chunks 1
  and 2 are reported only. Racing's chunk 0 is the sink latent, so its chunk 1 is gated too, against 1.5x the
  torch floor of that chunk.

Fixtures are generated with the real upstream model. The prompt embeddings come from the MLX umT5 and are
stored in the fixtures, so both sides see the same context; the text encoder has its own real-weight test
against upstream's bf16 umT5 (`tests/test_text_encoder.py`, gate 1.5x the torch-variant floor).

```bash
cd Matrix-Game-4-mlx
MATRIX_GAME_4_MLX_WEIGHTS_DIR=... python -m tests.parity.gen_fixtures --scene castle --device auto [--long]
MATRIX_GAME_4_MLX_WEIGHTS_DIR=... python -m tests.parity.gen_fixtures --scene castle --preset realtime
```

| Flag | Effect |
| --- | --- |
| `--scene {ancient-egypt,castle,racing}` | Scene to generate (default ancient-egypt) |
| `--device {auto,mps,cpu}` | Torch device; `auto` tries MPS first. The DiT falls back to CPU (MPS has no float64), the VAE stays on MPS |
| `--chunks N` | Number of chunks (default 3) |
| `--long` | Also run 25 chunks (pred and decoded) and the chunk-22 per-call fixture |
| `--floor-only` | Regenerate only the torch floor chunks from the saved inputs |
| `--calls-only` | Regenerate only the per-call fixtures |
| `--call-floors-only` | Re-measure only the torch kernel-variant floors of the per-call fixtures |
| `--preset {quality,realtime}` | Preset to generate (default quality); realtime writes `<scene>_realtime` |
| `--taehv-only` | Realtime only: regenerate only the TAEHV fixtures (~20 s) |
| `--blocks-only` | Regenerate only the per-block fixture (chunk 0 call 0; chunk 1 call 0 for racing) and its floors |

They are written to `$MG4_FIXTURES_DIR/<scene>_quality/` or `<scene>_realtime/` (`egypt_quality`, `castle_quality`, `racing_quality`, `egypt_realtime`, `castle_realtime`, `racing_realtime`, `racing_take_realtime`, `racing_take_quality`, `racing_online_dynamic`, `castle_online_realtime`;
default `~/.cache/matrix-game-4-mlx/fixtures`): about 1.5 GB per scene without `--long` (ancient-egypt 1.7 GB)
and about 6 GB with `--long` (ancient-egypt 5.7 GiB, racing 5.2 GiB) (the chunk-22 per-call fixture alone is 3.8 GB). A full castle set takes about 1 h to
generate, a racing set with `--long` about 1 h 15. For racing the torch DiT runs on the CPU (no sink-priming
smoke on MPS is possible).
Recorded in `meta.json` for ancient-egypt (M2 Pro, VAE on MPS, DiT on CPU): the 3-chunk run took 219 s
(`run_seconds`), the 25-chunk `--long` run 2668 s (about 44 min, `long_seconds`), plus 335 s for the floor
run and 1267 s for the per-call floors. Native torch bf16 GEMMs on the CPU run at
about 1 GFLOPS on Apple Silicon (a 5B forward would take about an hour), so the generator emulates them:
bf16 linear, conv and SDPA calls on the CPU compute in fp32 and round the output to bf16 (fp32-accumulate
kernel semantics). Every other op, including the float64 islands, runs unchanged, and `meta.json` records the
emulation and the device of each part.

Variant fixtures (the CLI's new options) are generated with `--variant NAME`, and the caption embeddings with
`--captions-only`; `--print-variant-commands` lists every command:

| Command | Validates (`tests/parity/`) |
| --- | --- |
| `python -m tests.parity.gen_fixtures --captions-only` | umT5 caption embeddings vs upstream torch (`test_real_weights_captions.py`) |
| `--variant racing_switch_realtime` / `racing_switch_quality` | Dynamic prompt switch (prompt A to prompt B) (`test_real_weights_variants.py`) |
| `--variant racing_take_realtime` / `racing_take_quality` | The first 9 frames of a real recorded racing take (scene sink image, fixed prompt); needs the take (`MG4_RACING_TAKE`) (`test_real_weights_variants.py`) |
| `--variant racing_online_dynamic` / `castle_online_realtime` | The first 9 frames of a real `serve` session capture (`python -m tests.parity.capture_online --scene racing\|castle`, into `$MG4_RESOURCES/online-captures/<scene>-online/`); racing replays the engine captions with the upstream dynamic-prompt rule (`set_prompt` on every caption change) (`test_real_weights_variants.py`) |
| `--variant castle_nosink_realtime` / `racing_nosink_realtime` | `--no-sink-image` |
| `--variant castle_taehv_2step` | TAEHV with a custom 2-step schedule |
| `--variant egypt_wan_custom_first` | Wan VAE with custom `--first-steps` |

A realtime fixture set takes about 10 min per scene and is about 200 MB; `--taehv-only` (~20 s) and `--floor-only` are
partial modes.

## Tests

Run with a Python that has the MLX/torch stack (the development machine uses Homebrew Python 3.14):

```bash
cd Matrix-Game-4-mlx
python -m pytest tests -v
```

The fast suite (no weights, no fixtures) is `python -m pytest -m "not needs_weights and not needs_fixtures"`;
it also runs the racing game (see "Racing tests"). Add `and not needs_racing_game` to skip the game.
`tests/test_cli_smoke.py` (`needs_weights`) renders a 9-frame 864x480 racing clip in realtime dynamic mode through the CLI
and checks `peak_memory_gb < 30`.

### Online tests

`tests/test_serve_app.py`, `test_display.py`, `test_media_environment.py`, `test_cli.py` and the web-asset check run anywhere. The
real-stack tests (`test_online_games.py`, `test_transport.py`, `test_serve_e2e.py`, `test_browser_upstream.py`) carry the
markers `needs_media`, `needs_castle_game`, `needs_racing_server`, `needs_weights`, `needs_playwright` and `slow` and skip with the
install or build command when an input is missing; setup and environment variables (`MG4_RESOURCES`, `PLAYWRIGHT_MODULE`) are
under "Online play on macOS". With every input present, `python3 -m pytest -q -rs` skips nothing.

### Racing tests

Tests marked `needs_racing_game` run the built upstream game (`Matrix-Game-4/game/racing/bin/CitySampleBBoxGame`)
and are skipped, with the download and build commands, when the binary or the assets are missing. They read
two environment variables: `MG4_RESOURCES` (default `Matrix-Game-4/resources`, where the game looks in
`racing/`) and `MG4_RACING_TAKE` (default `$MG4_RESOURCES/racing/captures/take-001`), the recorded take used by
the render tests and the `racing_take_*` fixtures. The fast suite with the game enabled runs 5 headless smoke
runs and 3 scripted recordings of about 48 s each; add `and not needs_racing_game` to the `-m` expression to
leave them out.

The render tests and `gen_fixtures --variant racing_take_*` need that take to exist first (they skip or fail
with a pointer to this section). Record the canonical take into the default location with
(`record_scripted_take`; from `Matrix-Game-4/`, with `MG4_RESOURCES` set or defaulting to `resources`):

```bash
game/racing/bin/CitySampleBBoxGame --data-dir "${MG4_RESOURCES:-resources}/racing" --hidden --race --route downtown \
  --fps 48 --physics-hz 480 --offline-record "${MG4_RESOURCES:-resources}/racing/captures" --offline-auto-record \
  --offline-input-script ../Matrix-Game-4-mlx/tests/data/racing/take_input_script.json --duration 48
```

On an empty `captures` directory this creates `take-001`. If you set `MG4_RACING_TAKE` to another take
directory, record there instead.

The real-weight parity suite runs once per scene whose fixtures exist (a missing or partial set is skipped with
its `gen_fixtures` command) and loads one DiT at a time. Install the `parity` extra for the upstream oracle. Tests marked `needs_fixtures` read the torch reference
fixtures from `MG4_FIXTURES_DIR`; tests marked `needs_weights` need `MATRIX_GAME_4_MLX_WEIGHTS_DIR`.
