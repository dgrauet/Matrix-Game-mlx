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
| Apple fast path | `--dit-runtime ring-fused` on MLX (GPU RoPE, ring KV, fused kernels, compiled steady graph), `--vae-graph`, `--t5-runtime graph` | Done: every parity slice passes on both runtimes with thresholds unchanged; castle realtime `serve` 0.95 -> 1.13 `online_fps` (see "Apple fast path") |
| Apple presets | Port-only `--preset apple` (float16 DiT) and `--preset apple-fast` (float16, 11-latent attention window) for `render` and `serve` | Done: a priori fixture gates pass on the three scenes (not upstream-exact); castle `serve` `online_fps` 1.13 (realtime, `ring-fused`) -> 1.25 (`apple`) and 1.43 (`apple-fast`) (see "Apple presets") |

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
| `--preset {apple,apple-fast}` | Port-only speed presets: the realtime base (TAEHV, steps `1000`, `--vae-graph` on) with `--dit-runtime` defaulting to the preset's name, as upstream resolves `rtx5090`. They leave upstream exactness (float16 DiT; `apple-fast` also an 11-latent attention window); see "Apple presets" |
| `--vae-backend {wan,taehv}` | Supported (default from the preset: `taehv` for realtime, `wan` otherwise) |
| `--dit-runtime {native,ring-fused}` | Supported on every preset; default from the preset as upstream (`ring-fused` for realtime, the preset's name for `apple` / `apple-fast`, `native` otherwise). `ring-fused` is the Apple fast path (ring KV cache, GPU RoPE kernel, fused block-glue kernels); see Known deviations |
| `--dit-runtime {apple,apple-fast}` | Port-only: `ring-fused` with the DiT in float16 (`apple-fast`: self-attention window of 11 latents instead of 21, sink and current latent included). Accepted on every preset, as upstream accepts `--dit-runtime rtx5090`; the default only under `--preset apple` / `apple-fast`. See "Apple presets" and Known deviations |
| `--dit-runtime` other values (`kernels`, `cudagraph`, `ring`, `ring-graph`, `rtx5090`, `rtx5090-sage`) | Rejected: CUDA-only; the MLX port runs `native`, `ring-fused`, `apple` or `apple-fast` |
| `--steps`, `--first-steps` | Supported (comma-separated timesteps, strictly decreasing from 1000) |
| `--vae-graph`, `--no-vae-graph` | Supported; default on for realtime + TAEHV, as upstream (`VAE graph requires TAEHV` otherwise). The streaming TAEHV encoder/decoder run as `mx.compile`d graphs instead of CUDA graphs, bitwise eager (see Known deviations) |
| `--prompt-mode {fixed,dynamic}`, `--prompt`, `--prompt-interval-chunks` | Supported; dynamic is racing only (see below) |
| `--captions FILE` | Supported, dynamic mode only (required there) |
| `--t5-runtime {native,graph}` | Supported (default `graph`, as upstream); `graph` runs the dynamic-caption encoder as one `mx.compile`d graph instead of a CUDA graph, bitwise `native` (see Known deviations) |
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
  --repo-type model --revision 2c53100457da7abcf0e966f75637c644d3423a63 --scene ancient-egypt --resources resources
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
  --repo-type model --revision 2c53100457da7abcf0e966f75637c644d3423a63 --scene racing \
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
deviations"). **It is not real time on a laptop**: the realtime preset generates about 1.2 frames per second (1.1 with
`--dit-runtime native`), the port-only `--preset apple` and `apple-fast` about 1.4 and 1.5 (see "Apple
presets"), and the quality preset about 0.15 (table below), so a session is a slow-motion game. Commands run from `Matrix-Game-4-mlx/`
unless noted; `matrix-game-mlx serve` and `python3 -m matrix_game_mlx.cli serve` are the same entry point.

### Setup

```bash
brew install cmake pkgconf sdl2 ffmpeg mediamtx gstreamer pygobject3        # build tools, media stack
pip install 'aiohttp>=3.10,<4'                                              # or: pip install -e '.[serve]'
```

The GStreamer sidecar needs a Python with `gi`, NumPy and the GStreamer 1.0 typelibs; `serve` picks the first of
the current interpreter, `/opt/homebrew/bin/python3`, `python3` on the `PATH` that imports them, or uses
`STUDIO_WEBRTC_PYTHON`. Without any, it exits with the install command. GStreamer needs `x264enc` and
`rtspclientsink` (Homebrew's `gstreamer` includes both). Fetch the scene assets first (the games and `serve` read
`../Matrix-Game-4/resources`; use `--resources` for another place), then build the games from `Matrix-Game-4/` (the
shooter build needs no asset flag; the racing build is described under "City racing on macOS"):

```bash
for scene in castle ancient-egypt racing; do
  matrix-game-mlx download --manifest releases/game-assets.json --repo-id Skywork/Matrix-Game-4.0 --repo-type model \
    --revision 2c53100457da7abcf0e966f75637c644d3423a63 --scene $scene --resources ../Matrix-Game-4/resources
done
cd ../Matrix-Game-4 && bash scripts/build_game.sh && bash scripts/build_racing.sh -DRACING_ASSET_ROOT="$PWD/resources/racing"; cd -
```

(Download first, then build: the builds run ctest under `set -e`; `build_game.sh`'s tests need the castle assets
(`castle-map.bin`) and `build_racing.sh`'s asset tests need the racing assets. `verify --scene castle|ancient-egypt|racing`
passes on the machine used for the measurements.)

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
matrix-game-mlx serve --scene castle --preset apple-fast --fps 4 $S          # fp16 DiT, window 11 ("Apple presets")
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
With the stream pipeline, `background_prepare_tensor_ms` measures building the lazy MLX graph on the reference thread; the
actual conversion runs later, in the producer's `encode`, so it is not comparable to upstream's figure.

### Measured rates

Apple M2 Pro, 32 GB, macOS 27.0.1, MLX 0.31.1, one headless Chromium page playing (`tests/measure_serve.py`, one session per
row, alone on the machine, 32 chunks; chunk = 4 frames; times are `processing_ms` medians). "Footprint" is the summed
physical footprint (`top` MEM, which includes MLX's Metal buffers) of server, game, sidecar and MediaMTX, excluding
the browser (1.1 GB). `ps` RSS undercounts MLX by 5 to 14 GB, so it is not used. Swap is the growth of
`vm.swapusage` during the session.

| Configuration | DiT runtime | `--fps` | Chunk | `model_fps` | `online_fps` | Footprint | Swap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| castle realtime | `native` | 40 | 3.75 s | 1.09 | 0.99 | 16.2 GB | 0 |
| castle realtime | `native` | 4 | 3.83 s | 1.07 | 0.97 | 16.2 GB | 0 |
| castle realtime, no browser | `native` | 4 | 3.89 s | 1.05 | 0.95 | 16.2 GB | 0 |
| **castle realtime** | **`ring-fused`** | 4 | **3.42 s** | **1.20** | **1.10** | 16.2 GB | 0 |
| **castle realtime, no browser** | **`ring-fused`** | 4 | **3.29 s** | **1.24** | **1.13** | 16.2 GB | 0 |
| castle `--preset apple`, no browser | `apple` | 4 | 2.98 s | 1.36 | 1.25 | 16.2 GB | 0 |
| castle `--preset apple-fast`, no browser | `apple-fast` | 4 | 2.78 s | 1.46 | 1.43 | 15.2 GB | 0 |
| racing realtime, fixed caption | `native` | 24 | 3.77 s | 1.09 | 1.01 | 16.5 GB | 0 |
| racing realtime, fixed caption | `native` | 4 | 3.75 s | 1.09 | 1.01 | 16.6 GB | 0 |
| **racing realtime, fixed caption, no browser** | **`ring-fused`** | 4 | **3.35 s** | **1.24** | **1.15** | 16.6 GB | 0 |
| racing `--preset apple`, fixed caption, no browser | `apple` | 4 | 2.98 s | 1.38 | 1.28 | 16.6 GB | 0 |
| racing `--preset apple-fast`, fixed caption, no browser | `apple-fast` | 4 | 2.72 s | 1.49 | 1.46 | 14.6 GB | 0 |
| racing realtime, dynamic captions, no browser | `native` | 24 | 4.0 s, then 31 to 35 s | 0.34 | 0.14 | 25.5 GB | +3.8 GB |
| racing realtime, dynamic captions, browser | `native` | 24 | 5.4 s, then 14 to 34 s | 0.55 (21 chunks) | n/a | 25.5 GB | +3.1 GB |
| racing realtime, dynamic captions, no browser | `ring-fused` | 4 | 17.6 s median, max 54.7 s | 0.19 | 0.10 | 25.5 GB | +3.8 GB |
| castle quality (6 chunks, no browser) | `native` | 4 | 27.8 s | 0.15 | n/a | 24.2 GB | +0.4 GB |
| castle quality (6 chunks, no browser) | `ring-fused` | 4 | 28.4 s | 0.14 | n/a | 24.2 GB | +2.3 GB |

`ring-fused` has been the realtime default since the Apple fast path (see "Apple fast path"). The `apple` and
`apple-fast` rows are the port-only presets (float16 DiT, `apple-fast` with an 11-latent window; see "Apple presets"),
single sessions on macOS 27.2, same machine and MLX; their chunk max is 3.33 s / 2.82 s (castle) and 3.24 s /
2.76 s (racing), and their first encoded frame came after 8.5 s / 7.2 s (castle) and 6.5 s / 5.6 s (racing).
`apple-fast`'s KV window fills 10 chunks earlier than the 21-latent one (castle: steady from chunk 11 instead of
21), so its chunk max stays within 1.5 % of its median. The `native` rows were
measured before it; `--dit-runtime native` still selects that path. The `ring-fused` rows are single sessions on
macOS 27.2 (same machine and MLX). Over the chunks with a full KV window (chunk 21 on), the `ring-fused` chunk median
is 3.53 s (castle), 3.63 s (castle with browser) and 3.48 s (racing). The quality `ring-fused` row was measured
alone; its swap is the system-wide growth of `vm.swapusage` (4.65 GB was already in use before the run; an earlier
run gave 27.3 s and +2.6 GB with 4.46 GB in use) and its footprint equals `native`'s. The dynamic-captions
`ring-fused` row (new defaults: `--t5-runtime graph`, `--vae-graph`; a new caption every chunk,
`--prompt-interval-chunks 1`) started with 6.78 GB of swap in use; like the `native` rows it is memory-bound (see
Memory). The 26.4 GB under Memory is MLX's peak allocation in a direct `CastleStream` harness, not a footprint, so
it does not compare with this column.

Time to the first encoded frame, `native`: realtime 3.6 to 7.8 s (10.5 s for the castle no-browser row, 13.5 s with
dynamic captions), quality 12.4 s (13.3 s in the end-to-end test). `ring-fused`: castle realtime 8.2 s, racing
realtime 6.6 s (21.6 s with dynamic captions), castle quality 16.4 s (16.6 s in an earlier run), including the compiled graphs' warm-up at stream creation; all well inside
`serve`'s deadline (45 s realtime, 300 s quality). With `native`, realtime chunk medians are about 3.3 s over a
14-chunk session and 3.8 s over 32 (the KV window fills at chunk 19).
The `--fps` flag does not change the generation rate: the model sets the pace and the game waits for it, so
`--fps` sets how much game time one generated frame covers.

What a player sees (Chromium counters over the last 18 s, `window.castleDebug()`: frames presented by the browser
and game frames, per second): about 0.7 to 1.1 presented and 0.9 to 1.2 game frames per second for every `native`
realtime row, and 1.09 presented and 1.09 game frames per second for `ring-fused` castle at `--fps 4`, whatever `--fps`: **a slideshow of about one image per second**, with the 4-frame chunks arriving in bursts. Only the
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

Realtime fixed-caption sessions peak at 16.2 to 16.6 GB footprint (`--preset apple` the same, `apple-fast` 14.6 to
15.2 GB), quality at 24.2 GB: both fit in 32 GB with the
browser. **Racing with dynamic captions does not fit sustainably on a 32 GB Mac.** T5 stays resident (no offload),
the footprint reaches 25.5 GB (24.5 GB in the end-to-end test, which stops after 9 frames), and the machine swaps:
swap grew by 3.8 GB without a browser and 3.1 GB with one, later chunks took 31 to 35 s instead of 4 s,
and with the browser the session was lost ("Browser disconnected": the page stopped answering the server's
15 s watchdog). These runs were at `--fps 24`; `--fps` changes neither the generation rate nor the memory, so the
same holds at `--fps 4`. Use `--prompt-mode fixed` (the default) on 32 GB; dynamic captions need a Mac with more memory.
(The swap file already held 4 to 5 GB from earlier work when these runs started, so a freshly booted Mac
may do better; that was not tested.)

The Apple fast path does not change this: dynamic captions on a 32 GB Mac are **memory-bound, not compute-bound**.
Measured with `ring-fused` on racing realtime (`CastleStream` driven directly, a new caption every other chunk, T5
resident): peak 26.4 GB; one caption update costs 4.6 to 7.9 s in situ against 1.5 s of T5 compute alone, and it
slows the next 1 to 2 chunks by up to several seconds. `--t5-runtime graph` saves 21 ms per encode; the levers that
would help are memory ones (T5 off the GPU or quantized, fewer distinct captions), outside this parity-kept path.

### Quality preset in a browser

A browser keeps an idle `/api/start` request open for a limited time (WebKit about 60 s), while the server waits for
the first encoded frame. On this Mac quality reached it in 12 to 14 s, well inside that, but a slower Mac may not.
`serve` allows 300 s for the Wan-VAE presets (45 s for `realtime`, upstream's value, and for `apple` / `apple-fast`, which share its base). The 15 s silence watchdog
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
- The first-frame deadline is 45 s for `--preset realtime` (upstream's value) and for `apple` / `apple-fast` (same base), and 300 s for the Wan-VAE presets, which need
  longer on Apple Silicon; the error text keeps upstream's shape.
- `Runtime.__init__` no longer resets `attention_backend` (the MLX loader sets `mlx-sdpa`).
- No torch: the session report drops `profile_torch_threads`; the stream hands an owned `uint8` array to `on_rgb` (no `.numpy()`); profile keys are
  `gpu_<stage>_ms` instead of `cuda_<stage>_ms`.
- CUDA-only model options keep the treatment `render` gives them (for example `--preset rtx5090` exits with "--preset rtx5090 is CUDA-only"); the realtime DiT runtime is `ring-fused`, as upstream.
  `serve --dit-runtime {native,ring-fused}` is accepted on every preset (`native` is the reference path; the CUDA-only
  values are rejected); `ring-fused` warms its compiled graphs up at stream creation, inside the first-frame deadline
  (see "Measured rates" and "Apple fast path").
- The web client is copied byte-for-byte, and `matrix_game_mlx/web/mediamtx-reader.LICENSE` ships in the wheel (`license-files`).
- Game patches: none. Both `CastleReferenceServer` and `CitySampleReferenceServer` create their hidden SDL/OpenGL window from a
  subprocess on macOS unchanged.

Upstream behavior kept as is: stopping a session (Stop, or closing the tab) while the browser is still sending controls
can log `ERROR Error handling request` with `OSError: [Errno 9] Bad file descriptor` from `game.py`'s `send`. It is
harmless: `CastleGame.close()` closes the game socket without taking the send lock, so a control that passed the
closed check writes to the closed socket, and `send` only ignores `BrokenPipeError` and `ConnectionResetError`. The
session is still cleaned up (`webrtc_cleanup_confirmed: true` in its report). It shows up mostly in racing, where the
client sends controls continuously. The server code is upstream's, so CUDA has the same race.

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
it (Safari itself is not automated; do upstream's manual Safari check). The exception is `test_browser_model`
(`needs_weights`): it runs `test_browser.cjs` in full (reference, model, model dual view at 1728 wide) against a castle
realtime session at `--fps 4` with `DISPLAY_ONLY=1`, from a temporary copy with two edits asserted against upstream's
text, because the script's timings assume CUDA speed: 8 new game frames in 20 s instead of 40, and a wait for the HUD
before the final inspection (the client hides it after 1.5 s without video). Keyboard, mouse and audio are therefore
browser-tested in reference mode only, not under the model; check them by hand. If WebKit receives no video on a fresh machine, check System Settings, Privacy & Security, Local Network for the
browser: after a new install WebKit got no WHEP video for about an hour, then worked without any change. Measure with
`python3 -m tests.measure_serve --scene castle --fps 4 --chunks 32 [--browser] [--prompt-mode dynamic] [--dit-runtime native|ring-fused]`.

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
These renders predate the Apple fast path (`native` DiT runtime); "Apple fast path" has the `ring-fused` timings.

## Apple fast path

`--dit-runtime ring-fused` is the MLX counterpart of upstream's CUDA `ring-fused` runtime and, as upstream, the
default of the realtime preset (`render` and `serve`); the quality and baseline presets default to `native`. Both
runtimes are accepted on every preset (`$S` as under "Online play on macOS", Run):

```bash
matrix-game-mlx render --scene castle --preset realtime --dit-runtime native --input clip.mp4 --output output/castle-native
matrix-game-mlx serve --scene castle --preset realtime --dit-runtime native --fps 4 $S      # the reference path
matrix-game-mlx serve --scene castle --preset quality --dit-runtime ring-fused --fps 4 $S   # fast path on quality
```

`native` stays the reference (upstream: "Fast runtime modes alter floating-point reduction order. Keep `native` for
reference checks."). `--vae-graph` (default on for realtime + TAEHV) and `--t5-runtime graph` (default) are the two
smaller compiled paths; `--no-vae-graph` and `--t5-runtime native` turn them off. An unsupported configuration
under `ring-fused` is a `ValueError`, never a silent fall back to `native`.

### What each lever does and what it gave

Measured with `python3 -m tests.bench_dit --scene castle --preset realtime` (real weights, `CastleStream.push` on
synthetic frames, each DiT call and codec stage timed with `mx.eval`; M2 Pro 32 GB, MLX 0.31.1), one run alone per
row. Chunk = one latent (4 frames): one denoising call, one cache-update (`clean_kv`) call, TAEHV encode and decode.

| Step (cumulative) | Chunks | Denoise | Clean KV | Chunk median | Steady chunk (window full) | Peak |
| --- | --- | --- | --- | --- | --- | --- |
| `native` (before) | 14 | 1766 ms | 1717 ms | 3589 ms | | 14.94 GB |
| + ring KV cache, GPU RoPE kernel | 14 | 1526 ms | 1516 ms | 3190 ms (−11 %) | | 14.94 GB |
| + fused block-glue kernels | 14 | 1463 ms | 1383 ms | 2941 ms (−18 %) | | 14.94 GB |
| same code, 32 chunks | 32 | 1681 ms | 1652 ms | 3454 ms | 3700 ms | 14.94 GB |
| + compiled steady-window graph, timestep-MLP memo, fewer syncs | 32 | 1675 ms | 1624 ms | 3422 ms | 3676 ms | 14.94 GB |
| + `--vae-graph` (TAEHV encode 46 -> 39 ms, decode 79 -> 71 ms) | 32 | 1643 ms | 1590 ms | 3345 ms | 3656 ms | 14.86 GB |

DiT time grows with the KV cache until the 21-latent window is full (about chunk 20 in castle realtime), then stays
flat; a 14-chunk median therefore sits below a 32-chunk one, and only the 32-chunk rows compare with each other.
Sequential runs drift by 30 to 40 ms (thermal), so the compiled graph's gain was measured interleaved instead: its
row shows −24 ms on the steady chunk between two sequential runs, item 5 below gives the interleaved A/B (−59 ms).

End to end in `serve` (castle realtime, `--fps 4`, 32 chunks, no browser, `tests/measure_serve.py`): chunk median
3.89 s with `native` and 3.29 s with `ring-fused` (−15 %), `model_fps` 1.05 -> 1.24, `online_fps` 0.95 -> 1.13
(+19 %), first frame 10.5 -> 8.2 s, footprint 16.2 GB both, no swap. All rows are under "Measured rates".

1. **RoPE on the GPU** (`inference/rope_kernel.py`, upstream's `build_causal_rope_rotations` / `apply_causal_rope`).
   The rotation table of a call's positions is built once per DiT call (float64 bits, about 0.3 ms on the host) and
   shared by the 30 layers and q/k; one Metal kernel applies it. One call's 60 RoPE applications: 326 ms on the
   CPU float64 path, 15.8 ms on the GPU, and no GPU -> CPU -> GPU round trip per layer. **Bitwise guarantee**: the
   kernel emulates the float64 chain in 64-bit integer soft-float (exact products, round-to-nearest-even sums with a
   sticky bit, then MLX's own float64 -> float32 -> bf16 double rounding), so its bf16 output equals the CPU path's
   bit for bit. `tests/test_rope_kernel.py` compares raw bf16 bits with `causal_rope_apply` on every finite bf16
   pattern (65,024, in both lanes) at start frames 0, 22 and 8000, on random tensors at 21 position/scale cases, on
   rounding hard cases (ties, cancellation), on inf/NaN inputs and for the fixed-sink re-RoPE
   (`rope_apply_frames`); the kernel's emulated float64 lanes also equal the CPU float64 lanes on all 64 bits.
2. **Ring KV cache** (`inference/runtime.py`: `plan_cache`, `run_blocks`, `CastleRuntime`, `install_runtime`, as
   upstream). The host planner writes each new latent into a ring slot and evicts one frame per chunk instead of
   shifting the window. Together with the GPU RoPE: −399 ms per chunk (14-chunk median; the two were measured
   together).
3. **Fused block glue** (`inference/kernels.py`): `ln_modulate`, `gate_ln_affine`, `gate_residual` as upstream,
   plus `rms_norm` (q/k norm) and a compiled `gelu_tanh`, one launch each. −249 ms per chunk. Cost of reverting
   each one alone (same session): LayerNorm pair +138 ms, `rms_norm` +60 ms, `gelu_tanh` +35 ms, `gate_residual`
   +26 ms (within noise, kept: upstream's kernel, bitwise).
4. **Fewer copies**: measured, not changed. Removing the fixed-sink K/V concatenation and the ring-eviction copy
   (timing-only ablations on a real steady call) changed nothing beyond the ±20 to 30 ms noise: MLX overlaps these
   memory-bound copies with the GEMMs.
5. **Compiled steady-window graph, fewer syncs** (`CastleRuntime.prime` / `graph` / `clear`, upstream's "one
   steady-window graph per session" on `mx.compile`). Once the window is full, the 30 blocks run as one compiled
   function; host graph build drops from 25 to 8 ms per call. With the timestep-MLP memo (about 25 ms of float64
   CPU-stream work per call) and one-sync stream checks: **−59 ms per steady chunk (−1.6 %)**, interleaved A/B,
   10 rounds, real weights. The warm-up runs at stream creation (as upstream), adding about 1.7 s to the first frame.
6. **Smaller items, done only above 3 % of the steady chunk (about 110 ms)**:

   | Item | Measured cost | Decision |
   | --- | --- | --- |
   | Timestep MLP (float64 accumulation), with the memo | 0.88 ms per chunk (0.02 %) | skipped |
   | flow-to-x0 (float64, CPU stream) | 0.39 ms per chunk (0.01 %) | skipped |
   | TAEHV encode + decode | 121 ms per chunk (3.3 %) | done: `--vae-graph`, −15 ms per chunk |
   | Caption encode (racing `--prompt-mode dynamic`) | 1535 ms of T5 compute per caption update | done: `--t5-runtime graph`, −21 ms per encode (1515 vs 1535 ms) |

Where the time goes now (one steady fixed-sink castle call, full window, real weights, each part timed alone):
the call takes 1786 ms, of which 1174 ms are the 210 GEMMs (about 3.1 TFLOPS) and 513 ms are attention (SDPA,
self 484 ms + cross 29 ms): **GEMM + SDPA are 94 % of a DiT call**. Their kernels are MLX's own; making them
faster at the same precision would need custom GEMM/attention kernels, and anything else (fp16, quantized weights,
a lighter cache-update pass) changes the numerics (fp16 is the port-only "Apple presets"). So the end-to-end gain stays modest. Per-part numbers here come
from direct `mx.eval` timings. smeltr's scope totals are consistent with them (GPU time per DiT call now equals
its wall time: the CPU RoPE was the off-GPU part), but its per-op split is not reliable on this machine (ops of one
command buffer share its time), so no per-op percentage is quoted.

### Parity: `ring-fused` against `native`

Every real-weight parity slice passes on both runtimes with **thresholds and fixtures unchanged**. Each slice
loads a real DiT: run one file per command and one command at a time (never several slices in one process, never two
heavy processes at once); `--dit-runtime native` runs the reference:

```bash
pytest tests/parity/test_real_weights.py --dit-runtime ring-fused            # quality: per call, per block, end to end
pytest tests/parity/test_real_weights_realtime.py --dit-runtime ring-fused
pytest tests/parity/test_real_weights_variants.py --dit-runtime ring-fused
pytest tests/parity/test_real_weights_captions.py
```

The fast path may differ from `native` only by floating-point summation order, never by precision:

- GPU RoPE, fused kernels, the compiled graph, the memo, `--vae-graph` and `--t5-runtime graph` are each **bitwise**
  their eager / native counterpart (tests in `tests/test_rope_kernel.py`, `test_kernels.py`, `test_runtime.py`,
  `test_taehv.py`, `test_caption_encoder.py`; on real castle weights, one steady fixed-sink position (denoise +
  cache update) on `ring-fused` and `apple`, the compiled DiT graph equals the eager one bit for bit, prediction and
  every K/V).
- The one numeric difference is the ring's attention key order, which changes SDPA's summation order. Among the
  parity tests it shows only in the `--long` chunk-22 per-call tests, past the window, where ring slots are permuted:

| Per-call test (teacher forced) | pred rel L2 `native` | `ring-fused` | gate |
| --- | --- | --- | --- |
| ancient-egypt chunk 22, call 0 | 0.0381 | 0.0308 | 0.07 |
| ancient-egypt chunk 22, call 1 | 0.0202 | 0.0200 | 0.04 |
| racing chunk 22, call 0 | 0.0129 | 0.0119 | 0.02 |
| racing chunk 22, call 1 | 0.0102 | 0.0108 | 0.02 |

  Every other printed metric is identical on both runtimes: per-call pred and layer 0/29 KV of chunks 0 to 2 on the
  three scenes, the 90 block errors (max 0.55 to 0.56 of the gate on both), the end-to-end quality stream (chunk 0
  pred 1.71e-2 / 1.41e-2 / 0, decoded 44.9 / 43.7 / 56.4 dB), and the realtime and variant chunk metrics. These are
  all **pre-window** fixtures (chunks 0 to 2, before the first ring eviction, where ring and linear slot orders
  coincide and the two runtimes are bitwise equal); only the teacher-forced chunk-22 calls above reach past it.
- Past the window, free running (`tests/compare_runtimes.py`, below), the two runtimes diverge, and the divergence
  compounds chunk over chunk.
- `mx.compile` is not bitwise for every op on MLX 0.31.1: a division by a non-power-of-two constant rounds
  differently once compiled, and `mean()` gets its 1/n scale fused into the reduction, which then sums in another
  order. The compiled DiT graph is nevertheless bitwise eager on real weights (measured, above); `wan/modules/t5.py`
  was rewritten so that the compiled T5 stays bitwise (see Known deviations).

#### Free-running streams past the window

`python3 -m tests.compare_runtimes` runs the same realtime stream (the parity fixtures' 8 px pan over the scene's
sink image, stream seed 0, 48 chunks) once per runtime, one process each, then compares chunk by chunk (pred rel L2
of `ring-fused` against `native`, decoded-frame PSNR with peak-to-peak 2, as the parity tests). Every chunk before
the first ring eviction (stream position 21: castle chunk 20, its sink being position 0; racing chunk 21) is bitwise
equal. From there on each runtime conditions on its own outputs, so the per-call summation-order difference grows:

| Scene | Chunks | pred rel L2, median / max | decoded PSNR, median / min |
| --- | --- | --- | --- |
| castle | 0 to 19 | 0 (bitwise) | identical |
| castle | 20 (first eviction) | 0.0445 | 40.9 dB |
| castle | 21 to 29 | 0.27 / 0.36 | 27.5 / 24.4 dB |
| castle | 30 to 39 | 0.47 / 0.54 | 19.0 / 17.0 dB |
| castle | 40 to 47 | 0.58 / 0.62 | 18.1 / 16.1 dB |
| racing | 0 to 20 | 0 (bitwise) | identical |
| racing | 21 to 29 | 0.047 / 0.085 | 39.8 / 34.5 dB |
| racing | 30 to 39 | 0.065 / 0.092 | 37.0 / 33.9 dB |
| racing | 40 to 47 | 0.096 / 0.14 | 34.0 / 31.1 dB |

Over chunks 21 to 47: castle 0.47 / 0.62 (median / max pred rel L2), 19.4 / 16.1 dB (median / min PSNR); racing
0.072 / 0.14, 36.1 / 31.1 dB. These numbers are the measured cost of the ring's key order in a free-running stream, reported, not
gated. For scale, the same tool measures how far each stream carries a small input change on one runtime: `native`
against `native` with one input frame raised by one 8-bit level at the first eviction chunk (`--perturb-frame 77`
castle, `81` racing) gives, over chunks 21 to 47, castle 0.46 / 0.59 and 19.7 / 17.0 dB, racing 0.070 / 0.20 and
36.4 / 29.0 dB: the same size as the runtime difference. The divergence is the free-running stream's own
sensitivity (castle amplifies small differences much faster than racing, as in the quality preset, where torch
against its own kernel variant already reaches 0.478 pred rel L2 at castle chunk 1), not a defect specific to the
ring. To reproduce (each run loads the real DiT: one at a time):

```bash
python3 -m tests.compare_runtimes --scene castle --preset realtime --chunks 48 --dit-runtime native --out out/native
python3 -m tests.compare_runtimes --scene castle --preset realtime --chunks 48 --dit-runtime ring-fused --out out/ring
python3 -m tests.compare_runtimes --compare out/native out/ring --json out/compare.json
```

## Apple presets

`--preset apple` and `--preset apple-fast` are port-only speed presets that knowingly leave upstream exactness, as
upstream does for its own hardware with `--preset rtx5090` / `rtx5090-sage` (FP8 linears, SageAttention). They
resolve as upstream resolves `rtx5090`: the realtime base (TAEHV codec, steps `1000`, `--vae-graph` on) and
`--dit-runtime` defaulting to the preset's name. `--dit-runtime apple` / `apple-fast` are also accepted on any
preset, as upstream accepts `--dit-runtime rtx5090`. Defaults do not change: `realtime` stays the default `serve`
preset with `ring-fused`, and `native` stays the reference.

| Preset / runtime | What it changes from realtime `ring-fused` |
| --- | --- |
| `apple` | The DiT runs in float16 instead of bfloat16: weights, activations, modulation, self- and cross-attention K/V caches, the fused kernels (`half` variants), the RoPE kernel output, the compiled steady graph and `prime` |
| `apple-fast` | `apple` plus a self-attention window of 11 latents instead of 21: the sink, the current latent and the 9 latents before it (the 21 window: 19), so from stream position 11 on it sees less history; the KV cache is sized by it, the sink stays one latent |

```bash
matrix-game-mlx serve --scene castle --preset apple-fast --fps 4 $S          # $S as under "Online play on macOS"
matrix-game-mlx serve --scene racing --preset apple --fps 4 $S
matrix-game-mlx render --scene ancient-egypt --preset apple-fast --input clip.mp4 --output output/egypt-apple-fast
```

What stays as in `ring-fused`: the precision islands (fp32 norm statistics, opmath SiLU / tanh-GELU, the float64
timestep MLP and its memo, float64 flow-to-x0 on the CPU stream, the float64 RoPE rotation, now rounded once to
fp16), the ring planner, the fixed-sink re-RoPE and the compiled graph. T5 and TAEHV stay bfloat16: the DiT casts the
bf16 prompt embeddings and the bf16 control latent to fp16 where it consumes them, and the stream casts the fp16 pred
to bf16 before the TAEHV decode (under `--preset quality --dit-runtime apple` the Wan decoder likewise receives
bf16, pinned by a tiny-model stream test; that combination was not timed on real weights).
The bf16 pack is converted to fp16 when it is loaded, one tensor at a time, so two DiT copies never coexist; there is
no fp16 pack. Anything the apple runtimes cannot run (a bf16 DiT, bf16 caches, another batch or chunk size) is a
`ValueError`, never a fall back to bf16 or `ring-fused`. The loader warns that the DiT runs in float16 and that
`native` is the baseline.

### Speed

`python3 -m tests.bench_dit --scene castle --preset realtime --chunks 32 --dit-runtime {ring-fused,apple,apple-fast}`
(same harness and machine as "What each lever does", one run alone per row, macOS 27.2, MLX 0.31.1):

| `--dit-runtime` | Denoise | Clean KV | Chunk median | Steady chunk (window full) | First frame | Peak |
| --- | --- | --- | --- | --- | --- | --- |
| `ring-fused` | 1649 ms | 1589 ms | 3358 ms | 3609 ms (chunks 21 to 32) | 7.0 s | 14.86 GB |
| `apple` | 1386 ms | 1369 ms | 2887 ms (−14 %) | 3095 ms (−14 %, chunks 21 to 32) | 6.1 s | 14.86 GB |
| `apple-fast` | 1272 ms | 1244 ms | 2654 ms (−21 %) | 2660 ms (chunks 11 to 32) | 5.3 s | 13.37 GB |

`apple-fast`'s window is full from chunk 11 instead of 21, so its steady median covers other chunks; its last 4
chunks average 2662 ms against 3606 ms for `ring-fused` (−26 %). In `serve` (castle and racing realtime, `--fps 4`,
32 chunks, no browser, rows under "Measured rates"):

| Scene | `ring-fused` | `apple` | `apple-fast` |
| --- | --- | --- | --- |
| castle: chunk median, `model_fps`, `online_fps`, footprint | 3.29 s, 1.24, 1.13, 16.2 GB | 2.98 s, 1.36, 1.25, 16.2 GB | 2.78 s, 1.46, 1.43, 15.2 GB |
| racing (fixed caption) | 3.35 s, 1.24, 1.15, 16.6 GB | 2.98 s, 1.38, 1.28, 16.6 GB | 2.72 s, 1.49, 1.46, 14.6 GB |

fp16 does not lower the footprint (same MLX peak, 14.86 GB); the shorter window does (smaller KV caches). No fixed-caption session
swapped. Racing with dynamic captions stays memory-bound, as under "Memory": `--preset apple --prompt-mode dynamic`
(8 chunks, a new caption every chunk) completed with a 15.5 s chunk median, `model_fps` 0.29, footprint 25.5 GB and
5.5 GB of swap growth.

### Quality

Gates fixed before the first run, on the realtime fixtures' gated chunks (the first denoise calls fed torch's sink,
prompt and noise; `tests/parity/test_real_weights_apple.py`): pred rel L2 vs torch ≤ max(6e-2, 2 x the realtime gate
of the chunk), decoded PSNR ≥ 30 dB, control rel L2 < 2e-2 (TAEHV is unchanged). One scene per process, one process
at a time (each loads the real DiT; about 4 min and up to 22.9 GB of process footprint per scene on the M2 Pro):

```bash
pytest tests/parity/test_real_weights_apple.py -s -k castle        # then -k ancient-egypt, -k racing
```

Every gate passes:

| Scene, chunk | pred rel L2 `apple` (gate) | `ring-fused` | decoded PSNR `apple` | `ring-fused` |
| --- | --- | --- | --- | --- |
| castle, 0 | 0.0246 (0.06) | 0.0203 | 41.8 dB | 42.7 dB |
| ancient-egypt, 0 | 0.0308 (0.0694) | 0.0275 | 41.9 dB | 41.8 dB |
| racing, 0 (reference-first sink) | the sink latent cast to fp16, bitwise (1.9e-10 vs torch) | 0 | 59.6 dB | 59.6 dB |
| racing, 1 | 0.0244 (0.06) | 0.0151 | 47.8 dB | 51.1 dB |

Control rel L2 is at most 8.3e-3. `apple-fast` equals `apple` bitwise on every fixture chunk (the window only acts
from position 11; the fixtures stop at chunk 2), which the test asserts. fp16 safety: 32 free-running chunks per
scene and per runtime (eager, the parity pan, seed 0) keep every pred and K/V finite, and the residual stream's
max |x| peaks at 790 (castle's sink pass at stream creation; 728 in a chunk) against a limit of 65504 / 16 = 4094.
The fp16 linear outputs that enter a fused kernel (q / k projections, self- and cross-attention outputs, FFN
hidden pre-activation and FFN output) peak at 208 (castle's FFN output; FFN hidden 57) against a limit of
65504 / 4, also fixed before the run: fp16 overflow is about 300x away.
The compiled fp16 graph is bitwise its eager run on real castle weights (one steady fixed-sink position, denoise
and cache update: prediction and every K/V; `tests/test_runtime.py`).

Free running (`tests/compare_runtimes.py`, as under "Free-running streams past the window": 48 chunks, the parity pan,
stream seed 0, against `native`; `--preset realtime --dit-runtime apple|apple-fast`), reported, not gated:

```bash
python3 -m tests.compare_runtimes --scene castle --preset realtime --chunks 48 --dit-runtime apple --out out/apple
python3 -m tests.compare_runtimes --scene castle --preset realtime --chunks 48 --dit-runtime apple-fast --out out/apple-fast
python3 -m tests.compare_runtimes --compare out/native out/apple --json out/compare-apple.json   # out/native as above
```

| Scene | Runtime | Chunks | pred rel L2, median / max | decoded PSNR, median / min |
| --- | --- | --- | --- | --- |
| castle | `apple` | 0 to 20 | 0.57 / 0.66 | 16.3 / 15.3 dB |
| castle | `apple` | 21 to 47 | 0.59 / 0.65 | 17.1 / 16.0 dB |
| castle | `apple-fast` | 0 to 20 | 0.57 / 0.66 | 16.3 / 15.3 dB |
| castle | `apple-fast` | 21 to 47 | 0.62 / 0.80 | 16.4 / 13.5 dB |
| racing | `apple` | 0 to 20 | 0.28 / 0.45 | 26.0 / 21.8 dB |
| racing | `apple` | 21 to 47 | 0.47 / 0.57 | 21.1 / 19.1 dB |
| racing | `apple-fast` | 0 to 20 | 0.28 / 0.45 | 26.0 / 21.0 dB |
| racing | `apple-fast` | 21 to 47 | 0.53 / 0.65 | 19.4 / 17.1 dB |
| castle | `ring-fused` (above) | 21 to 47 | 0.47 / 0.62 | 19.4 / 16.1 dB |
| racing | `ring-fused` (above) | 21 to 47 | 0.072 / 0.14 | 36.1 / 31.1 dB |
| castle | `native`, input frame 77 +1 level | 21 to 47 | 0.46 / 0.59 | 19.7 / 17.0 dB |
| racing | `native`, input frame 81 +1 level | 21 to 47 | 0.070 / 0.20 | 36.4 / 29.0 dB |

Unlike `ring-fused`, which is bitwise `native` until the first ring eviction, fp16 differs from the first denoised
chunk on (castle chunk 0: 0.020, 44.0 dB; chunk 1: 0.096, 35.9 dB; racing chunk 1: 0.022, 49.3 dB, its chunk 0 being
the sink, bitwise), and each runtime then conditions on its own outputs, so the difference compounds over all 48
chunks while the perturbation baseline starts at chunk 20 (castle) or 21 (racing). Castle reaches the size of a
one-level input change on `native` (and of `ring-fused`) within 4 chunks and stays somewhat above it (chunks 21 to 47: pred rel L2 median 0.59 vs 0.46, PSNR median / min 17.1 / 16.0 dB vs 19.7 / 17.0 dB). Racing, which amplifies small
differences much more slowly, ends far above both: over chunks 21 to 47 its pred rel L2 median is 0.47 (`apple`) and
0.53 (`apple-fast`) against 0.070 for the perturbation. `apple-fast` is bitwise `apple` until its window acts (castle
chunk 10, racing chunk 11, both stream position 11) and diverges from it afterwards, by as much as fp16 does from
bf16 (chunks 21 to 47: castle 0.59 / 0.80, racing 0.49 / 0.60 pred rel L2 median / max). These are distances between
trajectories, not quality scores: the same pan through the same model takes another path, and the comparison does
not say which path looks better. The `native` runs were redone for this table (the perturbation runs were not: the
`native` path is unchanged since they were measured).

## Numerics

Upstream's float64 islands (RoPE complex product and tables, flow-to-x0, timestep sinusoid) run in float64,
on the MLX CPU stream where MLX has no GPU float64 (under `ring-fused`, the RoPE product runs on the GPU as an
integer emulation of the same float64 arithmetic, bitwise equal; see "Apple fast path"). bf16 rounding points follow torch's opmath rather than
MLX's per-op defaults: SiLU / tanh-GELU and the VAE `RMS_norm` compute in fp32 and round once, and the
timestep-MLP linears (one row per timestep entry, B x F) accumulate in float64, bias included, and round once
(MLX's single-row bf16 addmm rounds the product before adding the bias, and fp32 summation order flips a few
outputs at any row count). The scheduler's sigma grid reproduces `torch.linspace` bitwise. The umT5 text
encoder runs in bf16, as upstream's loader runs it (`WanTextEncoder(...).to(device, torch.bfloat16)`),
attention bias included (`x.new_zeros`). The port-only apple runtimes run the DiT in float16 instead, with the same
islands (the RoPE kernel then rounds the float64 rotation once to fp16); see "Apple presets".

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
  on the GPU; the values are upstream's, the device is not. Under `ring-fused`, RoPE runs on the GPU instead
  (bitwise the same values, below).
- KV-cache end indices are Python ints instead of 1-element tensors.
- `CastleStream.push` has a test-only `renoise` argument.
- Only the inference path (`kv_cache is not None`) is ported; training branches raise `NotImplementedError`.
- `--dit-runtime ring-fused` (the realtime default, as upstream) is an MLX runtime, not upstream's CUDA one: the
  same host ring-KV planner (`plan_cache`, `run_blocks`, `CastleRuntime`, `install_runtime`), RoPE applied by a GPU
  kernel bitwise equal to the float64 CPU path, fused Metal block-glue kernels (`inference/kernels.py`, below), the
  steady-window graph compiled with `mx.compile` (below) instead of a CUDA graph, and no varlen `cu_seqlens`. Its
  only numeric difference from `native` is the
  attention key order (ring slots change the SDPA summation order; with the linear layout it is bitwise native,
  fused or not). This is the difference upstream itself documents for its fast runtimes ("Fast runtime modes alter
  floating-point reduction order. Keep `native` for reference checks.", `docs/architecture.md`; the loader warns
  "Ring/fused runtime changes floating-point reduction order; use native for baseline comparisons"). On real weights
  among the parity tests it shows only in the `--long` chunk-22 per-call tests; free running past the window, the
  streams diverge as far as a one-level input change carries them (see "Apple fast path", free-running streams). Measured on the tiny bf16 model over 60 chunks
  (`tests/test_runtime.py`): max rel L2 vs native 2.4e-3 per call, 1.2e-3 on the written KV slot; most calls are
  bitwise (the fp32 reorder rarely changes a bf16 output bit). Like upstream, the loader logs that ring changes the
  reduction order and that native is the baseline. Port-only checks: the input, KV caches and modulation must be in
  the runtime's dtype (bfloat16 for `ring-fused`), and every layer is validated before any cache is planned; an
  unsupported configuration is a `ValueError`, never a native fallback. `CastleRuntime` accepts `ring-fused` and the
  port-only `apple` / `apple-fast` (below); `install_runtime` also `native`.
- `ring-fused` RoPE: a Metal kernel (`inference/rope_kernel.py`, `mx.fast.metal_kernel`) instead of upstream's
  Triton kernel, with upstream's names (`build_causal_rope_rotations`, `apply_causal_rope`) and validation messages.
  Apple GPUs have no float64, so the rotation table holds the float64 bits as uint64 and the kernel emulates the
  float64 complex product in 64-bit integer soft-float, then rounds as MLX's CPU float64 -> bf16 cast does (through
  float32). Its bf16 output is bitwise the CPU float64 path's (`causal_rope_apply`, and `rope_apply_frames` for the
  fixed-sink re-RoPE; `tests/test_rope_kernel.py`). For the apple runtimes' float16 input the same kernel emits
  float16, rounded as MLX's CPU float64 -> float16 cast does (once, with fp16 subnormals and overflow to inf), and is
  bitwise the CPU float64 path cast to float16 (every finite fp16 pattern, random tensors up to start frame 8000,
  rounding and overflow ties): no float32 rotation is involved. The fixed-sink re-RoPE uses
  `build_causal_rope_rotations(freqs, 1, h, w, position - window)` (the sink's one frame, as upstream,
  `inference/runtime.py`); `build_rope_rotations_frames` (explicit frame positions, as `rope_apply_frames`) is a
  test-only helper of the bitwise tests.
- `ring-fused` fused kernels (`inference/kernels.py`, upstream's `kernels.py` names on MLX arrays): `ln_modulate`,
  `gate_ln_affine` and `gate_residual` as upstream, plus two port additions, `rms_norm` (the q/k `WanRMSNorm`) and
  `gelu_tanh` (the FFN activation as one `mx.compile`d kernel). They follow upstream's rounding contract (every bf16
  intermediate of the separate ops rounded to bf16, fp32 statistics, `#pragma METAL fp contract(off)` so no FMA the
  separate kernels would not do) and in addition sum each row in the layout of MLX's own contiguous row reduction
  (`row_reduce_simple`: C/4 threads, 4 consecutive values each), so at the model widths (3072, and 128 in the tests)
  they are bitwise equal to the native block pieces on MLX 0.31.1 (`tests/test_kernels.py`; a 60-chunk rollout is
  bitwise fused vs unfused in `tests/test_runtime.py`). Upstream's Triton kernels differ from its unfused block in
  the summation order inside a row reduction; here that difference is not used at the model widths, and at other
  widths (or with another MLX version whose row reduction is dispatched differently) the summation order is the only
  possible difference. Port-only: float16 variants for the apple runtimes (same contract against the native fp16
  expressions, `*_fp16` tests; the bf16 kernels are unchanged); the fused path requires modulation and norm weights
  in the runtime's dtype, an affine `norm3` and RMS q/k norms (`ValueError` otherwise, checked before any cache is
  planned);
  `*_row` argument names (upstream `*_k`); no stride arguments (MLX makes kernel inputs row-contiguous).
- `ring-fused` steady-window graph: upstream's `CastleRuntime.prime` / `graph` / `clear` / `use_graph` names and its
  "one steady-window graph per session" rule (positions, RoPE rotations and timestep modulation are inputs; no
  per-position cache; `captures` / `replays` counters, the tracing call counted as a replay as upstream's capture
  call), on `mx.compile` instead of CUDA graph capture: the compiled
  function is keyed on the model, shapes, sink mode and write spans (not buffer addresses), takes every KV / text
  cache tensor as an input and returns the updated K/V, and reads the block weights as implicit inputs. Compilation
  keeps the traced arithmetic (the fused kernels and SDPA are opaque to it): a 60-chunk rollout is bitwise compiled
  vs eager (`tests/test_runtime.py`), and on real castle weights one steady fixed-sink position (denoise + cache
  update) gives bitwise equal predictions and K/V (compiled vs eager, `ring-fused` and `apple`). `mx.compile` on MLX 0.31.1 is not bitwise for every op
  (a division by a non-power-of-two constant, `mean()`; see `--t5-runtime graph` below), so the DiT graph's
  equality is a measured result, not assumed. `CastleStream` calls `prime` before gameplay and `clear` on `close()`, as
  upstream; the warm-up (one DiT call plus the trace) is part of stream creation.
- `ring-fused` memoizes the timestep-embedding MLP per timestep value (`CastleRuntime.time_embedding` and its
  `time_memo` dict, port-only: upstream has no memo): the float64-accumulated MLP runs on the MLX CPU stream and every block waits for it, while a stream
  only uses its few denoising timesteps and t=0. A hit returns the arrays computed for the same t (bitwise the same
  values); an entry misses when any MLP weight array is replaced, and `clear()` empties it.
- `CastleStream._validate` reads its value checks (pixels finite and in [-1, 1], initial noise finite) in one host
  sync, and the finite check of the generated latent is read with the denoising step's own evaluation (same
  checks and messages as upstream).
- `--t5-runtime graph` (`CaptionEncoder`, dynamic captions): upstream's `entry` and one graph traced once, on
  `mx.compile` instead of CUDA graph capture (the padded `(1, 512)` ids and mask are inputs, the T5 weights implicit
  inputs, `captures` / `replays` counters, `replays` counting the tracing encode too); no CUDA / eval-mode check (MLX has neither). Its embeddings are bitwise
  `native` on real weights. Two port edits in `wan/modules/t5.py` make that so without changing the native values:
  the relative-position buckets are computed eagerly once per length and kept (`T5RelativeEmbedding.buckets`;
  compiled, the bucket formula's division lands one bucket lower at distances 16/32/64), and `T5LayerNorm` writes
  its variance as `sum / dim` instead of `mean()` (compiled, mean's 1/n scale is fused into the reduction, which
  then sums in another order; with dim 4096 a power of two, `sum / dim` is bitwise the eager `mean()`).
  Measured: −21 ms (−1.3 %) per caption encode, 1515 vs 1535 ms (M2 Pro, T5 alone).
- `--vae-graph` (`inference/vae_graph.py`: upstream's `StreamingVAEGraph`, `StatefulGraph`, `prime`,
  `encoder_graph` / `decoder_graph`, `replays`): each steady streaming chunk (4-frame encode, 1-latent decode)
  replays one `mx.compile`d function instead of a CUDA graph. The retained MemBlock state is passed in and
  returned (MLX arrays are immutable: no persistent buffers to copy), the TAEHV weights are implicit inputs, and the
  first chunk of a stream runs the eager adapter, as upstream. Bitwise eager (random bf16 weights in
  `tests/test_taehv.py`; on the real pack, latents and frames over 8 chunks in a one-off development check, and the
  realtime parity slices, which run with it on, print unchanged metrics); upstream's CUDA / no-grad checks do not
  apply. Port-only: the compiled decoder refuses (`ValueError`) a decoder still inside its startup frames or holding
  non-array state, the two facts it relies on (it carries only the MemBlock arrays and emits every frame).
  `CastleStream` primes it at stream creation, as upstream. Measured: −15 ms per chunk (encode 46 -> 39 ms,
  decode 79 -> 71 ms, castle realtime on M2 Pro).
- `--preset apple` / `apple-fast` and `--dit-runtime apple` / `apple-fast` are port-only (upstream's hardware
  presets are the CUDA-only `rtx5090` / `rtx5090-sage`); they resolve as upstream resolves `rtx5090` (realtime base,
  `--dit-runtime` defaulting to the preset's name). See "Apple presets" for the measurements behind each item below.
- The apple runtimes run the DiT in **float16** instead of upstream's bfloat16. This is a precision change, not only a
  reduction-order change: weights, activations, modulation and K/V caches have 10 mantissa bits instead of 7 and a
  range up to 65504 instead of bf16's float32 range. The bf16 pack is converted at load, one tensor at a time. The
  realtime gated chunks pass a priori gates (pred rel L2 ≤ max(6e-2, 2 x the realtime gate), PSNR ≥ 30 dB); free
  running, the stream departs from `native` from the first denoised chunk (castle chunk 0: pred rel L2 0.020; racing
  chunk 1: 0.022) and compounds (chunks 21 to 47: castle 0.59, racing 0.47 median). The residual stream's max |x|
  measured 790, against 65504 / 16.
- `apple-fast` attends over **11 latents** instead of upstream's 21: the sink, the current latent and the 9 latents
  before it (the 21 window: 19; each realtime latent is 4 video frames), and the KV caches are sized for 11. So from
  stream position 11 on it sees less history than `apple`. It is bitwise `apple` until then and diverges from it
  afterwards (chunks 21 to 47 pred rel L2 median: castle 0.59, racing 0.49). The window is a
  fixed property of the runtime; there is no window flag (upstream has none).
- Casts at the bf16 / fp16 boundaries under the apple runtimes: T5 and TAEHV (and the Wan VAE) stay bf16; the DiT
  casts the bf16 prompt embeddings and control latent to fp16 where it consumes them, the stream casts the bf16 sink
  latent to fp16 (racing's reference-first chunk 0 is that cast, bitwise) and the fp16 pred to the VAE dtype before
  decoding.
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
