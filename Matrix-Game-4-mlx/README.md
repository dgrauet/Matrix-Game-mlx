# Matrix-Game-4-mlx

MLX inference port of [Matrix-Game 4.0](https://github.com/SkyworkAI/Matrix-Game) for Apple Silicon.
It targets offline rendering: a causal video-to-video renderer that turns a clip into a stylized one
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
export MATRIX_GAME_4_MLX_WEIGHTS_DIR=/path/to/matrix-game-4.0-mlx   # required, see Weights
python -m matrix_game_mlx.inference.offline --scene ancient-egypt --input lowpoly.mp4 --output output/egypt
```

To render a game take recorded with the upstream game (`bash scripts/play.sh --scene ancient-egypt` in
`../Matrix-Game-4/`, F9 to start and stop), pass the take directory; the scene comes from its `session.json`:

```bash
python -m matrix_game_mlx.inference.offline --take /path/to/take-1 --output output/take-1
```

The output directory gets `rendered.mp4` (same frame count and FPS as the input; the last incomplete
four-frame block is padded internally, then trimmed) and `render.json` (configuration, frames, seconds,
`seconds_per_chunk`, `peak_memory_gb`). Other flags: `--seed` (default 0), `--sink-image` (overrides the
scene's sink image), `--preset {quality,realtime}` (default `quality`, as upstream's offline.py). Realtime example:

```bash
python -m matrix_game_mlx.inference.offline --scene castle --preset realtime --input lowpoly.mp4 --output output/castle-realtime
```

## Weights

The weight pack is produced by the [mlx-forge](https://github.com/dgrauet/mlx-forge) conversion of
`Skywork/Matrix-Game-4.0`. Its publication on the Hub as `dgrauet/matrix-game-4.0-mlx` is planned; until then,
set `MATRIX_GAME_4_MLX_WEIGHTS_DIR` to the local pack directory (required). Tests marked `needs_weights` are
skipped when it is unset.

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

They are written to `$MG4_FIXTURES_DIR/<scene>_quality/` or `<scene>_realtime/` (`egypt_quality`, `castle_quality`, `racing_quality`, `egypt_realtime`, `castle_realtime`, `racing_realtime`;
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

A realtime fixture set takes about 10 min per scene and is about 200 MB; `--taehv-only` (~20 s) and `--floor-only` are
partial modes.

## Tests

Run with a Python that has the MLX/torch stack (the development machine uses Homebrew Python 3.14):

```bash
cd Matrix-Game-4-mlx
python -m pytest tests -v
```

The real-weight parity suite runs once per scene whose fixtures exist (a missing or partial set is skipped with
its `gen_fixtures` command) and loads one DiT at a time. Install the `parity` extra for the upstream oracle. Tests marked `needs_fixtures` read the torch reference
fixtures from `MG4_FIXTURES_DIR`; tests marked `needs_weights` need `MATRIX_GAME_4_MLX_WEIGHTS_DIR`.
