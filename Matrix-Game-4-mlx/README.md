# Matrix-Game-4-mlx

MLX inference port of [Matrix-Game 4.0](https://github.com/SkyworkAI/Matrix-Game) for Apple Silicon.
It targets offline rendering: a causal video-to-video renderer that turns a clip into a stylized one
using the quality preset, with the ancient-egypt style supported first.

## Status

| Milestone | Scope | Status |
| --- | --- | --- |
| M1 | ancient-egypt, quality preset (Wan VAE, steps `1000,500`), offline CLI, real-weight parity | Done; the render of a recorded game take is pending |
| M2 | castle and racing (first-frame sink, absolute RoPE, racing RNG role), fixtures for both | Next |
| M3 | realtime preset: TAEHV encode/decode (`taehv.py`), one-step schedule | Next |

## Quick start

```bash
cd Matrix-Game-4-mlx
pip install -e .
export MATRIX_GAME_4_MLX_WEIGHTS_DIR=/path/to/matrix-game-4.0-mlx   # optional, see Weights
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
scene's sink image), `--preset quality` (the only preset in M1).

## Weights

Converted weights are published as `dgrauet/matrix-game-4.0-mlx` (produced by
[mlx-forge](https://github.com/dgrauet/mlx-forge)) and downloaded on first run. To use a local copy, point
`MATRIX_GAME_4_MLX_WEIGHTS_DIR` at the pack directory. Tests marked `needs_weights` are skipped when it is unset.

```text
matrix-game-4.0-mlx/
  config.json                    # DiT config, codecs, per-scene entries (component, inference, sink_image)
  split_model.json               # pinned shared files: repo, revision, sha256
  dit_ancient_egypt.safetensors  # also dit_castle, dit_racing
  taehv.safetensors              # realtime codec (M3)
  fps-egypt/                     # inference.json, sink.png (also fps-castle, driving_game)
```

The umT5 encoder, the Wan 2.2 VAE and the `google/umt5-xxl` tokenizer are the same files as Matrix-Game 3.0.
They are not duplicated: `split_model.json` pins them to `dgrauet/matrix-game-3.0-mlx` at a fixed revision,
and every file is sha256-checked after download. `MATRIX_GAME_4_MLX_SHARED_DIR` points at a local copy of
these files (still sha256-checked).

## Performance

Apple M2 Pro, 32 GB, MLX 0.31.1, ancient-egypt, quality preset, 1280x720 24 FPS input (rendered at 480x864).
A chunk is one latent frame (four video frames, one for the first chunk), with two denoising calls and one
cache-update call. `seconds` excludes model loading (~30 s); `seconds_per_chunk` is `seconds / chunks`,
so it includes video decode, VAE encode/decode and MP4 encoding.

| Run | Frames | Chunks | seconds | s/chunk | Peak memory |
| --- | --- | --- | --- | --- | --- |
| Before the float64 islands (fp32 RoPE and x0) | 6 | 3 | 77.5 | 25.8 | 23.9 GB |
| Current (float64 islands on the CPU stream) | 6 | 3 | 73.6 | 24.5 | 24.4 GB |

Kernel dtype audit (smeltr, 5-frame render): all DiT GEMMs and attention kernels are bf16
(`steel_gemm_fused_*_bfloat16_bfloat16`, `steel_attention_bfloat16_*`), and the VAE convolutions are bf16
(`implicit_gemm_conv_3d_bfloat16`, 42 % of GPU time). The only float32 GEMMs are 48 small calls (0.1 % of
GPU time) in the one-off umT5 attention, where the float32 position bias promotes the scores. The float32
elementwise and reduce kernels are the norm statistics and the torch-opmath activations listed under
Numerics. The float64 CPU-stream work costs about 0.54 s per DiT forward (RoPE ~0.49 s over 30 layers,
timestep MLP ~0.05 s, flow-to-x0 negligible), so 1.1 to 1.6 s per chunk (5 to 7 %).

## Numerics

Upstream's float64 islands (RoPE complex product and tables, flow-to-x0, timestep sinusoid) run in float64,
on the MLX CPU stream where MLX has no GPU float64. bf16 rounding points follow torch's opmath rather than
MLX's per-op defaults: SiLU / tanh-GELU and the VAE `RMS_norm` compute in fp32 and round once, and the
timestep-MLP linears (one row per timestep entry, B x F) accumulate in float64, bias included, and round once
(MLX's single-row bf16 addmm rounds the product before adding the bias, and fp32 summation order flips a few
outputs at any row count). The scheduler's sigma grid reproduces `torch.linspace` bitwise.

## Known deviations from upstream

- float64 runs on the MLX CPU stream (RoPE, flow-to-x0, timestep MLP accumulation) because MLX has no float64
  on the GPU; the values are upstream's, the device is not.
- The umT5 text encoder runs in bf16; upstream runs it in float32 on the CPU. Both cast the prompt
  embeddings to bf16 before the DiT.
- KV-cache end indices are Python ints instead of 1-element tensors.
- `CastleStream.push` has a test-only `renoise` argument.
- Only the inference path (`kv_cache is not None`) is ported; training branches raise `NotImplementedError`.

## Parity methodology

The upstream PyTorch code in `../Matrix-Game-4/` is the oracle. Unit and module tests compare against it
directly (single module fp32 `max_abs < 1e-3`, block `< 5e-3`). Real-weight parity
(`tests/parity/test_real_weights.py`) compares the MLX stream with torch fixtures on three levels:

- **Per call, teacher forced.** At each chunk boundary, torch's exact cache state and inputs are injected and
  one MLX generator call runs. Its pred and the layer 0 / 29 KV it writes are compared with torch. Each gate is
  1.5x the largest rel L2 between torch and three torch kernel variants of the same call (native bf16 SDPA,
  native CPU bf16 without emulation, float64 accumulation), so MLX must stay within the noise torch shows
  against itself. The rest of the cache (history, sink, rolled window), the end indices and `k_sink_raw`
  must match exactly. A `--long` chunk 22 runs past the 21-frame window (rolling eviction, fixed-sink re-RoPE).
- **Per block.** For chunk 0's first denoising call, each of the 30 blocks gets torch's exact input, `e0`,
  context and sink KV; its output must be within 1.5x the largest torch-variant rel L2 on that block. This
  catches a systematic divergence that per-call noise would hide.
- **End to end, chunk 0 strict.** The full MLX stream runs from the same inputs. Chunk 0 is gated with the
  spec thresholds (control, pred and KV rel L2 `< 2e-2`, decoded PSNR `> 35 dB`). The autoregressive stream
  amplifies bf16 rounding chunk over chunk (torch against itself reaches ~0.3 rel L2 by chunk 2), so chunks 1
  and 2 are reported only.

Fixtures are generated with the real upstream model (prompt embeddings come from the MLX umT5, so the
fixtures do not cover the text encoder):

```bash
cd Matrix-Game-4-mlx
MATRIX_GAME_4_MLX_WEIGHTS_DIR=... python -m tests.parity.gen_fixtures --device auto [--long]
```

| Flag | Effect |
| --- | --- |
| `--device {auto,mps,cpu}` | Torch device; `auto` tries MPS first. The DiT falls back to CPU (MPS has no float64), the VAE stays on MPS |
| `--chunks N` | Number of chunks (default 3) |
| `--long` | Also run 25 chunks (pred and decoded) and the chunk-22 per-call fixture |
| `--floor-only` | Regenerate only the torch floor chunks from the saved inputs |
| `--calls-only` | Regenerate only the per-call fixtures |
| `--call-floors-only` | Re-measure only the torch kernel-variant floors of the per-call fixtures |
| `--blocks-only` | Regenerate only the per-block fixture of chunk 0 call 0 and its floors |

They are written to `$MG4_FIXTURES_DIR/egypt_quality/` (default `~/.cache/matrix-game-4-mlx/fixtures`):
about 1.7 GB for the default run and 6.2 GB with `--long` (the chunk-22 per-call fixture alone is 3.8 GB).
The default run takes under 20 minutes, `--long` about 50 minutes. Native torch bf16 GEMMs on the CPU run at
about 1 GFLOPS on Apple Silicon (a 5B forward would take about an hour), so the generator emulates them:
bf16 linear, conv and SDPA calls on the CPU compute in fp32 and round the output to bf16 (fp32-accumulate
kernel semantics). Every other op, including the float64 islands, runs unchanged, and `meta.json` records the
emulation and the device of each part.

## Tests

```bash
cd Matrix-Game-4-mlx
python3 -m pytest tests -v
```

Install the `parity` extra for the upstream oracle. Tests marked `needs_fixtures` read the torch reference
fixtures from `MG4_FIXTURES_DIR`; tests marked `needs_weights` need `MATRIX_GAME_4_MLX_WEIGHTS_DIR`.
