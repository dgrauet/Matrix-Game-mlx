# CLAUDE.md — Matrix-Game-4-mlx

## Project Overview

MLX inference port of Matrix-Game 4.0 (`../Matrix-Game-4/`, SkyworkAI/Matrix-Game). We port the
causal video-to-video renderer for offline rendering on Apple Silicon. Model conversion is handled by
[mlx-forge](https://github.com/dgrauet/mlx-forge); the converted pack is local for now
(`MATRIX_GAME_4_MLX_WEIGHTS_DIR`), its Hub publication as `dgrauet/matrix-game-4.0-mlx` is planned. Inference only.

## Guiding Principles

### Correctness Above All

Every ported module MUST be validated against the upstream PyTorch code before it is done.

- Single module fp32: `max_abs < 1e-4` (ideal), `< 1e-3` accepted
- Block: `< 5e-3`
- Full DiT pass in bf16: relative L2 `< 2e-2`
- Decoded frames: PSNR `> 35 dB`
- Fixed seeds, deterministic inputs; compare intermediate activations when debugging

### Mirror the Reference

File paths, class/function/variable names and call order mirror `Matrix-Game-4/matrix_game/`, so
upstream changes diff minimally. Only the inference path (`kv_cache is not None`) is ported; training
branches raise `NotImplementedError`. Deviate only when MLX idioms require it.

### Package Rename

The upstream package is `matrix_game`; ours is `matrix_game_mlx`. The rename lets both be importable
in the same test process (the tests import upstream as the oracle via `tests/conftest.py`).

### Dtype Policy

DiT and T5 run in bf16 end to end, with no fp32 residual stream (upstream casts the whole DiT to bf16
without autocast, unlike MG3; upstream's loader also casts the T5 to bf16). The VAE is stored fp32 and cast to bf16
at load. fp32 islands, as upstream: RMSNorm/LayerNorm statistics, scheduler `add_noise`, and torch's
opmath for SiLU / tanh-GELU / VAE `RMS_norm`. float64 islands, as upstream, on the MLX CPU stream (no
float64 on the GPU): RoPE tables and complex product, timestep sinusoid and timestep-MLP accumulation,
flow-to-x0. A float32 GEMM or attention kernel in a smeltr op summary is a leak; the CPU-stream float64
work is intended.

### Layout

Channels-last internally (`[F, H, W, C]` per sample in the DiT, `[B, T, H, W, C]` in the VAE);
wrapper-level APIs keep upstream shapes `[B, F, C, h, w]`.

## Known Deviations (documented in README)

- float64 islands run on the MLX CPU stream instead of the GPU
- KV-cache end indices are Python ints instead of 1-element tensors
- Test-only `renoise` argument on `CastleStream.push`

## Validation Status

Ancient-egypt, castle and racing are parity-validated (quality preset) with the same methodology: per-block and
per-call teacher-forced gates at 1.5x the torch kernel-variant floor, strict end-to-end on chunk 0 (racing: chunks
0 and 1, since its first_frame sink makes chunk 0 the sink latent). Fixtures: `python -m tests.parity.gen_fixtures
--scene {ancient-egypt,castle,racing}` into `$MG4_FIXTURES_DIR/{egypt,castle,racing}_quality`; the real-weight
suite runs per scene whose fixtures exist, one DiT loaded at a time. Racing's torch DiT runs on the CPU.

## File Map

```text
matrix_game_mlx/
  scenes.py                      # scene profiles (prompts, sink policy), model-loading parts only
  configs/                       # ancient-egypt.json, castle.json, racing.json (scene name and prompt)
  inference/
    offline.py                   # CLI: --input clip | --take recording -> rendered.mp4 + render.json
    loader.py                    # ModelComponents: pack dir, pinned shared MG3 files (sha256), T5 -> DiT -> VAE
    loading.py                   # strict DiT build from the mlx-forge pack
    stream.py                    # CastleStream: sink priming, chunk push (2 denoising calls + cache update)
    wrappers.py                  # WanDiffusionWrapper (flow -> x0 in float64), WanTextEncoder, WanVAE22Wrapper, KV caches
    decoder.py                   # incremental Wan VAE decoding with persistent causal features
    scheduler.py                 # flow-matching scheduler, bitwise torch.linspace sigma grid
    rollout.py                   # warp_steps, set_attention_window
    preprocessing.py             # RGB crop/resize and tensor <-> uint8
    wan/modules/
      causal_model.py            # causal DiT, rolling KV cache, fixed-sink RoPE, float64 timestep MLP
      model.py                   # embeddings, float64 RoPE, norms, opmath SiLU/GELU, cross-attention
      attention.py               # MLX SDPA wrapper
      t5.py, tokenizers.py       # umT5-XXL encoder (from MG3-mlx)
      vae2_2.py                  # Wan 2.2 VAE (streaming encode/decode)
tests/
  conftest.py                    # upstream import, helpers, needs_weights / needs_fixtures markers
  test_*.py                      # module parity vs upstream torch (no weights needed unless marked)
  parity/gen_fixtures.py         # torch fixture generator (python -m tests.parity.gen_fixtures)
  parity/test_real_weights.py    # per-call, per-block and end-to-end real-weight parity
  parity/test_gen_fixtures.py    # generator helper checks
```

## Tech Stack

Python 3.11+, MLX >= 0.31.0, NumPy, Pillow, imageio-ffmpeg, HuggingFace `tokenizers`/`transformers`/
`huggingface_hub`/`safetensors`. Parity extra: torch, diffusers, einops.

## Testing

Run from this directory: `/opt/homebrew/bin/python3.14 -m pytest tests -v` (the MLX/torch stack is
installed for Homebrew Python 3.14)

`tests/conftest.py` provides `import_upstream`, `to_np`, `assert_close`, `rel_l2`, `psnr`,
`fixtures_dir`, `weights_dir`, `UPSTREAM_ROOT`, and the markers `needs_weights` (env
`MATRIX_GAME_4_MLX_WEIGHTS_DIR`) and `needs_fixtures` (env `MG4_FIXTURES_DIR`, default
`~/.cache/matrix-game-4-mlx/fixtures`); marked tests skip when their inputs are missing.

## What NOT to Port

- The server and the game/interactive layers
- `rtx5090*` files
- Ring / fused / cudagraph runtimes
- Training branches (teacher forcing, ctx slots, kv_mem, kv_capture, flex_attention masks,
  sequence parallel)

## Critical Rules

- NEVER skip validation tests; port bottom-up (dependencies before dependents)
- NEVER deviate from reference names without documenting why
- ALWAYS check https://ml-explore.github.io/mlx/ for API details
