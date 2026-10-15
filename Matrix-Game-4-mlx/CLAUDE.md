# CLAUDE.md — Matrix-Game-4-mlx

## Project Overview

MLX inference port of Matrix-Game 4.0 (`../Matrix-Game-4/`, SkyworkAI/Matrix-Game). We port the
causal video-to-video renderer for offline rendering on Apple Silicon. Model conversion is handled by
[mlx-forge](https://github.com/dgrauet/mlx-forge); converted weights live at
`dgrauet/matrix-game-4.0-mlx`. Inference only.

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
without autocast, unlike MG3). The VAE is stored fp32 and cast to bf16 at load. fp32 islands, as
upstream: RMSNorm/LayerNorm statistics, sinusoidal timestep embedding, RoPE rotation, scheduler
`add_noise`.

### Layout

Channels-last internally (`[F, H, W, C]` per sample in the DiT, `[B, T, H, W, C]` in the VAE);
wrapper-level APIs keep upstream shapes `[B, F, C, h, w]`.

## Known Deviations (documented in README)

- float64 -> fp32 for `_convert_flow_pred_to_x0` and the RoPE complex product (no float64 on the MLX
  GPU); RoPE tables are built in numpy float64 then cast to fp32
- KV-cache end indices are Python ints instead of 1-element tensors
- Test-only `renoise` argument on `CastleStream.push`

## Tech Stack

Python 3.11+, MLX >= 0.31.0, NumPy, Pillow, imageio-ffmpeg, HuggingFace `tokenizers`/`transformers`/
`huggingface_hub`/`safetensors`. Parity extra: torch, diffusers, einops.

## Testing

Run from this directory: `/opt/homebrew/bin/python3 -m pytest tests -v`

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
