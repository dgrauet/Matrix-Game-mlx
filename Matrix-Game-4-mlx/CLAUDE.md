# CLAUDE.md — Matrix-Game-4-mlx

## Project Overview

MLX inference port of Matrix-Game 4.0 (`../Matrix-Game-4/`, SkyworkAI/Matrix-Game). We port the
causal video-to-video renderer for offline rendering on Apple Silicon, with a quality preset (Wan VAE) and a realtime
preset (TAEHV codec, one denoising step). Model conversion is handled by
[mlx-forge](https://github.com/dgrauet/mlx-forge); the converted pack is published as `dgrauet/matrix-game-4.0-mlx`
(pinned revision, auto-downloaded) or read locally from `MATRIX_GAME_4_MLX_WEIGHTS_DIR`. Inference only.

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
at load. TAEHV (realtime codec) keeps bf16 parameters, but its convs accumulate in float32 and round once to bf16
(MLX's bf16 Metal conv2d loses precision on some kernel paths). fp32 islands, as upstream: RMSNorm/LayerNorm statistics, scheduler `add_noise`, and torch's
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
- Realtime DiT runs upstream's `native` runtime (not the CUDA-only `ring-fused`), no CUDA VAE graph (`vae_graph`)
- TAEHV convs accumulate in float32 and round once to bf16 (torch semantics)

## Validation Status

Ancient-egypt, castle and racing are parity-validated (quality preset) with the same methodology: per-block and
per-call teacher-forced gates at 1.5x the torch kernel-variant floor, strict end-to-end on chunk 0 (racing: chunks
0 and 1, since its first_frame sink makes chunk 0 the sink latent; chunk 1 pred gated at 1.5x the torch floor). Fixtures: `python -m tests.parity.gen_fixtures
--scene {ancient-egypt,castle,racing}` into `$MG4_FIXTURES_DIR/{egypt,castle,racing}_quality`; the real-weight
suite runs per scene whose fixtures exist, one DiT loaded at a time. Racing's torch DiT runs on the CPU.

The realtime preset (M3) is parity-validated on the three scenes against fixtures `$MG4_FIXTURES_DIR/{egypt,castle,racing}_realtime`
(`python -m tests.parity.gen_fixtures --scene {ancient-egypt,castle,racing} --preset realtime`, ~10 min and ~200 MB per
scene; `--taehv-only` ~20 s and `--floor-only` partial modes). TAEHV gates are MLX rel L2 vs torch at
ceil_1sig(1.5 x max torch variant); end to end, chunk 0 is gated (racing chunks 0 and 1) and the pred gate is
max(2e-2, 1.5 x the torch floor of the chunk), since with one step torch against itself already exceeds 2e-2. Renders
(97-frame pans, M2 Pro 32 GB): about 3.5 s/chunk on all three scenes, peak 14.96 GB. Default preset stays `quality`.

## File Map

```text
matrix_game_mlx/
  scenes.py                      # scene profiles (prompts, sink policy), model-loading parts only
  configs/                       # ancient-egypt.json, castle.json, racing.json (scene name and prompt)
  cli.py                         # `matrix-game-mlx` console script: render, download, verify (serve not ported)
  inference/
    offline.py                   # `render`: --input clip | --take recording -> rendered.mp4 + render.json; dynamic captions
    loader.py                    # ModelComponents: pack dir, pinned shared MG3 files (sha256), T5 -> DiT -> VAE
    loading.py                   # strict DiT build from the mlx-forge pack
    stream.py                    # CastleStream: sink priming, chunk push (1-2 denoising calls per preset + cache update)
    wrappers.py                  # WanDiffusionWrapper (flow -> x0 in float64), WanTextEncoder, WanVAE22Wrapper, KV caches
    decoder.py                   # incremental Wan VAE decoding with persistent causal features
    taehv.py                     # TAEHV encode/decode (realtime codec), float32-accumulating convs
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
  parity/test_real_weights_variants.py  # prompt switch, no sink, TAEHV 2-step, Wan custom first-steps vs torch
  parity/test_real_weights_captions.py  # caption embeddings vs torch
  parity/test_gen_fixtures.py    # generator helper checks
  test_cli_smoke.py              # needs_weights: realtime dynamic render through the CLI (peak < 30 GB)
  racing_game.py                 # helpers: locate/build-check/run the upstream racing game, scripted take (env MG4_RESOURCES, MG4_RACING_TAKE)
  test_racing_game.py            # needs_racing_game: headless smoke of the built upstream game (free, race, pursuit)
  test_racing_take.py            # scripted take: input script, complete and finalizable, reproducible
  test_racing_take_render.py     # realtime and dynamic-caption renders of the take finalize; captions file matches the take (needs weights)
  data/racing_captions.jsonl     # hand-made 9-frame captions (3 records)
  data/racing/take_input_script.json  # scripted racing take inputs ({frame, key, down}; 720 frames at 48 FPS)
  data/racing/take_captions.jsonl     # offline captions for the racing take (181 records covering the 720 frames)
```

## Tech Stack

Python 3.11+, MLX >= 0.31.0, NumPy, Pillow, imageio-ffmpeg, HuggingFace `tokenizers`/`transformers`/
`huggingface_hub`/`safetensors`. Parity extra: torch, diffusers, einops.

## Testing

Fast suite: `python -m pytest -m "not needs_weights and not needs_fixtures"`. Variant fixtures: `python -m tests.parity.gen_fixtures --variant NAME` / `--captions-only` (`--print-variant-commands` lists them). CLI: `matrix-game-mlx render|download|verify`; quality + dynamic captions needs more than 32 GB (T5 resident).

Racing game tests carry the marker `needs_racing_game` (built game in `../Matrix-Game-4/game/racing/bin`, assets under
`$MG4_RESOURCES/racing`, default `../Matrix-Game-4/resources`; the recorded take is `$MG4_RACING_TAKE`, default
`$MG4_RESOURCES/racing/captures/take-001`). Parity variants `racing_take_realtime` / `racing_take_quality`
(`python -m tests.parity.gen_fixtures --variant NAME`) use the first 9 frames of that take (`VariantSpec.take`,
`take_frames`); fixtures `$MG4_FIXTURES_DIR/racing_take_{realtime,quality}`. See README "City racing on macOS".

Run from this directory: `/opt/homebrew/bin/python3.14 -m pytest tests -v` (the MLX/torch stack is
installed for Homebrew Python 3.14)

`tests/conftest.py` provides `import_upstream`, `to_np`, `assert_close`, `rel_l2`, `psnr`,
`fixtures_dir`, `weights_dir`, `UPSTREAM_ROOT`, and the markers `needs_weights` (env
`MATRIX_GAME_4_MLX_WEIGHTS_DIR`) and `needs_fixtures` (env `MG4_FIXTURES_DIR`, default
`~/.cache/matrix-game-4-mlx/fixtures`); marked tests skip when their inputs are missing.

## What NOT to Port

- `serve` (the server, browser demo); the game is upstream's native racing game, built and used as is (README "City racing on macOS")
- `rtx5090*` files
- Ring / fused / cudagraph runtimes (realtime uses the `native` DiT runtime) and the CUDA VAE graph
- Training branches (teacher forcing, ctx slots, kv_mem, kv_capture, flex_attention masks,
  sequence parallel)

## Critical Rules

- NEVER skip validation tests; port bottom-up (dependencies before dependents)
- NEVER deviate from reference names without documenting why
- ALWAYS check https://ml-explore.github.io/mlx/ for API details
