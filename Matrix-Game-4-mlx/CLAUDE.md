# CLAUDE.md — Matrix-Game-4-mlx

## Project Overview

MLX inference port of Matrix-Game 4.0 (`../Matrix-Game-4/`, SkyworkAI/Matrix-Game; reference commit `51913b2`). We port the
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

Exception, port-only: `--dit-runtime apple` / `apple-fast` (and `--preset apple` / `apple-fast`) run the DiT in
float16 (weights converted from the bf16 pack at load, one tensor at a time, so two DiT copies never coexist;
activations, modulation, self- and cross-attention K/V caches in fp16). Every precision island above stays as is
(fp32 norm statistics, opmath SiLU / tanh-GELU, float64 timestep MLP and flow-to-x0, RoPE as a float64 rotation
rounded once to fp16). T5 and TAEHV stay bf16: the DiT casts the bf16 prompt embeddings and the bf16 control latent
to fp16 where it consumes them, and the stream casts the fp16 pred to the VAE dtype (bf16) before decode.

### Layout

Channels-last internally (`[F, H, W, C]` per sample in the DiT, `[B, T, H, W, C]` in the VAE);
wrapper-level APIs keep upstream shapes `[B, F, C, h, w]`.

## Known Deviations (documented in README)

- Sink policy: racing runs upstream's `future` sink RoPE by default (`inference/sink_policy.py` mirrors upstream's: `SINK_ROPE_MODES = (absolute, fixed, future)`, `FUTURE_SINK_OFFSET = 11`; sink keys rotated to position + 11 once position > window); shooters `fixed`; `--sink-rope-mode` overrides on every scene; the sink image is mandatory (no `--no-sink-image`, no no-sink fixture variants); unknown modes raise. `apple-fast` applies `FUTURE_SINK_OFFSET = 11` once its own 11-latent window is full (position > 11): unvalidated by upstream, no torch gate (bitwise tiny-model test at window 11, fp16 safety, free-running comparison). The `rtx5080*` / `rtx5070ti*` names are CUDA-only and rejected
- float64 islands run on the MLX CPU stream instead of the GPU
- KV-cache end indices are Python ints instead of 1-element tensors
- Test-only `renoise` argument on `CastleStream.push`
- `--dit-runtime ring-fused` (realtime default, as upstream; `native` stays the reference on every preset) is the MLX ring-KV + GPU-RoPE + fused-kernel runtime (`inference/runtime.py`, `inference/kernels.py`): no CUDA graphs/varlen, caches/modulation in the runtime's dtype required (bf16 for `ring-fused`), all layers validated before planning; one `mx.compile`d steady-window graph per session instead of a CUDA graph (`prime` at stream creation, `clear` on close, upstream's `use_graph` flag, bitwise eager), port-only timestep-MLP memo per t value (`time_memo`); the fused Metal kernels (`ln_modulate`, `gate_ln_affine`, `gate_residual`, plus port-only `rms_norm` and compiled `gelu_tanh`) reproduce native rounding and MLX's row-reduction order at the model widths (bitwise native), so it differs from `native` only by SDPA key order (ring), the reduction-order change upstream documents for its fast runtimes ("Keep `native` for reference checks."); on the parity fixtures (all pre-window but the teacher-forced `--long` chunk 22) only the chunk-22 per-call metrics differ, within the unchanged gates; free running past the window the streams diverge (`tests/compare_runtimes.py`, 48 chunks, chunks 21-47: castle pred rel L2 median/max 0.47/0.62, decoded PSNR min 16.1 dB; racing 0.081/0.15, 30.6 dB under the `future` sink), as much as a one-8-bit-level input change on `native` does
- RoPE under `ring-fused` / `apple*` is a Metal kernel (`rope_kernel.py`, upstream: Triton): float64 bits in a uint64 table, float64 product emulated in 64-bit integer soft-float, rounded as MLX's CPU float64 cast to the input dtype: bf16 through float32 (double rounding), fp16 once (RNE with fp16 subnormals and overflow to inf; fp16 NaN payloads kept as the CPU does); both bitwise the CPU float64 path (`tests/test_rope_kernel.py`, every finite pattern of each dtype); no float32 rotation is used; the fixed-sink re-RoPE uses `build_causal_rope_rotations(..., position - window)` as upstream (`build_rope_rotations_frames` is a test-only helper)
- Fused kernels: upstream's Triton kernels differ from the unfused block by row-reduction summation order; ours copy MLX's `row_reduce_simple` layout and are bitwise at the model widths on MLX 0.31.1 (summation order is the only possible difference elsewhere); the `half` variants (apple runtimes) hold the same contract against the native fp16 expressions (`*_fp16` tests in `tests/test_kernels.py`); the bf16 kernels are unchanged
- `mx.compile` replaces CUDA graphs (DiT steady window, TAEHV, T5); on MLX 0.31.1 it is not bitwise for division by a non-power-of-two constant or `mean()`, so every compiled graph is checked bitwise against eager (DiT: real weights, one steady position, castle `fixed` and racing `future`, denoise + cache update, `ring-fused` and `apple`, K/V equal)
- `--vae-graph` (default on for realtime + TAEHV, as upstream): `inference/vae_graph.py` `StreamingVAEGraph` replays one `mx.compile`d steady encoder chunk and decoder latent (state in, state out; first chunk eager; primed at stream creation), bitwise eager
- `--t5-runtime graph` (default): `CaptionEncoder` runs T5 as one `mx.compile`d graph (upstream: CUDA graph), bitwise native; it relies on two t5.py port edits (relative-position buckets computed eagerly and cached, `T5LayerNorm` variance as `sum / dim`), since compiled division and fused mean() differ from eager
- Port-only `--preset apple` / `apple-fast` (`--dit-runtime apple` / `apple-fast` on any preset, as upstream's `rtx5090`): `RUNTIME_SPECS` in `runtime.py` (`ring-fused` bf16/21, `apple` fp16/21, `apple-fast` fp16/11), `dit_dtype(mode)` picks the DiT dtype the loader builds, `CastleStream` takes the window from the installed runtime (21 for `native`); a mismatch (apple on a bf16 DiT, bf16 caches) is a `ValueError`, never a fallback. fp16 DiT is a precision change, not only a reduction order: free running vs `native` (48 chunks, chunks 21-47) castle pred rel L2 median/max 0.59/0.65, PSNR min 16.0 dB; racing 0.45/0.52, 19.8 dB (racing's one-level perturbation baseline under `future` is 0.097/0.15, 29.1 dB, so racing ends far above it; castle's is 0.46/0.59, 17.0 dB). Window 11: attention covers the sink, the current latent and the 9 latents before it (window 21: 19; a realtime latent is 4 video frames), so from stream position 11 on `apple-fast` sees less history than `apple`; bitwise `apple` before that. Casts: T5/TAEHV/Wan VAE stay bf16; the DiT casts prompt and control to fp16, the stream casts the sink to fp16 and the pred to the VAE dtype before decode
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

Apple fast path (`ring-fused`): every real-weight parity slice passes on both runtimes with thresholds and fixtures
unchanged (`pytest tests/parity/test_real_weights<slice>.py --dit-runtime {native,ring-fused}`, one file per process,
one process at a time).
Measured rates and per-lever gains: README "Apple fast path" and "Measured rates" (castle realtime `serve` `--fps 4`:
`online_fps` 0.95 native -> 1.13 ring-fused). The DiT is now ~94 % GEMM + SDPA per call; further speed at stage-1
precision would need faster GEMM/attention kernels. Heavy runs (real weights, parity, serve) one at a time.

Apple presets (`apple`, `apple-fast`, port-only, not upstream-exact): `tests/parity/test_real_weights_apple.py` (one
scene per process, `-k castle|ancient-egypt|racing`, ~4 min and up to 22.9 GB process footprint each) gates the
realtime fixtures' gated chunks a priori (pred rel L2 <= max(6e-2, 2 x realtime gate), PSNR >= 30 dB, control < 2e-2):
castle chunk 0 0.0246 / 41.8 dB, egypt 0.0308 / 41.9 dB, racing chunk 0 = fp16 sink cast bitwise, chunk 1 0.0244 /
47.8 dB; `apple-fast` bitwise `apple` on the fixtures; fp16 safety over 32 free-running chunks per scene and runtime
(all finite, residual max 790 < 65504/16). Castle `tests.bench_dit` 32 chunks: chunk median 3358 / 2887 / 2654 ms
(ring-fused / apple / apple-fast), MLX peak 14.86 / 14.86 / 13.37 GB. Serve `--fps 4` no browser: castle
`online_fps` 1.13 / 1.25 / 1.43 at 16.2 / 16.2 / 15.2 GB, racing (`future` sink) 1.13 / 1.33 / 1.52 at 16.6 / 16.6 / 15.5 GB.

## File Map

```text
matrix_game_mlx/
  scenes.py                      # scene profiles (prompts, sink policy), model-loading parts only
  configs/                       # ancient-egypt.json, castle.json, racing.json (scene name and prompt)
  cli.py                         # `matrix-game-mlx` console script: render, download, verify, serve
  profiling.py                   # bounded opt-in per-chunk wall-clock profiling (ChunkTimer) of the online producer, `--profile-chunks`
  server/                        # online `serve` (upstream copies, macOS changes marked `# macOS port:` in app.py)
    app.py                       # aiohttp app, Runtime/Session, `main`; first_frame_timeout, Apple software renderer check
    game.py, racing.py           # native game IPC (CastleReferenceServer, CitySampleReferenceServer) and racing controls
    reference_input.py           # per-chunk reference preprocessing for the stream
    media_env.py                 # sidecar interpreter (gi + numpy + GStreamer) and serve.sh's environment defaults
    transport/                   # GStreamer/MediaMTX publisher sidecar (output, publisher, rtsp)
  web/                           # upstream browser client, byte-identical (mediamtx-reader.LICENSE ships via license-files)
  inference/
    offline.py                   # `render`: --input clip | --take recording -> rendered.mp4 + render.json; dynamic captions
    loader.py                    # ModelComponents: pack dir, pinned shared MG3 files (sha256), T5 -> DiT -> VAE
    loading.py                   # strict DiT build from the mlx-forge pack
    stream.py                    # CastleStream: sink priming, chunk push (1-2 denoising calls per preset + cache update)
    wrappers.py                  # WanDiffusionWrapper (flow -> x0 in float64), WanTextEncoder, WanVAE22Wrapper, KV caches
    decoder.py                   # incremental Wan VAE decoding with persistent causal features
    taehv.py                     # TAEHV encode/decode (realtime codec), float32-accumulating convs
    vae_graph.py                 # --vae-graph: StreamingVAEGraph (compiled steady TAEHV encode/decode, prime)
    text.py                      # CaptionEncoder (dynamic captions); --t5-runtime graph runs T5 as one compiled graph
    scheduler.py                 # flow-matching scheduler, bitwise torch.linspace sigma grid
    rollout.py                   # warp_steps, set_attention_window
    runtime.py                   # --dit-runtime ring-fused / apple / apple-fast (RUNTIME_SPECS: dtype, window): plan_cache (ring KV), to/from_linear, run_blocks, CastleRuntime (compiled steady graph, prime/clear, time-MLP memo), install_runtime
    rope_kernel.py               # GPU RoPE kernel (bf16 or fp16 output), bitwise equal to the float64 CPU path
    kernels.py                   # fused Metal block-glue kernels (bf16 ring-fused, fp16 apple*), bitwise equal to the native block pieces
    scopes.py                    # optional smeltr scopes (encode/denoise/clean_kv/decode in CastleStream.push); inert without smeltr
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
  parity/test_real_weights_variants.py  # prompt switch, TAEHV 2-step, Wan custom first-steps vs torch
  parity/test_real_weights_captions.py  # caption embeddings vs torch
  parity/test_real_weights_apple.py     # apple presets: a priori gates on the realtime fixtures, apple-fast == apple, fp16 safety
  parity/test_gen_fixtures.py    # generator helper checks
  test_cli_smoke.py              # needs_weights: realtime dynamic render through the CLI (peak < 30 GB)
  media.py                       # media_missing(), mediamtx_server(): MediaMTX for the online tests
  serve_process.py               # serve(args, runtime): the serve CLI as a child process, MemoryMonitor (footprint incl. Metal)
  measure_serve.py               # not collected: python3 -m tests.measure_serve, one measured online session (README "Measured rates"); --dit-runtime pass-through
  bench_dit.py                   # not collected: python3 -m tests.bench_dit, CastleStream per-stage timings, --dit-runtime, --json (README "Apple fast path")
  compare_runtimes.py            # not collected: python3 -m tests.compare_runtimes, free-running stream dump per runtime (meta: DiT dtype, window) + --compare (README "Free-running streams past the window", "Apple presets"); --perturb-frame baseline
  test_rope_kernel.py, test_kernels.py, test_runtime.py, test_runtime_plan.py  # ring-fused: bitwise RoPE/kernels vs native, ring planner, runtime vs native, compiled vs eager
  test_serve_app.py, test_display.py, test_media_environment.py  # serve app, display and media environment (fakes, no model)
  test_online_games.py           # needs_castle_game/needs_racing_server: game IPC handshakes, stream-vs-serial frame equality
  test_transport.py              # needs_media: real sidecar and MediaMTX
  test_serve_e2e.py              # real serve sessions over HTTP/ws (reference, realtime, dynamic, quality, disconnect)
  test_browser_upstream.py       # needs_playwright: upstream tests/browser/*.cjs unmodified
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

Fast suite: `python -m pytest -m "not needs_weights and not needs_fixtures"`. Variant fixtures: `python -m tests.parity.gen_fixtures --variant NAME` / `--captions-only` (`--print-variant-commands` lists them). CLI: `matrix-game-mlx render|download|verify|serve`; quality + dynamic captions needs more than 32 GB (T5 resident).

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

- The games are upstream's native games, built and used as is (README "City racing on macOS", "Online play on macOS")
- `rtx5090*` files
- The CUDA-only DiT runtime names (`kernels`, `cudagraph`, `ring`, `ring-graph`: rejected); `ring-fused` is ported on MLX (`runtime.py`, `rope_kernel.py`, `kernels.py`), and the CUDA VAE / T5 graphs are `mx.compile` (`vae_graph.py`, `text.py`)
- Training branches (teacher forcing, ctx slots, kv_mem, kv_capture, flex_attention masks,
  sequence parallel)

## Critical Rules

- NEVER skip validation tests; port bottom-up (dependencies before dependents)
- NEVER deviate from reference names without documenting why
- ALWAYS check https://ml-explore.github.io/mlx/ for API details
