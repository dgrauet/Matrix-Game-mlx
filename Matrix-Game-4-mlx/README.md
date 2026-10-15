# Matrix-Game-4-mlx

MLX inference port of [Matrix-Game 4.0](https://github.com/SkyworkAI/Matrix-Game) for Apple Silicon.
It targets offline rendering: a causal video-to-video renderer that turns a clip into a stylized one
using the quality preset, with the ancient-egypt style supported first.

Status: work in progress (milestone 1).

## Weights

Converted weights are published as `dgrauet/matrix-game-4.0-mlx` (produced by
[mlx-forge](https://github.com/dgrauet/mlx-forge)). To use a local copy, point the
`MATRIX_GAME_4_MLX_WEIGHTS_DIR` environment variable at the directory containing `config.json`.
Tests marked `needs_weights` are skipped when it is unset.

## Numerics

Upstream's float64 islands (RoPE complex product and tables, flow-to-x0, timestep sinusoid) run in float64,
on the MLX CPU stream where MLX has no GPU float64. bf16 rounding points follow torch's opmath rather than
MLX's per-op defaults: SiLU / tanh-GELU and the VAE `RMS_norm` compute in fp32 and round once, and the M=1
timestep-MLP linears add the bias before rounding (MLX's single-row addmm rounds the product first). The
scheduler's sigma grid reproduces `torch.linspace` bitwise.

## Known deviations from upstream

- KV-cache end indices are Python ints instead of 1-element tensors.
- `CastleStream.push` has a test-only `renoise` argument.

## Tests

```bash
cd Matrix-Game-4-mlx
python3 -m pytest tests -v
```

Parity tests compare against the upstream PyTorch code in `../Matrix-Game-4/` (install the `parity`
extra). Tests marked `needs_fixtures` read torch reference fixtures from `MG4_FIXTURES_DIR`
(default `~/.cache/matrix-game-4-mlx/fixtures`).
