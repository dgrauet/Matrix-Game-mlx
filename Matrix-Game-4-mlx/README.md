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

## Known deviations from upstream

- float64 is replaced by fp32 in `_convert_flow_pred_to_x0` and in the RoPE complex product (the MLX GPU
  has no float64); RoPE tables are built in numpy float64 and then cast to fp32.
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
