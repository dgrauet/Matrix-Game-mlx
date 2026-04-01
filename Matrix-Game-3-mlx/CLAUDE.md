# CLAUDE.md — Matrix-Game-3-mlx

## Project Overview

MLX inference port of [Matrix-Game-3.0](https://github.com/SkyworkAI/Matrix-Game) — a real-time,
streaming interactive world model that generates 720p first-person video from an image, text prompt,
and keyboard/mouse actions. Built on a 5B Diffusion Transformer (DiT) backbone based on Wan2.2.

This is the **inference-only** MLX runtime. Model conversion/quantization is handled by
[mlx-forge](https://github.com/dgrauet/mlx-forge). Converted models are hosted at
[huggingface.co/dgrauet/matrix-game-3.0-mlx](https://huggingface.co/dgrauet/matrix-game-3.0-mlx).

## Guiding Principles

### Correctness Above All

Every ported module MUST be numerically validated against the PyTorch reference before it is
considered done. "It runs" is not "it works". Write a validation test, compare outputs with tight
tolerances, and only move on when the test passes.

- fp32 tolerance: `atol=1e-4`, `rtol=1e-5`
- fp16/bf16 tolerance: `atol=1e-2`, `rtol=1e-3`
- Test with fixed seeds and deterministic inputs
- Compare intermediate activations, not just final outputs, when debugging mismatches

### Mirror the Reference

The directory structure, file names, class names, and method signatures mirror `Matrix-Game-3/`
as closely as possible. This is intentional — when SkyworkAI updates their implementation, the
diff between their changes and our port should be minimal.

- Same file layout under `wan/`, `pipeline/`, `utils/`
- Same class names: `WanModel`, `WanAttentionBlock`, `WanSelfAttention`, etc.
- Same method names and signatures where feasible
- Deviate only when MLX idioms require it (e.g., channels-last layout, `__call__` vs `forward`)

### Quality and Standards

- Type hints on all functions and methods
- Google-style docstrings for public APIs
- No dead code, no commented-out code, no TODOs left behind
- Each module should be self-contained and testable in isolation
- Prefer clarity over cleverness

## Tech Stack

- Python 3.11+
- MLX >= 0.31.0 (`mlx`, `mlx.nn`, `mlx.core`)
- NumPy (camera math, data processing)
- Pillow (image I/O)
- imageio (video output)
- HuggingFace `tokenizers` (T5 tokenizer)
- HuggingFace `huggingface_hub` (model download)

## MLX Documentation

Official MLX docs: https://ml-explore.github.io/mlx/

Always consult the official docs for API signatures, conventions, and best practices.

## Architecture

```
Matrix-Game-3-mlx/
├── generate.py                    # CLI entry point
├── pipeline/
│   ├── inference_pipeline.py      # MatrixGame3Pipeline
│   └── vae_config.py              # VAE loading config
├── utils/
│   ├── cam_utils.py               # Camera math (numpy)
│   ├── conditions.py              # Predefined action sequences
│   ├── transform.py               # Image preprocessing
│   ├── utils.py                   # Plucker rays, action-to-pose
│   └── visualize.py               # Video output with overlays
├── wan/
│   ├── configs/
│   │   ├── config.py              # Model hyperparameters
│   │   └── shared_config.py       # Shared config
│   └── modules/
│       ├── model.py               # WanModel DiT backbone
│       ├── action_module.py       # ActionModule (keyboard/mouse)
│       ├── attention.py           # Attention (mx.fast.scaled_dot_product_attention)
│       ├── posemb_layers.py       # RoPE (real arithmetic)
│       ├── t5.py                  # T5-XXL text encoder
│       ├── tokenizers.py          # HuggingFace tokenizer wrapper
│       └── vae2_2.py              # Wan2.2 VAE (CausalConv3d)
│   └── utils/
│       ├── fm_solvers.py          # Flow matching DPM solver
│       └── fm_solvers_unipc.py    # UniPC multistep solver
└── tests/
    ├── test_posemb.py             # RoPE validation
    ├── test_attention.py          # Attention validation
    ├── test_t5.py                 # T5 encoder validation
    ├── test_action_module.py      # Action module validation
    ├── test_model.py              # DiT backbone validation
    ├── test_vae.py                # VAE validation
    ├── test_solvers.py            # Solver validation
    └── test_pipeline.py           # End-to-end pipeline validation
```

## Porting Conventions

### PyTorch to MLX Mapping

| PyTorch | MLX |
|---------|-----|
| `nn.Module` + `forward()` | `nn.Module` + `__call__()` |
| `nn.Linear` | `nn.Linear` |
| `nn.LayerNorm` | `nn.LayerNorm` |
| `nn.Conv3d` (NCDHW) | `nn.Conv3d` (NDHWC) |
| `nn.SiLU` | `nn.silu` |
| `nn.GELU` | `nn.gelu` |
| `flash_attn` | `mx.fast.scaled_dot_product_attention` |
| `torch.view_as_complex` / `torch.polar` | Real arithmetic (cos/sin multiply) |
| `einops.rearrange` | Explicit `mx.reshape` / `mx.transpose` |
| `torch.compile` | `mx.compile` where beneficial |
| `diffusers.ModelMixin` | Plain `nn.Module` + `mx.load` |

### Conv3d Layout

- PyTorch weights: `(O, I, D, H, W)` — channels-second
- MLX weights: `(O, D, H, W, I)` — channels-last
- PyTorch input: `(N, C, D, H, W)` -> MLX input: `(N, D, H, W, C)`
- Weight transposition is handled by the mlx-forge recipe

### CausalConv3d Pattern

Wrap `nn.Conv3d` with manual temporal padding via `mx.pad()` before calling the convolution.
The temporal padding ensures causal (no future frame leakage) behavior.

### Patchify Conv3d

Replace patchify `Conv3d(in_dim, dim, patch_size, stride=patch_size)` with
`reshape` + `nn.Linear` for better performance (established MLX best practice).

### Memory Management

- Use `mx.eval()` at strategic points to control peak memory
- Chunk VAE decode along temporal dimension for large sequences
- No model offloading — unified memory handles everything

## What NOT to Port

- `wan/distributed/` — no multi-GPU needed on Apple Silicon
- `wan/triton_kernels.py` — quantization via mlx-forge
- `pipeline/vae_worker.py` — no async multi-GPU VAE
- `pipeline/inference_interactive_pipeline.py` — batch mode only (for now)

## Model Weights

- Source: `Skywork/Matrix-Game-3.0` on HuggingFace
- Converted: `dgrauet/matrix-game-3.0-mlx` on HuggingFace
- Conversion tool: [mlx-forge](https://github.com/dgrauet/mlx-forge)
- Format: per-component safetensors (`dit.safetensors`, `t5_encoder.safetensors`, etc.)

## Testing

Every module has a corresponding validation test in `tests/`. Tests load identical weights into
both PyTorch and MLX implementations, feed identical inputs, and assert numerical equivalence.

Run tests: `python -m pytest tests/ -v`

## Critical Rules

- NEVER skip validation tests — every module must pass before moving on
- NEVER deviate from reference class/method names without documenting why
- ALWAYS check the MLX docs (https://ml-explore.github.io/mlx/) for API details
- ALWAYS port bottom-up: dependencies before dependents
- ALWAYS use fixed seeds for reproducible test inputs
