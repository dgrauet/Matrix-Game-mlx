<h1 align="center">Matrix-Game 4.0</h1>
<h3 align="center">Skywork AI</h3>
<p align="center">Streaming Neural Rendering for Playable Games</p>

<p align="center">
  <a href="https://matrix-game-v4-0.github.io/page/">Project Website</a> |
  <a href="https://huggingface.co/Skywork/Matrix-Game-4.0">Models &amp; Game Assets</a> |
  <a href="Matrix-Game-4">Code &amp; Quick Start</a> |
  <a href="#previous-versions">Previous Versions</a>
</p>

**Matrix-Game** is Skywork AI's series of open-source interactive world models.
**Matrix-Game 4.0** combines executable game engines with a causal video model:
the engine handles gameplay, physics and player input, while the model transforms
low-poly RGB frames into detailed visuals through streaming neural rendering.

https://github.com/user-attachments/assets/53a0cfb7-4193-42c8-9819-7c023389a3d7

## News

- **October 15, 2026:** Released **Matrix-Game 4.0**. [Inference, browser demo and game source code](Matrix-Game-4) are now available, with model weights and game assets for Medieval Castle, Ancient Egypt and City Racing on [Hugging Face](https://huggingface.co/Skywork/Matrix-Game-4.0).
- **March 27, 2026:** Released [Matrix-Game 3.0](Matrix-Game-3), a real-time streaming interactive world model with long-horizon memory.
- **August 12, 2025:** Released [Matrix-Game 2.0](Matrix-Game-2), an interactive world foundation model for real-time long video generation.
- **May 12, 2025:** Released [Matrix-Game 1.0](Matrix-Game-1), the first open-source release of the Matrix-Game series.

## Matrix-Game 4.0

- **Engine-grounded gameplay:** game logic and physical state remain in an executable engine, separate from neural rendering.
- **Streaming generation:** a causal, few-step video model renders successive chunks conditioned on low-poly RGB and text prompts.
- **Three scene models:** Medieval Castle and Ancient Egypt for third-person shooting, and City Racing for driving.

Start with the [v4 README](Matrix-Game-4/README.md) for installation, resource
downloads, offline rendering and the online browser demo. The release includes
the three native games and shared single-GPU inference code.

| Scene | Model Weights |
| --- | --- |
| Medieval Castle | [fps-castle](https://huggingface.co/Skywork/Matrix-Game-4.0/tree/main/fps-castle) |
| Ancient Egypt | [fps-egypt](https://huggingface.co/Skywork/Matrix-Game-4.0/tree/main/fps-egypt) |
| City Racing | [driving_game](https://huggingface.co/Skywork/Matrix-Game-4.0/tree/main/driving_game) |

Game assets, sink images and inference configuration files are included in the
[Hugging Face repository](https://huggingface.co/Skywork/Matrix-Game-4.0).
See the [project website](https://matrix-game-v4-0.github.io/page/) for demos and
the broader project overview.

## Matrix-Game 3.0 MLX Port (Apple Silicon)

This repository provides an implementation of Matrix-Game-3.0 for Apple Silicon using [MLX](https://ml-explore.github.io/mlx/).

Models are converted with [mlx-forge](https://github.com/dgrauet/mlx-forge) and hosted on [HuggingFace](https://huggingface.co/dgrauet/matrix-game-3.0-mlx).

## 🚀 Usage

```bash
cd Matrix-Game-3-mlx
pip install -r requirements.txt

python3 generate.py \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --image demo_images/001/image.png
```

Models are downloaded automatically from HuggingFace on first run. To use a local model directory:

```bash
python3 generate.py \
  --model_path /path/to/matrix-game-3.0-mlx \
  --prompt "Your prompt" \
  --image your_image.png \
  --num_iterations 3 \
  --num_inference_steps 3
```

Use `--use_base_model` for the 50-step base model with classifier-free guidance (higher quality, slower).

### VAE variants

`--vae_type` selects the VAE decoder, mirroring the reference's choices and
default:

| `--vae_type` | Weights | Decoder | Decode speed | Quality |
|--------------|---------|---------|--------------|---------|
| `mg_lightvae_v2` (default) | `vae_lightvae_v2.safetensors` | MG-LightVAE v2, pruned 75% | fastest (~30x vs full) | good |
| `mg_lightvae` | `vae_lightvae.safetensors` | MG-LightVAE, pruned 50% | fast | better |
| `wan` | `vae.safetensors` | full Wan2.2 VAE | slow (dominates clip time) | best |

With the LightVAE variants, encoding (image conditioning) still goes through
the full Wan2.2 VAE "teacher" encoder, exactly like the reference. The VAE
runs in bfloat16.

### Interactive mode

```bash
python3 generate.py \
  --prompt "Your prompt" \
  --image your_image.png \
  --interactive \
  --num_iterations 5 \
  --num_inference_steps 3
```

At each clip iteration, you'll be prompted to choose a movement direction (WASD) and camera rotation (IJKL). Each clip generates ~2 seconds of video.

### Lower resolution (32GB machines)

On machines with 32GB RAM, use a lower resolution to avoid out-of-memory errors:

```bash
python3 generate.py \
  --prompt "Your prompt" \
  --image your_image.png \
  --size "960*544" \
  --num_inference_steps 3
```

### Distributed inference (experimental, multi-Mac)

The DiT can be sharded across 2+ Macs with head-wise tensor parallelism over
`mx.distributed` (24 attention heads and the FFN split per rank, combined with
`all_sum` after each block). Launch with `mlx.launch`:

```bash
# 2 Macs over Thunderbolt/Ethernet (ring backend)
mlx.launch --hosts mac1,mac2 --backend ring python generate.py \
  --prompt "Your prompt" \
  --image your_image.png \
  --num_inference_steps 3
```

Notes:

- The number of ranks must divide 24 (heads) and 14336 (ffn_dim): 2, 4, or 8.
- On macOS 26.2+ with a direct Thunderbolt cable, use `--backend jaccl`
  (RDMA, much lower latency than the TCP ring).
- Requires fp16/bf16 weights — quantized weights cannot be sharded.
- `--interactive` is not supported distributed.
- The video is written by rank 0.
- Communication is ~2 `all_sum` (~77 MB each at 720p) per block per step —
  a few seconds per clip on Thunderbolt, negligible next to compute. Expect
  near-linear speedup of the DiT forward pass, which dominates clip time.

This replaces the reference's Ulysses sequence parallelism (`mx.distributed`
has no `all_to_all` collective); see `Matrix-Game-3-mlx/wan/distributed/`.

## ⚠️ Performance & Limitations

### Not real-time

The original Matrix-Game-3.0 achieves real-time performance (40fps) on NVIDIA A100/H100 GPUs using Flash Attention, multi-GPU parallelism, and Distribution Matching Distillation. **This MLX port does not achieve real-time performance on Apple Silicon.**

Typical generation times per 2-second clip (3 denoising steps, distilled
model, default `mg_lightvae_v2` VAE):

| Machine | Resolution | Time per clip |
|---------|-----------|--------------|
| M2 Pro 32GB | 480p (960x544) | ~3 min |
| M2 Pro 32GB | 720p (1280x704) | ~7 min (used to OOM before the bf16/LightVAE fixes) |
| M4 Max 64GB+ | 720p (1280x704) | ~15 min (measured with the full VAE) |

### Why it can't be real-time on Apple Silicon

1. **Attention is O(n²)**: 13,200 patches at 720p, computed across 24 heads and 30 transformer blocks. NVIDIA GPUs use Flash Attention (custom CUDA kernels) which is significantly faster than MLX's `scaled_dot_product_attention`.

2. **Limited multi-device parallelism**: The PyTorch reference uses Ulysses sequence parallelism across 8 datacenter GPUs linked by NVLink (900 GB/s). The experimental tensor-parallel mode (above) gives near-linear speedup across Macs, but Thunderbolt bandwidth and realistic Mac counts (2–4) keep it far from a 40fps target.

3. **Memory bandwidth**: Even an M5 Ultra (~1.2 TB/s) has ~5x less memory bandwidth than an H100 (3.35 TB/s), and the compute gap is larger.

4. **Quantization doesn't help**: Standard int4/int8 quantization produces noise output with diffusion transformers — the accumulated rounding errors across 30 blocks and multiple denoising steps are too large. This is a known limitation of post-training quantization for diffusion models.

### What this port is useful for

- **Offline video generation** from an image + prompt + actions
- **Experimentation** with world model architectures on Apple Silicon
- **Reference implementation** for porting video diffusion models to MLX
- **Interactive exploration** (choose actions between clips, ~5 min wait per clip)

## Previous Versions

Earlier implementations remain available in their original directories:

- [Matrix-Game 3.0](Matrix-Game-3): streaming interaction with long-horizon memory.
- [Matrix-Game 2.0](Matrix-Game-2): real-time interactive long video generation.
- [Matrix-Game 1.0](Matrix-Game-1): the first Matrix-Game world model.

## License

Matrix-Game 4.0 is released under **Apache-2.0**; see the
[v4 license](Matrix-Game-4/LICENSE).
Third-party components and assets retain their respective licenses.
Earlier versions retain their existing license terms; see the repository's
[MIT license](LICENSE) and the notices within each version directory.
