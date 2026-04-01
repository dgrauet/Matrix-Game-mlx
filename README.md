<p align="center">
<h1 align="center">Matrix-Game</h1>
<h3 align="center">Skywork AI</h3>
</p>

## 🔥🔥🔥 News!!
* March 27, 2026: 🔥 We released [Matrix-Game-3.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3). This  is a real-time and streaming interactive world model with long-horizon Memory.
* Aug 12, 2025: 🔥 We released [Matrix-Game-2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2). This is an interactive world foundation model for real-time long video generation.
* May 12, 2025: 🔥 We released [Matrix-Game-1.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-1). The first open-source release of Skywork AI's Matrix-Game series world models.


## 📝 Overview

**Matrix-Game** is a series of open-source world models launched by Skywork AI.

This repository provides an implementation of Matrix-Game-3.0 for Apple Silicon using [MLX](https://ml-explore.github.io/mlx/).

Models are converted with [mlx-forge](https://github.com/dgrauet/mlx-forge) and hosted on [HuggingFace](https://huggingface.co/dgrauet/matrix-game-3.0-mlx).

https://github.com/user-attachments/assets/f5387c64-1d18-414a-935f-00a1f6eec8de

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

## ⚠️ Performance & Limitations

### Not real-time

The original Matrix-Game-3.0 achieves real-time performance (40fps) on NVIDIA A100/H100 GPUs using Flash Attention, multi-GPU parallelism, and Distribution Matching Distillation. **This MLX port does not achieve real-time performance on Apple Silicon.**

Typical generation times per 2-second clip (3 denoising steps, distilled model):

| Machine | Resolution | Time per clip |
|---------|-----------|--------------|
| M4 Max 32GB | 480p (960x544) | ~5 min |
| M4 Max 32GB | 720p (1280x704) | OOM |
| M4 Max 64GB+ | 720p (1280x704) | ~15 min |

### Why it can't be real-time on Apple Silicon

1. **Attention is O(n²)**: 13,200 patches at 720p, computed across 24 heads and 30 transformer blocks. NVIDIA GPUs use Flash Attention (custom CUDA kernels) which is significantly faster than MLX's `scaled_dot_product_attention`.

2. **No multi-device parallelism**: The PyTorch reference uses Ulysses sequence parallelism to split patches across multiple GPUs. MLX runs on a single device. Network-based parallelism (e.g., Exo Labs) would be bottlenecked by inter-machine communication at every attention block (30x per denoising step).

3. **Memory bandwidth**: Even an M5 Ultra (~1.2 TB/s) has ~5x less memory bandwidth than an H100 (3.35 TB/s), and the compute gap is larger.

4. **Quantization doesn't help**: Standard int4/int8 quantization produces noise output with diffusion transformers — the accumulated rounding errors across 30 blocks and multiple denoising steps are too large. This is a known limitation of post-training quantization for diffusion models.

### What this port is useful for

- **Offline video generation** from an image + prompt + actions
- **Experimentation** with world model architectures on Apple Silicon
- **Reference implementation** for porting video diffusion models to MLX
- **Interactive exploration** (choose actions between clips, ~5 min wait per clip)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
