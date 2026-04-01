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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
