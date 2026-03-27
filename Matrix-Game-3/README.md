<div align="center">
<h1 align="center">Matrix-Game 3.0</h1>
<h3 align="center">Real-Time and Streaming Interactive World Model with Long-Horizon Memory</h3>
</div>

<font size=7><div align='center' >  [[🤗 HuggingFace](https://huggingface.co/Skywork/Matrix-Game-3.0)] [[📖 Technical Report](assets/pdf/report.pdf)] [[🚀 Project Website](https://matrix-game-v3.github.io/)] </div></font>

https://github.com/user-attachments/assets/958cbdba-cae5-4b9c-afa2-33db3ec2757c

## 📝 Overview
**Matrix-Game-3.0** is an open-sourced, memory-augmented interactive world model designed for 720p real-time long-form video generation.
- **Upgraded Data Engine**: Combines Unreal Engine-based synthetic data, large-scale automated AAA game data, and real-world video augmentation to generate high-quality Video–Pose–Action–Prompt data. 
- **Long-horizon Memory & Consistency**: Uses prediction residuals and frame re-injection for self-correction, while camera-aware memory ensures long-term spatiotemporal consistency. 
- **Real-Time Interactivity & Open Access**: It employs a multi-segment autoregressive distillation strategy based on Distribution Matching Distillation (DMD), combined with model quantization and VAE decoder distillation to support [40fps] real-time generation at 720p resolution with a 5B model, while maintaining stable memory consistency over minute-long sequence.
- **Scale Up 28B-MoE Model**: Scaling up to a 2×14B model further improves generation quality, dynamics, and generalization. 

## 🤗 Matrix-Game-3.0 Model
we provide two pretrained model weights including base model and distilled model. Please refer to our HuggingFace page to reach these resources.

## Requirements
It supports one gpu or multi-gpu inference. We tested this repo on the following setup:
* A/H series GPUs are tested.
* Linux operating system.
* 64 GB RAM.

## ⚙️ Quick Start
### Installation
Create a conda environment and install dependencies:
```
conda create -n matrix-game-3.0 python=3.12 -y
conda activate matrix-game-3.0
# install FlashAttention
# Our project also depends on [FlashAttention](https://github.com/Dao-AILab/flash-attention)
git clone https://github.com/SkyworkAI/Matrix-Game-3.0.git
cd Matrix-Game-3.0
pip install -r requirements.txt
```

### Model Download
```
pip install "huggingface_hub[cli]"
huggingface-cli download Matrix-Game-3.0 --local-dir Matrix-Game-3.0
```
### Inference
Before running inference, you need to prepare:
- Input image
- Text prompt

After downloading pretrained models, you can use the following command to generate an interactive video with random actions:
``` sh
torchrun --nproc_per_node=$NUM_GPUS generate.py --size 704*1280 --dit_fsdp --t5_fsdp --ckpt_dir Matrix-Game-3.0 --fa_version 3 --use_int8 --num_iterations 12 --num_inference_steps 3 --image demo_images/000/image.png --prompt "a vintage gas station with a classic car parked under a canopy, set against a desert landscape." --save_name test --seed 42 --compile_vae --lightvae_pruning_rate 0.5 --vae_type mg_lightvae --output_dir ./output
# "num_iterations" refers to the number of iterations you want to generate. The total number of frames generated is given by:57 + (num_iterations - 1) * 40 
```
Tips: 
If you want to use the base model, you can use "--use_base_model --num_inference_steps 50". Otherwise if you want to generating the interactive videos with your own input actions, you can use "--interactive".
With multiple GPUs, you can pass `--use_async_vae --async_vae_warmup_iters 1` to speed up inference (see [`test.sh`](test.sh)).

## ⭐ Acknowledgements
- [Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their excellent work
- [GameFactory](https://github.com/KwaiVGI/GameFactory) for their idea of action control module
- [LightX2V](https://github.com/ModelTC/lightx2v) for their excellent quantization framework
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) for their strong base model
- [lingbot-world](https://github.com/Robbyant/lingbot-world) for their context parallel framework 
## 📜 License
This project is licensed under the Apache License, Version 2.0 — see [LICENSE.txt](LICENSE.txt).

## 📖 Citation
If you find this work useful for your research, please kindly cite our paper:

```
  @misc{2026matrix,
    title={Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory},
    author={{Skywork AI Matrix-Game Team}},
    year={2026},
    howpublished={Technical report},
    url={https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/assets/pdf/report.pdf}
  }
```
