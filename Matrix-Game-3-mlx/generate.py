"""CLI entry point for generating videos with Matrix-Game-3 (MLX)."""

import sys
import os
import argparse
import logging
import random

import mlx.core as mx
from PIL import Image

from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from pipeline.inference_pipeline import MatrixGame3Pipeline
from utils.misc import set_seed


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from an image, text prompt and actions using Matrix-Game-3 (MLX)."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to mlx-forge converted model directory or HuggingFace repo ID.")
    parser.add_argument(
        "--size", type=str, default="1280*704",
        help="Output video size (e.g. '1280*704').")
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt describing the scene.")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.")
    parser.add_argument(
        "--num_iterations", type=int, default=12,
        help="Number of clip iterations.")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="Number of denoising steps per clip.")
    parser.add_argument(
        "--sample_shift", type=float, default=None,
        help="Sampling shift for flow matching.")
    parser.add_argument(
        "--sample_guide_scale", type=float, default=5.0,
        help="Classifier-free guidance scale.")
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory.")
    parser.add_argument(
        "--save_name", type=str, default="generated_video",
        help="Output filename (without extension).")
    parser.add_argument(
        "--use_base_model", action="store_true",
        help="Use base model (50 steps with CFG) instead of distilled.")

    args = parser.parse_args()

    cfg = WAN_CONFIGS["matrix_game3"]
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.seed < 0:
        args.seed = random.randint(0, sys.maxsize)

    return args


def generate(args):
    """Load the pipeline and generate a video."""
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    set_seed(args.seed)

    cfg = WAN_CONFIGS["matrix_game3"]
    logging.info(f"Generation config: {cfg}")
    logging.info(f"Input image: {args.image}")
    logging.info(f"Input prompt: {args.prompt}")

    pil_image = Image.open(args.image).convert("RGB")

    logging.info("Creating Matrix-Game-3 pipeline (MLX).")
    pipeline = MatrixGame3Pipeline(
        config=cfg,
        model_path=args.model_path,
    )

    logging.info("Generating video...")
    pipeline.generate(
        text=args.prompt,
        pil_image=pil_image,
        max_area=MAX_AREA_CONFIGS[args.size],
        shift=args.sample_shift,
        num_inference_steps=args.num_inference_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.seed,
        num_iterations=args.num_iterations,
        output_dir=args.output_dir,
        save_name=args.save_name,
        use_cfg=args.use_base_model,
    )

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
