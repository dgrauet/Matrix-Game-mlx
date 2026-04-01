"""Inference pipeline for Matrix-Game-3 on MLX.

Ports MatrixGame3Pipeline from the PyTorch reference, removing all distributed,
async VAE, Int8 quantization, and torch.compile logic. Single-device inference
on Apple Silicon via MLX unified memory.
"""

import logging
import math
import os
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.cam_utils import (
    compute_relative_poses,
    get_intrinsics,
    select_memory_idx_fov,
    _interpolate_camera_poses_handedness,
)
from utils.utils import get_data, build_plucker_from_c2ws, build_plucker_from_pose
from utils.visualize import process_video
from pipeline.vae_config import load_vae

logger = logging.getLogger(__name__)

__all__ = ["MatrixGame3Pipeline"]


class MatrixGame3Pipeline:
    """End-to-end inference pipeline for Matrix-Game-3 on MLX.

    Loads T5 text encoder, DiT backbone, and VAE decoder, then runs the
    autoregressive multi-clip diffusion loop with memory-based conditioning.

    Args:
        config: EasyDict with model hyperparameters (from ``wan.configs``).
        model_path: Path to directory containing mlx-forge converted weights
            (``dit.safetensors``, ``t5_encoder.safetensors``, ``vae.safetensors``).
        dtype: Model precision (default ``mx.bfloat16``).
    """

    def __init__(
        self,
        config,
        model_path: str,
        dtype: mx.Dtype = mx.bfloat16,
        use_distilled: bool = True,
    ):
        self.config = config
        self.dtype = dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        # Resolve model_path: if it's a HuggingFace repo ID, download to cache
        if not os.path.isdir(model_path) and "/" in model_path:
            from huggingface_hub import snapshot_download
            logger.info("Downloading model from HuggingFace: %s", model_path)
            model_path = snapshot_download(model_path)
            logger.info("Model cached at: %s", model_path)

        # --- T5 text encoder ---
        t5_ckpt = os.path.join(model_path, "t5_encoder.safetensors")
        t5_tokenizer = getattr(config, "t5_tokenizer", "google/umt5-xxl")
        logger.info("Loading T5 text encoder from %s", t5_ckpt)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            checkpoint_path=t5_ckpt,
            tokenizer_path=t5_tokenizer,
        )

        # --- DiT backbone ---
        dit_filename = "dit_distilled.safetensors" if use_distilled else "dit.safetensors"
        dit_path = os.path.join(model_path, dit_filename)
        logger.info("Loading DiT model from %s", dit_path)
        self.model = WanModel(
            model_type=getattr(config, "model_type", "ti2v"),
            patch_size=config.patch_size,
            text_len=config.text_len,
            in_dim=config.in_dim,
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=getattr(config, "text_dim", 4096),
            out_dim=config.out_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=config.window_size,
            qk_norm=config.qk_norm,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            action_config=getattr(config, "action_config", {}),
            use_memory=getattr(config, "use_memory", True),
            sigma_theta=getattr(config, "sigma_theta", 0.0),
        )
        weights = mx.load(dit_path)
        # Strip component prefix if present (dit. or dit_distilled.)
        prefix = "dit_distilled." if use_distilled else "dit."
        clean_weights = {k.replace(prefix, "", 1): v for k, v in weights.items()}

        # Detect quantized weights and convert Linear -> QuantizedLinear
        has_scales = any(k.endswith(".scales") for k in clean_weights)
        if has_scales:
            # Infer quantization config from weight shapes
            for k, v in clean_weights.items():
                if k.endswith(".scales"):
                    bits = 8 if clean_weights[k.replace(".scales", ".weight")].dtype == mx.uint8 else 4
                    group_size = v.shape[-1]
                    break
            logger.info("Detected quantized weights (bits=%d, group_size=%d)", bits, group_size)
            nn.quantize(self.model, bits=bits, group_size=group_size)

        self.model.load_weights(list(clean_weights.items()))
        mx.eval(self.model.parameters())  # materialize weights
        logger.info("DiT model loaded (%d layers).", config.num_layers)

        # --- VAE ---
        vae_path = os.path.join(model_path, "vae.safetensors")
        logger.info("Loading VAE from %s", vae_path)
        self.vae = load_vae(
            model_path=vae_path,
            dtype=mx.float32,
        )

    def generate(
        self,
        text: str,
        pil_image: Image.Image,
        output_dir: str = "outputs",
        save_name: str = "output",
        height: int = 704,
        width: int = 1280,
        max_area: int = 704 * 1280,
        shift: float = 5.0,
        num_inference_steps: int = 40,
        guide_scale: float = 5.0,
        num_iterations: int = 12,
        seed: int = -1,
        use_cfg: bool = True,
    ) -> Optional[np.ndarray]:
        """Generate video frames from an input image and text prompt.

        Runs the multi-clip autoregressive diffusion loop with camera control.

        Args:
            text: Text prompt for content generation.
            pil_image: Input PIL image.
            output_dir: Directory to save the output video.
            save_name: Base name for the output file (without extension).
            height: Target video height.
            width: Target video width.
            max_area: Maximum pixel area for latent space calculation.
            shift: Noise schedule shift parameter.
            num_inference_steps: Number of diffusion sampling steps.
            guide_scale: Classifier-free guidance scale.
            num_iterations: Number of autoregressive clip iterations.
            seed: Random seed (-1 for random).
            use_cfg: Whether to use classifier-free guidance (True for base
                model, False for distilled model).

        Returns:
            Denormalized video as uint8 numpy array of shape (T, H, W, 3),
            or None if no frames were generated.
        """
        mouse_icon = "assets/images/mouse.png"

        clip_frame = 56
        first_clip_frame = clip_frame + 1  # 57
        past_frame = 16

        if seed == -1:
            seed = int(np.random.randint(0, 2**31))
        mx.random.seed(seed)

        num_frames = first_clip_frame + (num_iterations - 1) * 40

        # --- Prepare input data ---
        current_image, extrinsics_all, keyboard_condition_all, mouse_condition_all = (
            get_data(num_frames, height, width, pil_image)
        )

        # --- Encode text ---
        logger.info("Encoding text prompt...")
        cond = self.text_encoder([text])
        neg_cond = self.text_encoder([self.config.sample_neg_prompt])

        # --- Compute latent dimensions ---
        # current_image shape: (1, 1, H, W, C) channels-last
        h_orig = current_image.shape[2]
        w_orig = current_image.shape[3]
        aspect_ratio = h_orig / w_orig
        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.vae_stride[1]
            // self.patch_size[1]
            * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.vae_stride[2]
            // self.patch_size[2]
            * self.patch_size[2]
        )
        target_h = lat_h * self.vae_stride[1]
        target_w = lat_w * self.vae_stride[2]

        base_K = get_intrinsics(target_h, target_w)

        # --- Encode input image with VAE ---
        logger.info("Encoding input image with VAE...")
        # current_image[0] is (1, H, W, C) channels-last — VAE encode expects (T, H, W, C)
        img_cond = self.vae.encode([current_image[0]])[0]  # (T, lat_h, lat_w, 48)
        img_cond = mx.expand_dims(img_cond, axis=0)  # (1, T, lat_h, lat_w, 48)
        img_cond = img_cond.astype(self.dtype)
        mx.eval(img_cond)

        # --- Compute max sequence length ---
        max_lat_f = (first_clip_frame - 1) // self.vae_stride[0] + 1
        max_mem_f = 5
        max_total_f = max_lat_f + max_mem_f
        max_seq_len = (
            max_total_f
            * lat_h
            * lat_w
            // (self.patch_size[1] * self.patch_size[2])
        )

        # --- Main generation loop ---
        all_latents_list: List[mx.array] = []
        all_videos_list: List[mx.array] = []

        current_end_frame_idx = 0  # will be updated each iteration

        for clip_idx in range(num_iterations):
            first_clip = clip_idx == 0
            logger.info("Iteration %d/%d", clip_idx + 1, num_iterations)

            def align_frame_to_block(frame_idx: int) -> int:
                return (frame_idx - 1) // 4 * 4 + 1 if frame_idx > 0 else 1

            def get_latent_idx(frame_idx: int) -> int:
                return (frame_idx - 1) // 4 + 1

            current_end_frame_idx = (
                first_clip_frame
                if first_clip
                else first_clip_frame + clip_idx * (clip_frame - past_frame)
            )
            current_start_frame_idx = (
                0 if first_clip else current_end_frame_idx - clip_frame
            )

            # --- Camera pose interpolation and Plucker rays ---
            c2ws_chunk = extrinsics_all[current_start_frame_idx:current_end_frame_idx]
            src_indices = np.linspace(
                current_start_frame_idx,
                current_end_frame_idx - 1,
                first_clip_frame if first_clip else clip_frame,
            )
            tgt_len = (
                (first_clip_frame - 1) // 4 + 1
                if first_clip
                else (clip_frame // 4)
            )
            tgt_indices = np.linspace(
                0 if first_clip else current_start_frame_idx + 3,
                current_end_frame_idx - 1,
                tgt_len,
            )

            c2ws_chunk_mx = mx.array(c2ws_chunk.astype(np.float32))
            plucker = build_plucker_from_c2ws(
                c2ws_chunk_mx,
                src_indices,
                tgt_indices,
                framewise=True,
                base_K=base_K,
                target_h=target_h,
                target_w=target_w,
                lat_h=lat_h,
                lat_w=lat_w,
            )
            plucker_no_mem = plucker

            # --- Memory frame selection (non-first clips) ---
            if first_clip:
                x_memory = None
                memory_mouse_condition = None
                memory_keyboard_condition = None
                latent_idx = None
                timestep_memory = None
            else:
                selected_index_base = [
                    current_end_frame_idx - o for o in range(1, 34, 8)
                ]
                selected_index = select_memory_idx_fov(
                    extrinsics_all,
                    current_start_frame_idx,
                    selected_index_base,
                )
                selected_index[-1] = 4

                memory_pluckers: List[mx.array] = []
                latent_idx: List[int] = []
                for mem_idx, reference_idx in zip(
                    selected_index, selected_index_base
                ):
                    l_idx = get_latent_idx(mem_idx)
                    latent_idx.append(l_idx)

                    mem_idx_aligned = align_frame_to_block(mem_idx)
                    mem_block = extrinsics_all[
                        mem_idx_aligned : mem_idx_aligned + 4
                    ]
                    mem_src = np.linspace(
                        mem_idx_aligned,
                        mem_idx_aligned + 3,
                        mem_block.shape[0],
                    )
                    mem_tgt = np.array(
                        [mem_idx_aligned + 3], dtype=np.float32
                    )
                    mem_pose = _interpolate_camera_poses_handedness(
                        src_indices=mem_src,
                        src_rot_mat=mem_block[:, :3, :3],
                        src_trans_vec=mem_block[:, :3, 3],
                        tgt_indices=mem_tgt,
                    )
                    reference_pose = extrinsics_all[
                        reference_idx : reference_idx + 1
                    ]
                    rel_pair = np.concatenate(
                        [reference_pose, mem_pose], axis=0
                    )
                    rel_pair_mx = mx.array(rel_pair.astype(np.float32))
                    rel_pose = compute_relative_poses(
                        rel_pair_mx, framewise=False
                    )[1:2]

                    memory_pluckers.append(
                        build_plucker_from_pose(
                            rel_pose,
                            base_K=base_K,
                            target_h=target_h,
                            target_w=target_w,
                            lat_h=lat_h,
                            lat_w=lat_w,
                        )
                    )

                # Concatenate memory pluckers with current clip plucker
                # All have shape (1, F_i, lat_h, lat_w, C), concatenate on axis=1
                plucker = mx.concatenate(
                    memory_pluckers + [plucker], axis=1
                )

                # Gather memory latents from all previous latents
                src = mx.concatenate(all_latents_list, axis=1)  # (1, total_T, lat_h, lat_w, C)
                x_memory = src[:, latent_idx]  # (1, 5, lat_h, lat_w, C)

                num_mem = len(selected_index)
                memory_mouse_condition = mx.ones(
                    (1, num_mem, 2), dtype=self.dtype
                )
                memory_keyboard_condition = -mx.ones(
                    (1, num_mem, 6), dtype=self.dtype
                )
                # timestep_memory: all zeros for memory frames
                mem_seq_len = (
                    x_memory.shape[1]
                    * x_memory.shape[2]
                    * x_memory.shape[3]
                    // 4
                )
                timestep_memory = mx.zeros(
                    (1, mem_seq_len), dtype=self.dtype
                )

            # --- Conditions for current clip ---
            keyboard_condition = keyboard_condition_all[
                :, current_start_frame_idx:current_end_frame_idx
            ]
            mouse_condition = mouse_condition_all[
                :, current_start_frame_idx:current_end_frame_idx
            ]
            plucker = plucker.astype(self.dtype)
            plucker_no_mem = plucker_no_mem.astype(self.dtype)

            # --- Noise schedule ---
            scheduler = FlowUniPCMultistepScheduler()
            scheduler.set_timesteps(num_inference_steps, shift=shift)
            timesteps = scheduler.timesteps

            latent_start_idx = get_latent_idx(current_start_frame_idx)
            latent_end_idx = get_latent_idx(current_end_frame_idx)
            num_lat_frames = latent_end_idx - latent_start_idx

            # --- Initial noise ---
            latents = mx.random.normal(
                shape=(
                    1,
                    num_lat_frames,
                    img_cond.shape[2],
                    img_cond.shape[3],
                    48,
                )
            ).astype(self.dtype)
            # Replace first frame(s) with image condition
            img_t = img_cond.shape[1]
            latents = mx.concatenate(
                [img_cond, latents[:, img_t:]], axis=1
            )

            # --- Build condition dicts ---
            conditions_full = {
                "mouse_cond": mouse_condition,
                "keyboard_cond": keyboard_condition,
                "context": cond,
                "plucker_emb": plucker,
                "x_memory": x_memory,
                "timestep_memory": timestep_memory,
                "keyboard_cond_memory": memory_keyboard_condition,
                "mouse_cond_memory": memory_mouse_condition,
                "memory_latent_idx": latent_idx,
                "predict_latent_idx": (latent_start_idx, latent_end_idx),
            }

            conditions_null = {
                "mouse_cond": mx.ones_like(mouse_condition).astype(self.dtype),
                "keyboard_cond": -mx.ones_like(keyboard_condition).astype(
                    self.dtype
                ),
                "context": neg_cond,
                "plucker_emb": plucker_no_mem,
                "x_memory": None,
                "timestep_memory": None,
                "keyboard_cond_memory": None,
                "mouse_cond_memory": None,
                "memory_latent_idx": None,
                "predict_latent_idx": (latent_start_idx, latent_end_idx),
            }

            # --- Denoising loop ---
            for t in tqdm(
                timesteps.tolist(),
                desc=f"Clip {clip_idx + 1}/{num_iterations}",
            ):
                latent_model_input = latents

                # Build per-patch timestep: (num_lat_frames, patches_per_frame)
                patches_per_frame = (
                    latents.shape[2] * latents.shape[3] // 4
                )
                timestep = mx.full(
                    (latents.shape[1], patches_per_frame),
                    t,
                    dtype=self.dtype,
                )
                # Zero out timesteps for conditioning frames
                if img_t > 0:
                    zeros = mx.zeros(
                        (img_t, patches_per_frame), dtype=self.dtype
                    )
                    timestep = mx.concatenate(
                        [zeros, timestep[img_t:]], axis=0
                    )
                timestep = timestep.reshape(1, -1)  # (1, total_patches)

                model_kwargs = {
                    "x": [latent_model_input[0]],
                    "t": timestep,
                    "seq_len": max_seq_len,
                    **conditions_full,
                }

                if use_cfg:
                    model_kwargs_null = {
                        "x": [latent_model_input[0]],
                        "t": timestep,
                        "seq_len": max_seq_len,
                        **conditions_null,
                    }
                    noise_pred_full = self.model(**model_kwargs)
                    noise_pred_null = self.model(**model_kwargs_null)
                    # CFG: null + scale * (full - null)
                    noise_pred_combined = [
                        n + guide_scale * (f - n)
                        for f, n in zip(noise_pred_full, noise_pred_null)
                    ]
                    noise_pred = noise_pred_combined[0]
                else:
                    noise_pred_list = self.model(**model_kwargs)
                    noise_pred = noise_pred_list[0]

                # noise_pred is channels-last (F, H, W, C) per the model output
                # Scheduler expects (B, T, H, W, C) for channels-last
                noise_pred_5d = mx.expand_dims(noise_pred, axis=0)
                result = scheduler.step(
                    noise_pred_5d, t, latents, return_dict=False
                )
                latents = result[0]
                latents = mx.concatenate(
                    [img_cond, latents[:, img_t:]], axis=1
                )
                mx.eval(latents)

            # --- Update image condition for next clip ---
            img_cond = latents[:, -4:]

            # --- Select frames to decode ---
            denoised_pred = (
                latents if first_clip else latents[:, -10:]
            )

            # --- VAE decode ---
            # denoised_pred: (1, T, lat_h, lat_w, 48) channels-last
            video = self.vae.decode([denoised_pred[0]])[0]  # (T_out, H, W, C)
            all_videos_list.append(video)
            mx.eval(video)

            all_latents_list.append(denoised_pred)

        # --- Assemble final video ---
        if len(all_videos_list) == 0:
            logger.warning("No video segments generated.")
            return None

        concatenated_video_mx = mx.concatenate(all_videos_list, axis=0)  # (T, H, W, C)

        # Denormalize: [-1, 1] -> [0, 255]
        video_np = np.array(concatenated_video_mx.astype(mx.float32))
        video_np = ((video_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        # Truncate to actual frame count
        video_np = video_np[:current_end_frame_idx]

        # Save video with overlays
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{save_name}.mp4")

        keyboard_np = np.array(
            keyboard_condition_all[0, :current_end_frame_idx].astype(mx.float32)
        )
        mouse_np = np.array(
            mouse_condition_all[0, :current_end_frame_idx].astype(mx.float32)
        )

        process_video(
            video_np,
            output_path,
            (keyboard_np, mouse_np),
            mouse_icon,
            mouse_scale=0.2,
            default_frame_res=(height, width),
        )
        logger.info(
            "Saved video with %d frames to %s", len(video_np), output_path
        )

        return video_np
