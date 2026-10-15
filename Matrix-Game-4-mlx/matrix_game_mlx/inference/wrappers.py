"""Model wrappers (MLX port of matrix_game/inference/wrappers.py, inference subset)."""
from typing import List, Optional

import mlx.core as mx

from matrix_game_mlx.inference.scheduler import FlowMatchScheduler

CONTROL_CHANNELS = 48


def init_kv_cache(generator, batch_size: int, kv_cache_size: int, dtype) -> List[dict]:
    """Per-layer self-attention KV cache sized from the model's layer, head and head-dim counts."""
    m = getattr(generator, "model", generator)
    n_head, head_dim = m.num_heads, m.dim // m.num_heads
    return [{"k": mx.zeros((batch_size, kv_cache_size, n_head, head_dim), dtype=dtype),
             "v": mx.zeros((batch_size, kv_cache_size, n_head, head_dim), dtype=dtype),
             "global_end_index": 0, "local_end_index": 0} for _ in range(m.num_layers)]


def init_crossattn_cache(generator, batch_size: int, dtype) -> List[dict]:
    """Per-layer cross-attention cache; k/v are filled by the first forward call."""
    m = getattr(generator, "model", generator)
    return [{"is_init": False} for _ in range(m.num_layers)]


def frame_seq_length_of(generator, latent_h: int, latent_w: int) -> int:
    """Tokens per latent frame = (h / patch_h) * (w / patch_w); 30x54 latents -> 405."""
    m = getattr(generator, "model", generator)
    return (latent_h // m.patch_size[1]) * (latent_w // m.patch_size[2])


class WanDiffusionWrapper:
    def __init__(self, model, timestep_shift: float = 5.0):
        self.model = model
        self.uniform_timestep = False
        self.is_causal = True
        self.extra_in_channels = model.in_dim - model.out_dim
        self.control_channels = min(CONTROL_CHANNELS, self.extra_in_channels)
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

    def get_scheduler(self):
        return self.scheduler

    def frame_seq_length(self, latent_h: int, latent_w: int) -> int:
        return frame_seq_length_of(self, latent_h, latent_w)

    def _seq_len(self, x_bfchw: mx.array) -> int:
        return x_bfchw.shape[1] * self.frame_seq_length(x_bfchw.shape[-2], x_bfchw.shape[-1])

    def _with_condition(self, x_bfchw: mx.array, control: Optional[mx.array]) -> List[mx.array]:
        """[B, F, 48, h, w] noisy + control -> per-sample [F, h, w, in_dim] (noisy 48 | control 48 | zeros 52)."""
        x = x_bfchw.transpose(0, 1, 3, 4, 2)
        b, f, h, w, _ = x.shape
        parts, n_zero = [x], self.extra_in_channels
        if control is not None:
            assert control.shape == (b, f, self.control_channels, h, w), (control.shape, x_bfchw.shape)
            parts.append(control.transpose(0, 1, 3, 4, 2).astype(x.dtype))
            n_zero -= self.control_channels
        parts.append(mx.zeros((b, f, h, w, n_zero), dtype=x.dtype))
        return list(mx.concatenate(parts, axis=-1))

    def _convert_flow_pred_to_x0(self, flow_pred: mx.array, xt: mx.array, timestep: mx.array) -> mx.array:
        """x0 = x_t - sigma_t * flow (upstream float64, here float32: documented deviation)."""
        original_dtype = flow_pred.dtype
        sigmas, timesteps = self.scheduler.sigmas, self.scheduler.timesteps
        timestep_id = mx.argmin(mx.abs(timesteps[None, :] - timestep[:, None]), axis=1)
        sigma_t = mx.where(timestep == 0, 0.0, sigmas[timestep_id]).reshape(-1, 1, 1, 1)
        x0_pred = xt.astype(mx.float32) - sigma_t * flow_pred.astype(mx.float32)
        return x0_pred.astype(original_dtype)

    def __call__(self, noisy_image_or_video: mx.array, conditional_dict: dict, timestep: mx.array,
                 kv_cache: Optional[List[dict]] = None, crossattn_cache: Optional[List[dict]] = None,
                 current_start: Optional[int] = None, cache_start: Optional[int] = None,
                 drop_control: bool = False, sink_rope_mode: Optional[str] = None):
        """KV-cache inference forward: returns (flow_pred, pred_x0), both [B, F, 48, h, w]."""
        prompt_embeds = conditional_dict["prompt_embeds"]
        control = None if drop_control else conditional_dict.get("control_latent", None)
        if control is not None and control.shape[1] != noisy_image_or_video.shape[1]:
            assert kv_cache is not None and current_start is not None
            fsl = self.frame_seq_length(noisy_image_or_video.shape[-2], noisy_image_or_video.shape[-1])
            start_frame = current_start // fsl
            control = control[:, start_frame:start_frame + noisy_image_or_video.shape[1]]
        model_in = self._with_condition(noisy_image_or_video, control)
        out = self.model(model_in, t=timestep, context=prompt_embeds, seq_len=self._seq_len(noisy_image_or_video),
                         kv_cache=kv_cache, crossattn_cache=crossattn_cache, current_start=current_start,
                         cache_start=cache_start or 0,
                         sink_rope_mode=None if sink_rope_mode == "absolute" else sink_rope_mode)
        flow_pred = out.transpose(0, 1, 4, 2, 3)            # [B, F, H, W, C] -> [B, F, C, H, W]
        b, f = flow_pred.shape[:2]
        pred_x0 = self._convert_flow_pred_to_x0(flow_pred.reshape(b * f, *flow_pred.shape[2:]),
                                                noisy_image_or_video.reshape(b * f, *noisy_image_or_video.shape[2:]),
                                                timestep.reshape(-1)).reshape(flow_pred.shape)
        return flow_pred, pred_x0
