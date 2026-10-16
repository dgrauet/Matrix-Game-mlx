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
        """x0 = x_t - sigma_t * flow in float64, as upstream (on the MLX CPU stream: float64 is CPU-only)."""
        original_dtype = flow_pred.dtype
        with mx.stream(mx.cpu):
            flow_pred, xt, sigmas, timesteps = (a.astype(mx.float64) for a in
                                                (flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps))
            timestep_id = mx.argmin(mx.abs(timesteps[None, :] - timestep.astype(mx.float64)[:, None]), axis=1)
            sigma_t = mx.where(timestep == 0, 0.0, sigmas[timestep_id]).reshape(-1, 1, 1, 1)
            x0_pred = xt - sigma_t * flow_pred
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


class WanTextEncoder:
    """UMT5-XXL prompt encoder; rows past each prompt's token count are zeroed by T5EncoderModel."""

    def __init__(self, t5_checkpoint_path: str, tokenizer_path: str, dtype=mx.bfloat16) -> None:
        from matrix_game_mlx.inference.wan.modules.t5 import T5EncoderModel
        self.text_encoder = T5EncoderModel(text_len=512, dtype=dtype, checkpoint_path=t5_checkpoint_path,
                                           tokenizer_path=tokenizer_path)

    def __call__(self, text_prompts: List[str]) -> dict:
        """Returns {"prompt_embeds": [B, 512, 4096]}."""
        context = self.text_encoder(text_prompts)
        return {"prompt_embeds": mx.stack(context)}


WAN22_VAE_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
WAN22_VAE_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


class WanVAE22Wrapper:
    """Wan2.2 VAE (48 latent channels, 4x temporal / 16x spatial) with streaming encode."""
    z_dim, temporal_stride, spatial_stride = 48, 4, 16

    def __init__(self, vae_path: str, dtype=mx.bfloat16):
        from matrix_game_mlx.inference.wan.modules.vae2_2 import WanVAE_
        model = WanVAE_(dim=160, z_dim=48, dim_mult=[1, 2, 4, 4], temperal_downsample=[False, True, True])
        clean = {}
        for key, value in mx.load(str(vae_path)).items():
            head, dot, _rest = key.partition(".")
            if dot and head.startswith("vae"):
                key = key[len(head) + 1:]
            clean[key] = value.astype(dtype) if mx.issubdtype(value.dtype, mx.floating) else value
        model.load_weights(list(clean.items()), strict=True)
        mx.eval(model.parameters())
        self._init(model, WAN22_VAE_MEAN, WAN22_VAE_STD, dtype)

    @classmethod
    def from_model(cls, model, mean, std, dtype) -> "WanVAE22Wrapper":
        self = cls.__new__(cls)
        self._init(model, mean, std, dtype)
        return self

    def _init(self, model, mean, std, dtype) -> None:
        self.model, self.dtype = model, dtype
        self.mean = mx.array(mean, dtype=mx.float32)
        self.std = mx.array(std, dtype=mx.float32)

    def _scale(self, dtype) -> List[mx.array]:
        return [self.mean.astype(dtype), (1.0 / self.std).astype(dtype)]

    def encode_stream(self, x: mx.array, feat_cache: list) -> mx.array:
        """One streaming encoder step: x [B, T, H, W, 3] (T = 1 then 4) -> normalized mu [B, 1, 48, h, w].

        Mirrors CastleStream.push's Wan branch: encoder(patchify(x), feat_cache, feat_idx=[0]) -> conv1 -> mu.
        """
        from matrix_game_mlx.inference.wan.modules.vae2_2 import patchify
        hidden = self.model.encoder(patchify(x.astype(self.dtype), patch_size=2), feat_cache=feat_cache, feat_idx=[0])
        mu = self.model.conv1(hidden)[..., :self.z_dim]
        mean, inv_std = self._scale(self.dtype)
        return ((mu - mean) * inv_std).transpose(0, 1, 4, 2, 3)

    def encode_to_latent(self, pixel: mx.array) -> mx.array:
        """pixel [B, F, 3, H, W] in [-1, 1] -> latents [B, F', 48, h, w] (float32)."""
        scale = self._scale(self.dtype)
        out = [self.model.encode(u.transpose(0, 2, 3, 1)[None].astype(self.dtype), scale).astype(mx.float32)[0]
               for u in pixel]
        return mx.stack(out).transpose(0, 1, 4, 2, 3)
