"""Scene-configured chunk1 sessions with Wan or TAEHV streaming codecs (MLX port of matrix_game/inference/stream.py)."""
from dataclasses import dataclass, field
import time
from typing import Optional

import mlx.core as mx
import numpy as np

from matrix_game_mlx.inference.scopes import scope

LATENT_SHAPE = (1, 1, 48, 30, 54)
FRAME_SIZE = (480, 864)


def encode_sink_image(vae, path, vae_backend="wan", *, first_frame=False):
    """Encode an independent image, without seeding either streaming VAE cache."""
    from PIL import Image, ImageOps
    from matrix_game_mlx.inference.preprocessing import crop_and_resize, frames_to_tensor
    if vae_backend not in ("wan", "taehv"):
        raise ValueError("Expected wan or taehv VAE backend")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = (ImageOps.fit(image, (FRAME_SIZE[1], FRAME_SIZE[0]), method=Image.Resampling.LANCZOS) if first_frame
                 else crop_and_resize(image, *FRAME_SIZE))
        pixels = frames_to_tensor([image])
    pixels = pixels[None].astype(vae.dtype)
    if vae_backend == "wan":
        latent = vae.encode_to_latent(pixels)
    else:
        vae.clear_cache()
        try:
            latent = vae.encode(pixels.transpose(0, 2, 1, 3, 4)).transpose(0, 2, 1, 3, 4)
        finally:
            vae.clear_cache()
    if tuple(latent.shape) != LATENT_SHAPE or not bool(mx.all(mx.isfinite(latent))):
        raise ValueError("Sink image must encode to one finite 48-channel latent")
    return latent


@dataclass
class Chunk:
    index: int
    frame_start: int
    frame_end: int
    pixels: Optional[mx.array]
    latent: mx.array
    control: mx.array
    denoising_steps: int
    timings_ms: dict = field(default_factory=dict)
    rgb: Optional[np.ndarray] = None            # uint8 [T, H, W, 3]; set (and pixels None) only with on_rgb


class CastleStream:
    """Accept [-1,1] RGB [T,3,480,864]; retain only bounded inference caches."""

    def __init__(self, generator, vae, prompt_embeds, *, seed=0, max_latents=8192, vae_backend="wan",
                 steps=(1000, 500), first_steps=(1000, 750, 500, 250), sink_latent=None, sink_mode="kv_only",
                 sink_rope_mode="fixed", profile=False):
        from matrix_game_mlx.inference.rollout import set_attention_window, warp_steps
        from matrix_game_mlx.inference.wrappers import frame_seq_length_of, init_crossattn_cache, init_kv_cache

        self.gen, self.vae, self.prompt = generator, vae, prompt_embeds
        if sink_mode not in ("kv_only", "first_frame") or sink_rope_mode not in ("absolute", "fixed"):
            raise ValueError("Invalid sink policy")
        self.sink_mode, self.sink_rope_mode = sink_mode, sink_rope_mode
        self.profile = profile
        self.first_frame_latent = sink_latent if sink_mode == "first_frame" else None
        if vae_backend not in ("wan", "taehv"):
            raise ValueError("Expected wan or taehv VAE backend")
        for values in (steps, first_steps):
            if not values or values[0] != 1000 or any(not 0 < v <= 1000 for v in values) or any(
                    a <= b for a, b in zip(values, values[1:])):
                raise ValueError("Denoising timesteps must decrease strictly from 1000")
        self.vae_backend = vae_backend
        self.dtype = generator.model.patch_embedding.weight.dtype
        self.shape = LATENT_SHAPE
        if sink_latent is not None and (tuple(sink_latent.shape) != self.shape
                                       or not mx.issubdtype(sink_latent.dtype, mx.floating)
                                       or not bool(mx.all(mx.isfinite(sink_latent)))):
            raise ValueError("Sink must be one finite floating-point latent frame")
        self.sink_offset = int(sink_latent is not None and sink_mode == "kv_only")
        self.index = self.frame = 0
        self.failed = self.closed = False
        if max_latents < 1:
            raise ValueError("max_latents must be positive")
        self.max_latents = max_latents
        # Three generator roles as upstream (initial, racing, renoise); MLX RNG is not torch-compatible.
        self.initial_key = mx.random.key(seed + 1)
        self.racing_key = mx.random.key(seed)
        self.renoise_key = mx.random.key(seed)
        self.scheduler = generator.get_scheduler()
        self.steps = warp_steps(self.scheduler, list(steps))
        self.steps_first = warp_steps(self.scheduler, list(first_steps))
        set_attention_window(generator, 21, 1)
        generator.model.set_rope_max_len(max(8192, max_latents + self.sink_offset))
        self.fsl = frame_seq_length_of(generator, self.shape[3], self.shape[4])
        self.kv = init_kv_cache(generator, 1, 21 * self.fsl, self.dtype)
        self.xc = init_crossattn_cache(generator, 1, self.dtype)
        for cache in self.kv:
            cache["k_sink_raw"] = mx.zeros_like(cache["k"][:, :self.fsl])
        self.encoder_features, self.decoder = [], None
        if vae_backend == "wan":
            from matrix_game_mlx.inference.decoder import IncrementalDecoder
            from matrix_game_mlx.inference.wan.modules.vae2_2 import count_conv3d
            self.encoder_features = [None] * count_conv3d(vae.model.encoder)
            self.decoder = IncrementalDecoder(vae)
        else:
            vae.clear_cache()
        runtime = getattr(getattr(generator, "model", None), "castle_runtime", None)
        if runtime is not None:
            runtime.prime(generator, self.prompt, self.shape, self.kv, self.xc, self.fsl, sink_rope_mode)
        if self.sink_offset:
            self.gen(noisy_image_or_video=sink_latent.astype(self.dtype), conditional_dict={"prompt_embeds": self.prompt},
                     timestep=mx.zeros((1, 1)), drop_control=True, kv_cache=self.kv, crossattn_cache=self.xc,
                     current_start=0, sink_rope_mode=self.sink_rope_mode)
            self._eval_caches()

    def _eval_caches(self, *extra):
        mx.eval(*extra, *[c[k] for c in self.kv for k in ("k", "v", "k_sink_raw")],
                *[c[k] for c in self.xc if c.get("is_init") for k in ("k", "v")])

    def _next_noise(self, attr):
        key, sub = mx.random.split(getattr(self, attr))
        setattr(self, attr, key)
        return mx.random.normal(self.shape, key=sub)

    def _validate(self, pixels, *, frame_start, initial_noise=None):
        if self.closed or self.failed:
            raise RuntimeError("Session is closed or failed; create a fresh session")
        count = 1 if self.index == 0 else 4
        if frame_start != self.frame or tuple(pixels.shape) != (count, 3, *FRAME_SIZE):
            raise ValueError("Expected consecutive RGB frames: 1 initially, then 4 per chunk")
        if self.index >= self.max_latents:
            raise ValueError("Session reached its configured RoPE/frame budget")
        if not mx.issubdtype(pixels.dtype, mx.floating):
            raise ValueError("RGB input must be finite floating point in [-1,1]")
        if initial_noise is not None and tuple(initial_noise.shape) != self.shape:
            raise ValueError("Initial noise must be finite and match one latent frame")
        # Every value check in one host sync (pixels finite and in range, noise finite).
        reads = [mx.all(mx.isfinite(pixels)), pixels.min(), pixels.max()]
        if initial_noise is not None:
            reads.append(mx.all(mx.isfinite(initial_noise)))
        mx.eval(reads)
        if not bool(reads[0]) or float(reads[1]) < -1 or float(reads[2]) > 1:
            raise ValueError("RGB input must be finite floating point in [-1,1]")
        if initial_noise is not None and not bool(reads[3]):
            raise ValueError("Initial noise must be finite and match one latent frame")
        return count

    def set_prompt(self, embeddings):
        if self.closed or self.failed or embeddings.shape != self.prompt.shape:
            raise ValueError("Prompt update requires an active session and unchanged token shape")
        self.prompt = embeddings
        for cross in self.xc:
            cross["is_init"] = False

    def push(self, pixels, *, frame_start, initial_noise=None, renoise=None, on_rgb=None) -> Chunk:
        """With on_rgb, publish owned uint8 THWC frames after decode and before the clean-KV pass.

        The call still returns only after clean KV, so the next chunk is unchanged by early publication.
        """
        count = self._validate(pixels, frame_start=frame_start, initial_noise=initial_noise)
        try:
            marks = [time.perf_counter()]
            with scope("encode"):
                if self.vae_backend == "wan":
                    x = pixels.transpose(0, 2, 3, 1)[None]                               # [1, T, H, W, 3]
                    control = self.vae.encode_stream(x, self.encoder_features)           # [1, 1, 48, h, w]
                else:
                    # TAEHV already uses the normalized Wan2.2 latent space.
                    x = pixels.transpose(1, 0, 2, 3)[None].astype(self.vae.dtype)        # [1, 3, T, H, W]
                    control = self.vae.encode_stream(x).transpose(0, 2, 1, 3, 4)
                mx.eval(control)
            marks.append(time.perf_counter())
            if tuple(control.shape) != self.shape:
                raise RuntimeError("VAE encoder output cadence changed")
            position = self.index + self.sink_offset
            if self.sink_mode == "first_frame":
                steps = self.steps_first if self.index == 0 else self.steps
            else:
                steps = self.steps_first if position < 2 else self.steps
            reference_first = self.index == 0 and self.first_frame_latent is not None
            if reference_first:
                steps = ()
                pred = self.first_frame_latent.astype(self.dtype)
            else:
                if initial_noise is None:
                    attr = "racing_key" if self.sink_mode == "first_frame" and self.index > 0 else "initial_key"
                    initial_noise = self._next_noise(attr)
                x = initial_noise.astype(self.dtype)
            cond = {"prompt_embeds": self.prompt, "control_latent": control}
            common = dict(kv_cache=self.kv, crossattn_cache=self.xc, current_start=position * self.fsl,
                          sink_rope_mode=self.sink_rope_mode)
            with scope("denoise"):
                finite = None
                for k, step in enumerate(steps):
                    t = mx.full((1, 1), float(step))
                    _, pred = self.gen(noisy_image_or_video=x, conditional_dict=cond, timestep=t, **common)
                    finite = mx.all(mx.isfinite(pred))             # read with the step's own sync
                    self._eval_caches(pred, finite)
                    if k + 1 < len(steps):
                        flat = pred.reshape(-1, *pred.shape[2:]).astype(mx.float32)
                        noise = renoise[k].reshape(flat.shape) if renoise is not None else \
                            self._next_noise("renoise_key").reshape(flat.shape)
                        x = self.scheduler.add_noise(flat, noise.astype(mx.float32), mx.full((1,), float(steps[k + 1])))
                        x = x.reshape(1, 1, *x.shape[1:]).astype(self.dtype)
                if not bool(mx.all(mx.isfinite(pred)) if finite is None else finite):
                    raise RuntimeError("Non-finite generated latent")
            marks.append(time.perf_counter())

            def clean():
                with scope("clean_kv"):
                    self.gen(noisy_image_or_video=pred, conditional_dict=cond, timestep=mx.zeros((1, 1)),
                             drop_control=True, **common)
                    self._eval_caches()

            if on_rgb is None:
                clean()
                marks.append(time.perf_counter())
            with scope("decode"):
                if self.vae_backend == "wan":
                    decoded = self.decoder.push(pred)
                else:
                    decoded = self.vae.decode_stream(pred.transpose(0, 2, 1, 3, 4))[0].transpose(1, 0, 2, 3)
                mx.eval(decoded)
            if tuple(decoded.shape) != (count, 3, *FRAME_SIZE):
                raise RuntimeError("VAE decoder output cadence changed")
            marks.append(time.perf_counter())
            rgb = None
            if on_rgb is not None:
                from matrix_game_mlx.inference.preprocessing import tensor_to_uint8
                rgb = tensor_to_uint8(decoded)                  # fresh owned NumPy array
                marks.append(time.perf_counter())
                try:
                    on_rgb(rgb)
                finally:
                    clean()
                    marks.append(time.perf_counter())
                decoded = None
            names = ("encode", "denoise", "clean_kv", "decode") if on_rgb is None else (
                "encode", "denoise", "decode", "pack_copy", "clean_kv")
            timings = {name: (b - a) * 1000 for name, a, b in zip(names, marks, marks[1:])} if self.profile else {}
            result = Chunk(self.index, self.frame, self.frame + count - 1, decoded, pred, control, len(steps),
                           timings, rgb)
            self.index += 1
            self.frame += count
            return result
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed:
            return
        runtime = getattr(getattr(self.gen, "model", None), "castle_runtime", None)
        if runtime is not None:
            runtime.clear()
        self.kv.clear()
        self.xc.clear()
        self.encoder_features.clear()
        if self.decoder is not None:
            self.decoder.feat.clear()
        if self.vae_backend == "taehv":
            self.vae.clear_cache()
        self.closed = True
