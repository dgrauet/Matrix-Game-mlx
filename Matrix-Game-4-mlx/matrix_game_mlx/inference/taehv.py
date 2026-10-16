"""Tiny AutoEncoder for Hunyuan Video (TAEHV), Wan 2.2 5B flavour (`taew2_2`) — MLX port.

Ported from https://github.com/madebyollin/taehv (upstream commit 011dfc2112197741c540e0bdd5b7b67bcc930771,
MIT license), as vendored in Matrix-Game-4/matrix_game/inference/taehv.py. This file is the MLX port.

Layout: layers run channels-last (NHWC, frames folded into the batch); videos are NTHWC inside the network
(upstream: NTCHW). Only the path used by the `TAEHV` adapter is ported: `taew2_2` (patch size 2, 48 latent
channels, default non-`_super` decoder). Weights use upstream's `encoder.<i>` / `decoder.<i>` numbering, so the
`Sequential` stacks are plain Python lists of layers.
"""
from collections import namedtuple
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


class Conv2d(nn.Conv2d):
    """nn.Conv2d (NHWC) that accumulates in float32 and rounds the output once to the activation dtype.

    MLX framework constraint: some Metal bf16 conv2d kernel paths lose precision (decoder.8, 256->128 3x3, measured
    on a 120x216 input: 3e-2 rel L2 vs fp64, where torch CPU bf16 is 1.7e-3). Upcasting input and weight to float32, adding the
    bias in float32 and rounding the output reproduces torch's fp32-accumulate-then-round bf16 convolution. Parameters
    keep their dtype (bf16 in the realtime preset); float32 activations are unaffected."""

    def __call__(self, x):
        y = mx.conv2d(x.astype(mx.float32), self.weight.astype(mx.float32), self.stride, self.padding, self.dilation,
                      self.groups)
        if "bias" in self:
            y = y + self.bias.astype(mx.float32)
        return y.astype(x.dtype)


def conv(n_in, n_out, **kwargs):
    return Conv2d(n_in, n_out, 3, padding=1, **kwargs)


def pixel_unshuffle(x, r):
    """F.pixel_unshuffle on channels-last [..., H, W, C]: channel index of the result is c*r*r + i*r + j."""
    *lead, H, W, C = x.shape
    n = len(lead)
    x = x.reshape(*lead, H // r, r, W // r, r, C)
    x = x.transpose(*range(n), n, n + 2, n + 4, n + 1, n + 3)            # (h, w, C, i, j)
    return x.reshape(*lead, H // r, W // r, C * r * r)


def pixel_shuffle(x, r):
    """F.pixel_shuffle on channels-last [..., H, W, C*r*r] (inverse of `pixel_unshuffle`)."""
    *lead, H, W, C = x.shape
    n = len(lead)
    x = x.reshape(*lead, H, W, C // (r * r), r, r)
    x = x.transpose(*range(n), n, n + 3, n + 1, n + 4, n + 2)            # (h, i, w, j, C)
    return x.reshape(*lead, H * r, W * r, C // (r * r))


class Clamp(nn.Module):
    def __call__(self, x):
        return mx.tanh(x / 3) * 3


class Upsample(nn.Module):
    """nn.Upsample(scale_factor=k), nearest, on NHWC."""

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, x):
        if self.scale_factor == 1:
            return x
        return mx.repeat(mx.repeat(x, self.scale_factor, axis=1), self.scale_factor, axis=2)


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = [conv(n_in * 2, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out)]
        self.skip = Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU()

    def __call__(self, x, past):
        h = mx.concatenate([x, past], axis=-1)
        for layer in self.conv:
            h = layer(h)
        return self.act(h + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = Conv2d(n_f * stride, n_f, 1, bias=False)

    def __call__(self, x):
        # torch x.reshape(-1, stride*C, H, W) folds consecutive frames as [frame0 C | frame1 C]
        _NT, H, W, C = x.shape
        x = x.reshape(-1, self.stride, H, W, C).transpose(0, 2, 3, 1, 4).reshape(-1, H, W, self.stride * C)
        return self.conv(x)


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = Conv2d(n_f, n_f * stride, 1, bias=False)

    def __call__(self, x):
        # inverse of TPool: split the stride*C output channels into (stride, C) consecutive frames
        NT, H, W, C = x.shape
        x = self.conv(x)
        return x.reshape(NT, H, W, self.stride, C).transpose(0, 3, 1, 2, 4).reshape(NT * self.stride, H, W, C)


def _apply(layers, x):
    for b in layers:
        x = b(x)
    return x


def apply_model_with_memblocks_parallel(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with parallelization over the time axis and iteration over blocks.

    Args:
    - model: list of blocks to apply
    - x: input data, of dimensions NTHWC
    - show_progress_bar: unused (no tqdm in the port)

    Returns NTHWC array of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTHWC arrays, but got {x.ndim}-dim array"
    N, T, H, W, C = x.shape
    x = x.reshape(N * T, H, W, C)

    # parallel over input timesteps, iterate over blocks
    for b in model:
        if isinstance(b, MemBlock):
            NT, H, W, C = x.shape
            T = NT // N
            _x = x.reshape(N, T, H, W, C)
            # pad with zeros along time axis (i.e. empty memory), slice
            block_memory = mx.concatenate([mx.zeros_like(_x[:, :1]), _x], axis=1)[:, :T].reshape(x.shape)
            x = b(x, block_memory)
        else:
            x = b(x)
    NT, H, W, C = x.shape
    T = NT // N
    return x.reshape(N, T, H, W, C)


def apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar=None):
    """
    Process the work queue (a graph traversal over blocks and timesteps)
    until an output frame is produced or the queue is empty.
    Mutates memory and work_queue in place (memory entries are replaced, never modified in place).

    Returns N1HWC output array, or None if the queue needs more input.
    """
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return mx.expand_dims(xt, 1)
        b = model[i]
        if isinstance(b, MemBlock):
            # mem blocks are simple since we're visiting the graph in causal order
            if memory[i] is None:
                xt_new = b(xt, mx.zeros_like(xt))
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(b, TPool):
            # pool blocks accumulate inputs until they have enough to pool
            if memory[i] is None:
                memory[i] = []
            memory[i] = memory[i] + [xt]
            if len(memory[i]) > b.stride:
                raise ValueError(f"TPool memory overflow: {len(memory[i])} items for stride {b.stride}")
            elif len(memory[i]) == b.stride:
                N, H, W, C = xt.shape
                # [N, stride, H, W, C] -> [N*stride, H, W, C] (batch-major, as torch's cat(1).view)
                xt = b(mx.stack(memory[i], axis=1).reshape(N * b.stride, H, W, C))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            NT, H, W, C = xt.shape
            frames = xt.reshape(NT // b.stride, b.stride, H, W, C)
            for k in reversed(range(b.stride)):
                work_queue.insert(0, TWorkItem(frames[:, k], i + 1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    return None


def apply_model_with_memblocks_sequential(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with iteration over timesteps as well as blocks.

    Args:
    - model: list of blocks to apply
    - x: input data, of dimensions NTHWC
    - show_progress_bar: unused (no tqdm in the port)

    Returns NTHWC array of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTHWC arrays, but got {x.ndim}-dim array"
    work_queue = [TWorkItem(x[:, t], 0) for t in range(x.shape[1])]
    memory = [None] * len(model)
    out = []
    while work_queue:
        xt = apply_model_with_memblocks_sequential_single_step(model, memory, work_queue)
        if xt is not None:
            out.append(xt)
    return mx.concatenate(out, axis=1)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: list of blocks to apply
    - x: input data, of dimensions NTHWC
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: unused (no tqdm in the port)

    Returns NTHWC array of output data.
    """
    if parallel:
        return apply_model_with_memblocks_parallel(model, x, show_progress_bar)
    else:
        return apply_model_with_memblocks_sequential(model, x, show_progress_bar)


class _TAEHVCore(nn.Module):
    def __init__(self, checkpoint_path=None, encoder_time_downscale=None, decoder_time_upscale=None, decoder_space_upscale=None, patch_size=None, latent_channels=None, arch_name=None):
        """Build the `taew2_2` network (weights are loaded by `TAEHV.from_pack`, not here).

        Arg:
            checkpoint_path: only used to guess the architecture from its name (nothing is loaded).
            encoder_time_downscale: whether temporal downsampling is enabled for each block.
            decoder_time_upscale: whether temporal upsampling is enabled for each block.
            decoder_space_upscale: whether spatial upsampling is enabled for each block.
            patch_size: input/output pixelshuffle patch-size for this model.
            latent_channels: number of latent channels (z dim) for this model.
            arch_name: checkpoint name to take the architecture from; only "taew2_2" is ported.
        Architecture arguments left as None are guessed from the checkpoint name.
        """
        super().__init__()
        # each architecture setting is guessed from the checkpoint name, unless it was passed explicitly
        self.arch_name = str(arch_name or checkpoint_path or "")
        for marker in ("taehv1_5", "taeh3", "taeltx", "taecvx", "_super"):
            if marker in self.arch_name:
                raise NotImplementedError(f"only taew2_2 is ported, got {self.arch_name!r} ({marker})")
        if "taew2_2" in self.arch_name: patch_size, latent_channels = patch_size or 2, latent_channels or 48
        self.patch_size = patch_size or 1
        self.latent_channels = latent_channels or 16
        self.image_channels = 3
        encoder_time_downscale = encoder_time_downscale or (True, True, False)
        decoder_time_upscale = decoder_time_upscale or (False, True, True)
        decoder_space_upscale = decoder_space_upscale or (True, True, True)
        if len(decoder_time_upscale) == 2:
            decoder_time_upscale = (False, *decoder_time_upscale)
        self.encoder = [
            conv(self.image_channels*self.patch_size**2, 64), nn.ReLU(),
            TPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        ]
        n_f = [256, 128, 64, 64]
        self.decoder = [
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), Upsample(2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), Upsample(2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), Upsample(2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(), conv(n_f[3], self.image_channels*self.patch_size**2),
        ]
        # computed properties
        self.t_downscale = 2**sum(t.stride == 2 for t in self.encoder if isinstance(t, TPool))
        self.t_upscale = 2**sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
        self.frames_to_trim = self.t_upscale - 1

    def preprocess_input_frames(self, x):
        """Preprocess RGB input frames prior to the main encoder sequence."""
        if self.patch_size > 1: x = pixel_unshuffle(x, self.patch_size)
        return x

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        """Encode a sequence of frames.

        Args:
            x: input NTHWC RGB (C=3) array with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTHWC latent array with ~Gaussian values.
        """
        x = self.preprocess_input_frames(x)
        if x.shape[1] % self.t_downscale != 0:
            # pad at end to multiple of self.t_downscale
            n_pad = self.t_downscale - x.shape[1] % self.t_downscale
            padding = mx.repeat(x[:, -1:], n_pad, axis=1)
            x = mx.concatenate([x, padding], axis=1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def postprocess_output_frames(self, x):
        """Postprocess RGB frames after the main decoder sequence."""
        if self.patch_size > 1: x = pixel_shuffle(x, self.patch_size)
        return mx.clip(x, 0, 1)

    def decode_video(self, x, parallel=True, show_progress_bar=True):
        """Decode a sequence of frames.

        Args:
            x: input NTHWC latent (C=self.latent_channels) array with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTHWC RGB array with ~[0, 1] values.
        """
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = self.postprocess_output_frames(x)
        return x[:, self.frames_to_trim:]


class _StreamingTAEHV:
    def __init__(self, taehv):
        """Streaming wrapper around TAEHV for real-time use-cases (where not all inputs are available immediately).

        Same contract as upstream, on NTHWC arrays. Not an nn.Module on purpose: it only references the core,
        so it must not register the core's parameters a second time under the adapter.

        Decode-only (world model) usage:
            streaming = _StreamingTAEHV(taehv)
            while running:
                latent = world_model.step()       # latent represents t_upscale frames
                frame = streaming.decode(latent)   # returns first frame immediately
                while frame is not None:           # retrieve remaining frames from this latent
                    display(frame)
                    frame = streaming.decode()
        """
        self.taehv = taehv
        self.reset()

    def reset(self):
        """Reset all internal state. Call this to start encoding/decoding a new stream."""
        self.encoder_work_queue, self.encoder_memory = [], [None] * len(self.taehv.encoder)
        self.decoder_work_queue, self.decoder_memory = [], [None] * len(self.taehv.decoder)
        self.n_frames_encoded, self.n_frames_decoded = 0, 0
        self._last_encoder_input_frame = None

    def encode(self, x=None):
        """Feed an input frame (optional) and try to produce an encoder output.

        The encoder accumulates t_downscale input frames before producing one latent,
        so most calls will return None. Use flush_encoder() at end-of-stream to pad and
        drain any remaining latents.

        Args:
            x: NTHWC RGB frame array with values in [0, 1], or None to just process pending work.
        Returns: N1HWC latent array, or None if not enough input has been accumulated.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[4] == self.taehv.image_channels, f"Expected NTHWC frames but got {x.shape=}"
            self._last_encoder_input_frame = x[:, -1:]
            x = self.taehv.preprocess_input_frames(x)
            self.encoder_work_queue.extend(TWorkItem(x[:, t], 0) for t in range(x.shape[1]))
            self.n_frames_encoded += x.shape[1]
        xt = apply_model_with_memblocks_sequential_single_step(
            self.taehv.encoder, self.encoder_memory, self.encoder_work_queue)
        return xt

    def decode(self, x=None):
        """Feed a latent (optional) and try to produce a decoded frame.

        Each latent produces t_upscale output frames due to temporal upscaling. The first
        decode(latent) call returns the first of these frames; call decode() with no argument
        to retrieve the rest, one at a time. Each call does the minimum decoder work needed to
        produce one frame.

        Startup frames (the first frames_to_trim raw decoder outputs, used for causal alignment
        with the reference VAE) are consumed internally and never returned.

        Args:
            x: NTHWC latent array, or None to retrieve the next pending frame.
        Returns: N1HWC decoded RGB frame array, or None if the queue needs more input.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[4] == self.taehv.latent_channels, f"Expected NTHWC latents but got {x.shape=}"
            self.decoder_work_queue.extend(TWorkItem(x[:, t], 0) for t in range(x.shape[1]))
        while True:
            xt = apply_model_with_memblocks_sequential_single_step(
                self.taehv.decoder, self.decoder_memory, self.decoder_work_queue)
            if xt is None:
                return None
            self.n_frames_decoded += 1
            # skip startup frames (to match decode_video trim behavior)
            if self.n_frames_decoded <= self.taehv.frames_to_trim:
                continue
            return self.taehv.postprocess_output_frames(xt)

    def flush_encoder(self):
        """Pad (if needed) and drain all remaining latents from the encoder.

        Returns list of N1HWC latent arrays.
        """
        latents = []
        if self._last_encoder_input_frame is not None and self.n_frames_encoded % self.taehv.t_downscale != 0:
            n_pad = self.taehv.t_downscale - self.n_frames_encoded % self.taehv.t_downscale
            for _ in range(n_pad):
                lat = self.encode(self._last_encoder_input_frame)
                if lat is not None:
                    latents.append(lat)
        while (lat := self.encode()) is not None:
            latents.append(lat)
        return latents

    def flush_decoder(self):
        """Drain all remaining decoded frames from the decoder.

        Returns list of N1HWC decoded RGB frame arrays.
        """
        frames = []
        while (frame := self.decode()) is not None:
            frames.append(frame)
        return frames

    def flush(self):
        """Flush encoder (with padding) and decoder, returning all remaining decoded frames.

        Returns list of N1HWC decoded RGB frame arrays.
        """
        frames = []
        for latent in self.flush_encoder():
            frame = self.decode(latent)
            if frame is not None:
                frames.append(frame)
        frames.extend(self.flush_decoder())
        return frames


def _arrays(*trees):
    """Flatten arrays out of nested lists/tuples/dicts (None entries skipped), for mx.eval."""
    out = []
    for t in trees:
        if isinstance(t, mx.array):
            out.append(t)
        elif isinstance(t, dict):
            out.extend(_arrays(*t.values()))
        elif isinstance(t, (list, tuple)):
            out.extend(_arrays(*t))
    return out


class TAEHV(nn.Module):
    """Wan 2.2 5B TAEHV exposed through the pipeline's VAE interface.

    ``taew2_2`` is the only accepted architecture: it has patch size 2 and
    48 latent channels, matching Wan 2.2 5B.  In particular, this adapter
    refuses the 16-channel ``taehv``/Wan 2.1 checkpoint instead of silently
    producing incompatible latents.

    Encoding prefixes three first-frame copies to align the upstream pooling
    with Wan's causal 1+4k input boundaries. This is an approximate encoder;
    it does not reproduce the full Wan VAE's latent values exactly.

    Tensor contract: NCTHW in/out like upstream (video in [-1, 1], latents with 48 channels). Unlike upstream,
    ``encode``/``encode_stream`` return the latent array itself rather than an ``AutoencoderKLOutput`` (callers
    use ``.latent_dist.mode()`` upstream), and ``decode``/``decode_stream`` return the video array rather than a
    ``DecoderOutput`` (``.sample`` upstream).  Layers run on NHWC internally.
    """

    latent_channels = 48
    patch_size = 2
    spatial_compression_ratio = 16
    temporal_compression_ratio = 4

    def __init__(self, checkpoint_path=None, dtype=mx.bfloat16, *, arch_name=None, load=True, **kwargs):
        """Build the adapter; a ``.safetensors`` checkpoint is loaded as a pack (see ``from_pack``).

        Any other checkpoint path (e.g. a torch ``.pth``) raises ValueError unless ``load=False``, which only
        selects the architecture by name; weights must then be loaded into ``self.core`` by the caller.
        """
        super().__init__()
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for TAEHV")
        name = str(arch_name or checkpoint_path or "").lower()
        if "taew2_2" not in name:
            raise ValueError(
                "TAEHV for Wan 2.2 5B requires a taew2_2 checkpoint "
                "(48 channels); refusing an incompatible 16-channel checkpoint"
            )
        self.core = _TAEHVCore(
            checkpoint_path=str(checkpoint_path),
            arch_name="taew2_2",
            **kwargs,
        )
        if self.core.latent_channels != self.latent_channels:
            raise ValueError(
                f"TAEHV checkpoint exposes {self.core.latent_channels} latent "
                f"channels, expected {self.latent_channels}"
            )
        self.config = SimpleNamespace(
            latent_channels=self.latent_channels,
            z_dim=self.latent_channels,
            spatial_compression_ratio=self.spatial_compression_ratio,
            temporal_compression_ratio=self.temporal_compression_ratio,
            scaling_factor=1.0,
            latents_mean=[0.0] * self.latent_channels,
            latents_std=[1.0] * self.latent_channels,
        )
        if load:
            if not str(checkpoint_path).endswith(".safetensors"):
                raise ValueError(
                    f"cannot load weights from {checkpoint_path!s} (expected a .safetensors pack); "
                    "pass load=False to build the network without weights"
                )
            weights = mx.load(str(checkpoint_path))
            self.core.load_weights([(k.removeprefix("taehv."), v) for k, v in weights.items()], strict=True)
        self.core.set_dtype(dtype)
        self.clear_cache()

    @classmethod
    def from_pack(cls, path, dtype=mx.bfloat16):
        """Load ``taehv.safetensors`` from the MLX pack (keys ``taehv.<encoder|decoder>.<i>...``, strict)."""
        return cls(path, dtype, arch_name="taew2_2")

    @property
    def dtype(self):
        """Track the actual module dtype."""
        return self.core.encoder[0].weight.dtype

    def clear_cache(self):
        """Reset active incremental encoder and decoder request state."""
        self._streaming_encode_frames = 0
        self._streaming_encode_batch = None
        self._streaming_encode_memory = {}
        self._streaming_encoder = None
        stream = getattr(self, "_streaming_decoder", None)
        if stream is not None:
            stream.reset()
        self._streaming_decoder = None

    def begin_streaming_decode(self):
        """Start a causal decode session and return the adapter for chaining."""
        stream = getattr(self, "_streaming_decoder", None)
        if stream is not None:
            stream.reset()
        self._streaming_decoder = _StreamingTAEHV(self.core)
        return self

    def begin_streaming_encode(self):
        """Start a causal encoder request without disturbing decoder state."""
        self._streaming_encoder = True
        self._streaming_encode_frames = 0
        self._streaming_encode_batch = None
        self._streaming_encode_memory = {}
        return self

    def end_streaming_encode(self):
        """Release causal encoder state at the end of a request."""
        self._streaming_encoder = None
        self._streaming_encode_frames = 0
        self._streaming_encode_batch = None
        self._streaming_encode_memory = {}

    def end_streaming_decode(self):
        """End a causal decode session, including partially accumulated state."""
        stream = getattr(self, "_streaming_decoder", None)
        if stream is not None:
            stream.reset()
        self._streaming_decoder = None

    def encode_stream(self, x, **kwargs):
        """Encode one causal pixel block and retain state across calls.

        The first call accepts ``4k+1`` frames (three prefix copies are added)
        and continuation calls accept ``4k`` frames.  The returned latent is
        NCTHW (an array; upstream wraps it in ``AutoencoderKLOutput``, use
        ``.latent_dist.mode()`` there) with three latent frames for the usual 9/12-pixel blocks.
        The latent and the retained encoder memory are evaluated before returning.
        """
        del kwargs
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"Expected NCTHW RGB input, got {tuple(x.shape)}")
        frames = x.shape[2]
        if not hasattr(self, "_streaming_encoder") or self._streaming_encoder is None:
            self.begin_streaming_encode()
        first = self._streaming_encode_frames == 0
        required_mod = 1 if first else 0
        if frames < 1 or frames % self.temporal_compression_ratio != required_mod:
            expected = "4k+1" if first else "4k"
            raise ValueError(f"streaming TAEHV encoder {expected} input required, got {frames} frames")
        if self._streaming_encode_batch is None:
            self._streaming_encode_batch = x.shape[0]
        elif self._streaming_encode_batch != x.shape[0]:
            raise ValueError(
                f"streaming TAEHV batch size changed from {self._streaming_encode_batch} to {x.shape[0]}"
            )
        video = (x.astype(self.dtype).transpose(0, 2, 3, 4, 1) + 1) * 0.5
        if first:
            video = mx.concatenate([mx.repeat(video[:, :1], 3, axis=1), video], axis=1)
        if video.shape[1] % self.temporal_compression_ratio:
            raise ValueError("streaming TAEHV encoder block must align temporal pooling")
        memory = self._streaming_encode_memory
        latent = self._encode_parallel_chunk(video, memory)
        self._streaming_encode_frames += frames
        latent = latent.transpose(0, 4, 1, 2, 3)
        mx.eval(latent, _arrays(memory))
        return latent

    def _encode_parallel_chunk(self, video, memory):
        batch = video.shape[0]
        hidden = self.core.preprocess_input_frames(video)
        hidden = hidden.reshape(-1, *hidden.shape[2:])
        for index, layer in enumerate(self.core.encoder):
            if isinstance(layer, MemBlock):
                shaped = hidden.reshape(batch, -1, *hidden.shape[1:])
                past = memory.get(index)
                if past is None:
                    past = mx.zeros_like(shaped[:, :1])
                history = mx.concatenate([past, shaped[:, :-1]], axis=1).reshape(hidden.shape)
                # "+ 0" materializes a copy: a bare slice would keep the whole chunk's buffer alive
                memory[index] = shaped[:, -1:] + 0
                hidden = layer(hidden, history)
            else:
                hidden = layer(hidden)
        return hidden.reshape(batch, -1, *hidden.shape[1:])

    def _encode_chunked(self, x, chunk_frames=84):
        """Bound encoder activations while preserving memory across time chunks.

        Chunks include the initial three-frame prefix and end on complete
        temporal pooling groups. MemBlock history is its previous input, not
        its output; the retained last frame is a copy so it cannot hold a whole chunk.
        State is local to this call, including after an interrupted encode.
        Takes NCTHW, returns NTHWC.
        """
        if chunk_frames < 4 or chunk_frames % self.temporal_compression_ratio:
            raise ValueError("TAEHV encoder chunks must be positive multiples of four")
        batch, _, frames, _, _ = x.shape
        x = x.astype(self.dtype)
        memory = {}
        output = []
        start = 0
        while start < frames:
            prefix_frames = self.temporal_compression_ratio - 1 if start == 0 else 0
            end = min(start + chunk_frames - prefix_frames, frames)
            video = (x[:, :, start:end].transpose(0, 2, 3, 4, 1) + 1) * 0.5
            if prefix_frames:
                video = mx.concatenate([mx.repeat(video[:, :1], prefix_frames, axis=1), video], axis=1)
            output.append(self._encode_parallel_chunk(video, memory))
            mx.eval(output[-1], _arrays(memory))             # bound the graph (and the live chunk) per chunk
            start = end
        return mx.concatenate(output, axis=1)

    def encode(self, x, **kwargs):
        """Encode a whole 4k+1-frame NCTHW video in [-1, 1]; returns the NCTHW latent array."""
        del kwargs
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"Expected NCTHW RGB input, got {tuple(x.shape)}")
        if (x.shape[2] - 1) % self.temporal_compression_ratio:
            raise ValueError("Causal Wan TAEHV encoding requires 4k+1 input frames")
        if x.shape[2] + self.temporal_compression_ratio - 1 > 84:
            latent = self._encode_chunked(x)
        else:
            # Wan/diffusers: NCTHW, [-1, 1] -> TAEHV: NTHWC, [0, 1].
            video = (x.astype(self.dtype).transpose(0, 2, 3, 4, 1) + 1) * 0.5
            # Upstream pools [0..3], [4..7], ... and pads the *end*. Wan's
            # block-causal contract instead uses [0], [1..4], [5..8], ... .
            # Prefix copies align those boundaries without reading future frames.
            # Keep this local adaptation outside the retained upstream network.
            prefix = mx.repeat(video[:, :1], self.temporal_compression_ratio - 1, axis=1)
            video = mx.concatenate([prefix, video], axis=1)
            latent = self.core.encode_video(video, parallel=True, show_progress_bar=False)
        return latent.transpose(0, 4, 1, 2, 3)

    def decode(self, z, **kwargs):
        """Decode an NCTHW latent array; returns NCTHW RGB in [-1, 1]."""
        del kwargs
        if z.ndim != 5 or z.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected NCTHW latent with {self.latent_channels} channels, "
                f"got {tuple(z.shape)}"
            )
        latent = z.astype(self.dtype).transpose(0, 2, 3, 4, 1)
        video = self.core.decode_video(latent, parallel=True, show_progress_bar=False)
        # TAEHV returns NTHWC [0, 1]; the pipeline's decode helper applies its
        # usual ``/ 2 + .5`` conversion after this method.
        return (video * 2 - 1).transpose(0, 4, 1, 2, 3)

    def decode_stream(self, z, **kwargs):
        """Decode one latent block while retaining causal decoder state.

        The first block of F latent frames emits 4*F-3 pixel frames (nine for
        the normal three-latent block); continuation blocks emit 4*F pixel
        frames (twelve for three latents). The state
        is deliberately owned by this adapter so the pipeline can reset it at
        request boundaries and after failures. Returns an NCTHW array (upstream: ``DecoderOutput``);
        the frames and the retained decoder state are evaluated before returning.
        """
        del kwargs
        if z.ndim != 5 or z.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected NCTHW latent with {self.latent_channels} channels, "
                f"got {tuple(z.shape)}"
            )
        if not hasattr(self, "_streaming_decoder") or self._streaming_decoder is None:
            self.begin_streaming_decode()
        latent = z.astype(self.dtype).transpose(0, 2, 3, 4, 1)
        frames = []
        try:
            frame = self._streaming_decoder.decode(latent)
            while frame is not None:
                frames.append(frame)
                frame = self._streaming_decoder.decode()
        except Exception:
            self.end_streaming_decode()
            raise
        if frames:
            video = mx.concatenate(frames, axis=1)
        else:
            video = mx.zeros((latent.shape[0], 0, latent.shape[2] * self.spatial_compression_ratio,
                              latent.shape[3] * self.spatial_compression_ratio, 3), dtype=latent.dtype)
        video = (video * 2 - 1).transpose(0, 4, 1, 2, 3)
        stream = self._streaming_decoder
        mx.eval(video, _arrays(stream.decoder_memory, [tuple(w) for w in stream.decoder_work_queue]))
        return video

    def __call__(self, sample):
        return self.decode(self.encode(sample))
