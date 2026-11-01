"""Compiled graphs around the existing online TAEHV, without changing time alignment.

MLX port of upstream ``matrix_game.inference.vae_graph`` (``--vae-graph``, default on for realtime + TAEHV). Upstream
captures one CUDA graph for the steady encoder chunk (4 frames) and one for the steady decoder latent, copying the
retained MemBlock state into persistent buffers around each replay. Here each graph is one ``mx.compile``d function
traced once: the chunk and the retained state are its inputs and the new state is returned (MLX arrays are
immutable, so there are no persistent buffers to copy), and the TAEHV weights are implicit inputs. mx.compile keeps
the arithmetic of the traced ops (the convolutions are opaque to it), so latents, frames and retained state are
bitwise the eager adapter's (tests/test_taehv.py). Only batch 1, one latent per chunk is supported, as upstream; the
first chunk of a stream runs the eager adapter, as upstream.
"""
import mlx.core as mx

from matrix_game_mlx.inference.taehv import TWorkItem, apply_model_with_memblocks_sequential_single_step


class StatefulGraph:
    """One compiled function `fn(value, state) -> (output, state)` for a fixed input signature (upstream: a captured
    CUDA graph with persistent memory buffers). `state` is the list of retained arrays, in a fixed order."""

    def __init__(self, fn, example, inputs):
        self.signature = (tuple(example.shape), example.dtype)
        self.fn = mx.compile(fn, inputs=inputs)
        self.replays = 0

    def __call__(self, value, state):
        if (tuple(value.shape), value.dtype) != self.signature:
            raise ValueError("VAE graph signature changed")
        output, state = self.fn(value, state)
        self.replays += 1
        return output, state


class StreamingVAEGraph:
    """`TAEHV` with compiled steady-state streaming encode/decode (upstream: CUDA graphs). Any other attribute
    (`dtype`, `decode`, `config`, ...) is the base adapter's."""

    def __init__(self, base):
        self.base = base
        self.encoder_graph = self.decoder_graph = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def clear_cache(self):
        self.base.clear_cache()

    def encode(self, *args, **kwargs):
        return self.base.encode(*args, **kwargs)

    def prime(self):
        """Build both graphs before gameplay on zero frames (first chunk eager, second compiled), then reset the
        streaming state, as upstream."""
        if self.encoder_graph is not None and self.decoder_graph is not None:
            return
        try:
            self.clear_cache()
            for frames in (1, 4):
                self.decode_stream(self.encode_stream(mx.zeros((1, 3, frames, 480, 864), self.base.dtype)))
        finally:
            self.clear_cache()

    @staticmethod
    def _check(value, channels, frames):
        if value.ndim != 5 or tuple(value.shape[:3]) != (1, channels, frames):
            raise ValueError("VAE graph requires batch1/chunk1")

    def encode_stream(self, pixels):
        base = self.base
        self._check(pixels, 3, 1 if base._streaming_encode_frames == 0 else 4)
        if not base._streaming_encode_memory:
            return base.encode_stream(pixels)
        memory = base._streaming_encode_memory
        if self.encoder_graph is None:
            indices = sorted(memory)

            def encode(value, state):
                work = dict(zip(indices, state))
                video = (value.astype(base.dtype).transpose(0, 2, 3, 4, 1) + 1) * 0.5
                latent = base._encode_parallel_chunk(video, work).transpose(0, 4, 1, 2, 3)
                return latent, [work[i] for i in indices]

            self.encoder_graph = StatefulGraph(encode, pixels, [base.core.state])
            self.encoder_indices = indices
        latent, state = self.encoder_graph(pixels, [memory[i] for i in self.encoder_indices])
        memory.update(zip(self.encoder_indices, state))
        base._streaming_encode_frames += pixels.shape[2]
        mx.eval(latent, state)
        return latent

    def decode_stream(self, latent):
        base = self.base
        self._check(latent, base.latent_channels, 1)
        stream = base._streaming_decoder
        if stream is None:
            return base.decode_stream(latent)
        if stream.decoder_work_queue:
            raise ValueError("Decoder contains pending frames")
        if self.decoder_graph is None:
            core = base.core
            indices = [i for i, value in enumerate(stream.decoder_memory) if isinstance(value, mx.array)]

            def decode(value, state):
                memory = [None] * len(core.decoder)
                for i, past in zip(indices, state):
                    memory[i] = past
                queue = [TWorkItem(value.astype(base.dtype).transpose(0, 2, 3, 4, 1)[:, 0], 0)]
                frames = []
                while queue:
                    frame = apply_model_with_memblocks_sequential_single_step(core.decoder, memory, queue)
                    if frame is not None:
                        frames.append(core.postprocess_output_frames(frame))
                video = (mx.concatenate(frames, axis=1) * 2 - 1).transpose(0, 4, 1, 2, 3)
                return video, [memory[i] for i in indices]

            self.decoder_graph = StatefulGraph(decode, latent, [core.state])
            self.decoder_indices = indices
        video, state = self.decoder_graph(latent, [stream.decoder_memory[i] for i in self.decoder_indices])
        for i, value in zip(self.decoder_indices, state):
            stream.decoder_memory[i] = value
        stream.n_frames_decoded += video.shape[2]
        mx.eval(video, state)
        return video
