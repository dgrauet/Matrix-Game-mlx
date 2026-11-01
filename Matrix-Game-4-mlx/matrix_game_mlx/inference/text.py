"""Resident T5 caption encoder with a fixed-length compiled graph and bounded caption reuse.

MLX port of upstream ``matrix_game.inference.text``. Upstream's ``graph=True`` captures the encoder once in a CUDA
graph and replays it for every new caption; here it is one ``mx.compile``d encoder call traced once (the padded
``(1, 512)`` ids and mask are its inputs, the encoder weights implicit inputs). mx.compile keeps the arithmetic of the
traced ops, so the embeddings are bitwise the native path's (tests/test_caption_encoder.py).
"""
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class CaptionEncoder:
    def __init__(self, text, *, graph=False, capacity=8):
        self.text, self.graph_enabled = text, graph
        self.capacity, self.cache = capacity, OrderedDict()
        self.entry = None
        self.encodes = self.hits = 0
        # Traces of the compiled encoder / encodes that ran it, the tracing encode included (as CastleRuntime's
        # counters; upstream's capture call also replays).
        self.captures = self.replays = 0

    def _graph(self, model):
        """The compiled encoder (upstream: the captured CUDA graph), built on first use for this model."""
        if self.entry is None or self.entry[0] is not model:
            def encode(ids, mask):
                self.captures += 1                     # Python runs only while tracing
                return model(ids, mask)
            # Weights are implicit inputs: a replaced weight array is read on the next call, without a new trace.
            if isinstance(model, nn.Module):
                inputs = [model.state]
            else:
                # Test-only: the fake encoders of tests/test_caption_encoder.py are plain callables without weights.
                inputs = []
            self.entry = (model, mx.compile(encode, inputs=inputs))
        return self.entry[1]

    def __call__(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192:
            raise ValueError("Expected a nonempty bounded caption")
        if prompt in self.cache:
            self.hits += 1
            self.cache.move_to_end(prompt)
            return self.cache[prompt]
        t5 = self.text.text_encoder
        ids, mask = t5.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids, mask = mx.array(np.asarray(ids)), mx.array(np.asarray(mask))
        if ids.shape != (1, 512) or mask.shape != ids.shape:
            raise ValueError("Caption encoding must retain native 512-token padding")
        if self.graph_enabled:
            output = self._graph(t5.model)(ids, mask)
            self.replays += 1
        else:
            output = t5.model(ids, mask)
        encoded = mx.where((mask != 0)[..., None], output, 0).astype(mx.bfloat16)
        mx.eval(encoded)
        self.encodes += 1
        self.cache[prompt] = encoded
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return encoded
