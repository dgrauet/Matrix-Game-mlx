"""Resident T5 caption encoder with bounded caption reuse.

MLX port of upstream ``matrix_game.inference.text``. There is no CUDA graph on
Apple Silicon, so ``graph=True`` takes the same native path (the flag is kept
for CLI/API parity).
"""
from collections import OrderedDict

import mlx.core as mx
import numpy as np


class CaptionEncoder:
    def __init__(self, text, *, graph=False, capacity=8):
        self.text, self.graph_enabled = text, graph
        self.capacity, self.cache = capacity, OrderedDict()
        self.encodes = self.hits = 0

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
        output = t5.model(ids, mask)
        encoded = mx.where((mask != 0)[..., None], output, 0).astype(mx.bfloat16)
        mx.eval(encoded)
        self.encodes += 1
        self.cache[prompt] = encoded
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return encoded
