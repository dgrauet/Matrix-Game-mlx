from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch

from matrix_game_mlx.inference.text import CaptionEncoder


def _fake_text(length=512):
    def tokenize(texts, **kwargs):
        ids = torch.full((1, length), len(texts[0]), dtype=torch.int64)
        mask = torch.zeros_like(ids)
        mask[:, :3] = 1
        return ids, mask

    class Model:
        def __call__(self, ids, mask):
            return ids[..., None].astype(mx.float32) * mx.ones((1, 1, 4096))

    return SimpleNamespace(
        text_encoder=SimpleNamespace(tokenizer=tokenize, model=Model()))


def test_padding_reuse_and_bounded_cache():
    encoder = CaptionEncoder(_fake_text(), capacity=2)
    one = encoder("one")
    assert one.shape == (1, 512, 4096) and one.dtype == mx.bfloat16
    arr = np.array(one.astype(mx.float32))
    assert np.all(arr[:, :3] == 3)
    assert np.all(arr[:, 3:] == 0)
    assert encoder("one") is one
    encoder("two")
    encoder("three")
    assert (encoder.encodes, encoder.hits, len(encoder.cache)) == (3, 1, 2)
    assert "one" not in encoder.cache


def test_graph_flag_uses_native_path():
    encoder = CaptionEncoder(_fake_text(), graph=True)
    assert encoder.graph_enabled
    assert encoder("x").shape == (1, 512, 4096)


@pytest.mark.parametrize("invalid", [None, "", "  ", "a" * 8193])
def test_invalid_caption(invalid):
    with pytest.raises(ValueError, match="Expected a nonempty bounded caption"):
        CaptionEncoder(_fake_text())(invalid)


def test_requires_native_padding():
    with pytest.raises(ValueError,
                       match="Caption encoding must retain native 512-token padding"):
        CaptionEncoder(_fake_text(256))("hello")
