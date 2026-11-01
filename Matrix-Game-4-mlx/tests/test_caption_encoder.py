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


def _tiny_t5_text():
    """A two-layer bf16 umT5-architecture encoder (random weights) behind the fake tokenizer interface; the mask
    length follows the caption, so captions differ in ids and mask."""
    from mlx.utils import tree_map
    from matrix_game_mlx.inference.wan.modules.t5 import T5Encoder
    mx.random.seed(0)
    model = T5Encoder(vocab=97, dim=64, dim_attn=64, dim_ffn=160, num_heads=4, num_layers=2, num_buckets=32,
                      shared_pos=False)
    model.update(tree_map(lambda a: (a + 0.1 * mx.random.normal(a.shape)).astype(mx.bfloat16), model.parameters()))

    def tokenize(texts, **kwargs):
        n = min(len(texts[0]), 512)
        ids = torch.zeros((1, 512), dtype=torch.int64)
        ids[0, :n] = torch.tensor([ord(c) % 97 for c in texts[0][:n]])
        mask = torch.zeros_like(ids)
        mask[0, :n] = 1
        return ids, mask

    return SimpleNamespace(text_encoder=SimpleNamespace(tokenizer=tokenize, model=model))


def test_graph_is_compiled_and_bitwise_native():
    """--t5-runtime graph (upstream: one CUDA graph replayed per caption) runs one mx.compile'd encoder traced once;
    its embeddings are bitwise the native path's."""
    text = _tiny_t5_text()
    native, graph = CaptionEncoder(text), CaptionEncoder(text, graph=True)
    for caption in ("a red car", "a much longer caption about a race at night", "x"):
        a, b = native(caption), graph(caption)
        assert b.dtype == mx.bfloat16 and b.shape == (1, 512, 64) and mx.array_equal(a, b), caption
    assert native.entry is None and (native.captures, native.replays) == (0, 0)
    assert graph.entry is not None and (graph.captures, graph.replays) == (1, 3)
    graph("a red car")                                   # a cache hit does not replay
    assert (graph.hits, graph.replays) == (1, 3)


def test_graph_sees_weight_updates():
    """The encoder weights are graph inputs: a replaced weight is used without a new trace."""
    text = _tiny_t5_text()
    graph = CaptionEncoder(text, graph=True)
    graph("first caption")
    model = text.text_encoder.model
    model.norm.weight = (model.norm.weight * 2).astype(mx.bfloat16)
    expected = CaptionEncoder(text)("second caption")
    assert mx.array_equal(graph("second caption"), expected) and graph.captures == 1


def test_t5_layer_norm_compiles_bitwise_and_keeps_native_values():
    """At umT5 width and residual magnitudes (|x| ~ 6e3 by block 3), mx.compile fuses mean()'s 1/n scale into the
    reduction and sums in another order; the variance is written sum / dim instead, which compiles bitwise and, dim
    being a power of two, is bitwise the former mean() (native embeddings unchanged)."""
    from matrix_game_mlx.inference.wan.modules.t5 import T5LayerNorm
    mx.random.seed(1)
    x = (mx.random.normal((1, 512, 4096)) * 1500).astype(mx.bfloat16)
    norm = T5LayerNorm(4096)
    norm.weight = (1 + 0.1 * mx.random.normal((4096,))).astype(mx.bfloat16)
    eager, compiled = norm(x), mx.compile(norm)(x)
    variance = (x.astype(mx.float32) ** 2).mean(axis=-1, keepdims=True)          # the former expression
    former = norm.weight * (x * mx.rsqrt(variance + norm.eps)).astype(mx.bfloat16)
    assert mx.array_equal(eager, former) and mx.array_equal(eager, compiled)


def test_graph_replays_count_the_tracing_encode():
    """`replays` counts every encode that ran the compiled encoder, the tracing one included (as CastleRuntime's
    counters; upstream's capture call also replays)."""
    graph = CaptionEncoder(_tiny_t5_text(), graph=True)
    graph("first caption")
    assert (graph.captures, graph.replays, graph.encodes) == (1, 1, 1)
    graph("second caption")
    assert (graph.captures, graph.replays, graph.encodes) == (1, 2, 2)


@pytest.mark.parametrize("dim", [64, 48, 4000])
def test_t5_layer_norm_keeps_mean_values_at_any_width(dim):
    """`sum / dim` is bitwise `mean()` only when 1/dim is exact (dim a power of two); other widths keep `mean()`, so
    the eager values equal the former expression at every width."""
    from matrix_game_mlx.inference.wan.modules.t5 import T5LayerNorm
    mx.random.seed(2)
    x = (mx.random.normal((1, 64, dim)) * 1500).astype(mx.bfloat16)
    norm = T5LayerNorm(dim)
    norm.weight = (1 + 0.1 * mx.random.normal((dim,))).astype(mx.bfloat16)
    variance = (x.astype(mx.float32) ** 2).mean(axis=-1, keepdims=True)
    former = norm.weight * (x * mx.rsqrt(variance + norm.eps)).astype(mx.bfloat16)
    assert mx.array_equal(norm(x), former)
