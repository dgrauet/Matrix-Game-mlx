"""T5 text encoder for MLX — ported from Matrix-Game-3/wan/modules/t5.py."""

import logging
import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]


def fp16_clamp(x: mx.array) -> mx.array:
    """Clamp values to avoid fp16 overflow."""
    if x.dtype == mx.float16:
        clamp = 65504.0 - 1000  # float16 max - 1000
        x = mx.clip(x, a_min=-clamp, a_max=clamp)
    return x


class GELU(nn.Module):
    """Custom GELU activation (tanh approximation)."""

    def __call__(self, x: mx.array) -> mx.array:
        return 0.5 * x * (1.0 + mx.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3.0))))


class T5LayerNorm(nn.Module):
    """RMS-style layer normalization used by T5."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        variance = (x.astype(mx.float32) ** 2).mean(axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        if self.weight.dtype in [mx.float16, mx.bfloat16]:
            x = x.astype(self.weight.dtype)
        return self.weight * x


class T5Attention(nn.Module):
    """Multi-head attention for T5 (no scaling)."""

    def __init__(self, dim: int, dim_attn: int, num_heads: int,
                 dropout: float = 0.1):
        assert dim_attn % num_heads == 0
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        # dropout is no-op for inference

    def __call__(self, x: mx.array, context: Optional[mx.array] = None,
                 mask: Optional[mx.array] = None,
                 pos_bias: Optional[mx.array] = None) -> mx.array:
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        context = x if context is None else context
        b, n, c = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).reshape(b, -1, n, c)
        k = self.k(context).reshape(b, -1, n, c)
        v = self.v(context).reshape(b, -1, n, c)

        # attention bias: [B, N, Lq, Lk]
        attn_bias = mx.zeros((b, n, q.shape[1], k.shape[1]))
        if pos_bias is not None:
            attn_bias = attn_bias + pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            if mask.ndim == 2:
                mask = mask.reshape(b, 1, 1, -1)
            else:
                mask = mx.expand_dims(mask, axis=1)
            min_val = mx.finfo(x.dtype).min
            attn_bias = mx.where(mask != 0, attn_bias, min_val)

        # compute attention (T5 does not use scaling)
        # einsum('binc,bjnc->bnij', q, k)
        q_t = mx.transpose(q, axes=(0, 2, 1, 3))  # [B, N, Lq, C]
        k_t = mx.transpose(k, axes=(0, 2, 3, 1))  # [B, N, C, Lk]
        attn = q_t @ k_t + attn_bias               # [B, N, Lq, Lk]

        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(attn.dtype)

        # einsum('bnij,bjnc->binc', attn, v)
        v_t = mx.transpose(v, axes=(0, 2, 1, 3))   # [B, N, Lk, C]
        x = attn @ v_t                              # [B, N, Lq, C]
        x = mx.transpose(x, axes=(0, 2, 1, 3))     # [B, Lq, N, C]

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        return x


class T5FeedForward(nn.Module):
    """Gated feed-forward network for T5."""

    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers — gate is nn.Sequential(nn.Linear, GELU) in PyTorch
        # Store as list so weight key is "gate.0.weight" matching PyTorch
        self.gate = [nn.Linear(dim, dim_ffn, bias=False)]
        self._gate_act = GELU()
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x) * self._gate_act(self.gate[0](x))
        x = self.fc2(x)
        return x


class T5SelfAttention(nn.Module):
    """Self-attention block with FFN for T5 encoder."""

    def __init__(self, dim: int, dim_attn: int, dim_ffn: int, num_heads: int,
                 num_buckets: int, shared_pos: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 pos_bias: Optional[mx.array] = None) -> mx.array:
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.shape[1], x.shape[1])
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):
    """Cross-attention block for T5 decoder."""

    def __init__(self, dim: int, dim_attn: int, dim_ffn: int, num_heads: int,
                 num_buckets: int, shared_pos: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 encoder_states: Optional[mx.array] = None,
                 encoder_mask: Optional[mx.array] = None,
                 pos_bias: Optional[mx.array] = None) -> mx.array:
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.shape[1], x.shape[1])
        x = fp16_clamp(x + self.self_attn(
            self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):
    """Relative position bias for T5 attention."""

    def __init__(self, num_buckets: int, num_heads: int,
                 bidirectional: bool, max_dist: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def __call__(self, lq: int, lk: int) -> mx.array:
        rel_pos = mx.arange(lk).reshape(1, -1) - mx.arange(lq).reshape(-1, 1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        # [Lq, Lk, N] -> [N, Lq, Lk] -> [1, N, Lq, Lk]
        rel_pos_embeds = mx.transpose(rel_pos_embeds, axes=(2, 0, 1))
        rel_pos_embeds = mx.expand_dims(rel_pos_embeds, axis=0)
        return rel_pos_embeds

    def _relative_position_bucket(self, rel_pos: mx.array) -> mx.array:
        """Compute relative position bucket indices."""
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(mx.int32) * num_buckets
            rel_pos = mx.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = mx.zeros(rel_pos.shape, dtype=mx.int32)
            rel_pos = -mx.minimum(rel_pos, mx.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (
            mx.log(rel_pos.astype(mx.float32) / max_exact) /
            math.log(self.max_dist / max_exact) *
            (num_buckets - max_exact)
        ).astype(mx.int32)
        rel_pos_large = mx.minimum(
            rel_pos_large,
            mx.full(rel_pos_large.shape, num_buckets - 1, dtype=mx.int32))
        rel_buckets = rel_buckets + mx.where(
            rel_pos < max_exact,
            rel_pos.astype(mx.int32),
            rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):
    """T5 encoder stack."""

    def __init__(self, vocab: int, dim: int, dim_attn: int, dim_ffn: int,
                 num_heads: int, num_layers: int, num_buckets: int,
                 shared_pos: bool = True, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        if isinstance(vocab, nn.Embedding):
            self.token_embedding = vocab
        else:
            self.token_embedding = nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.blocks = [
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ]
        self.norm = T5LayerNorm(dim)

    def __call__(self, ids: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        x = self.token_embedding(ids)
        e = self.pos_embedding(
            x.shape[1], x.shape[1]) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        return x


class T5Decoder(nn.Module):
    """T5 decoder stack."""

    def __init__(self, vocab: int, dim: int, dim_attn: int, dim_ffn: int,
                 num_heads: int, num_layers: int, num_buckets: int,
                 shared_pos: bool = True, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        if isinstance(vocab, nn.Embedding):
            self.token_embedding = vocab
        else:
            self.token_embedding = nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.blocks = [
            T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                             shared_pos, dropout) for _ in range(num_layers)
        ]
        self.norm = T5LayerNorm(dim)

    def __call__(self, ids: mx.array, mask: Optional[mx.array] = None,
                 encoder_states: Optional[mx.array] = None,
                 encoder_mask: Optional[mx.array] = None) -> mx.array:
        b, s = ids.shape

        # causal mask
        if mask is None:
            mask = mx.tril(mx.ones((1, s, s)))
        elif mask.ndim == 2:
            mask = mx.tril(
                mx.broadcast_to(mx.expand_dims(mask, axis=1), (b, s, s)))

        # layers
        x = self.token_embedding(ids)
        e = self.pos_embedding(
            x.shape[1], x.shape[1]) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        return x


class T5Model(nn.Module):
    """Full T5 model (encoder + decoder)."""

    def __init__(self, vocab_size: int, dim: int, dim_attn: int, dim_ffn: int,
                 num_heads: int, encoder_layers: int, decoder_layers: int,
                 num_buckets: int, shared_pos: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, encoder_ids: mx.array, encoder_mask: mx.array,
                 decoder_ids: mx.array,
                 decoder_mask: mx.array) -> mx.array:
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(name: str,
        encoder_only: bool = False,
        decoder_only: bool = False,
        return_tokenizer: bool = False,
        tokenizer_kwargs: dict = {},
        dtype: mx.Dtype = mx.float32,
        **kwargs):
    """Internal factory for T5 models."""
    assert not (encoder_only and decoder_only)

    if encoder_only:
        model_cls = T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        model_cls = T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model

    model = model_cls(**kwargs)

    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    """Factory for UMT5-XXL configuration."""
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:
    """Wrapper that loads weights and runs T5 encoding for inference."""

    def __init__(
        self,
        text_len: int,
        dtype: mx.Dtype = mx.bfloat16,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ):
        self.text_len = text_len
        self.dtype = dtype

        # init model
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype)
        logging.info(f'loading {checkpoint_path}')
        if checkpoint_path:
            weights = mx.load(checkpoint_path)
            model.load_weights(list(weights.items()))
        self.model = model

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts: List[str]) -> List[mx.array]:
        """Encode texts and return list of context arrays."""
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        ids = mx.array(ids.numpy())
        mask = mx.array(mask.numpy())
        seq_lens = mask.astype(mx.int32).sum(axis=1)
        context = self.model(ids, mask)
        # zero out padding
        result = []
        for i, v in enumerate(seq_lens.tolist()):
            ctx = context[i]
            if v < ctx.shape[0]:
                ctx = mx.concatenate([
                    ctx[:v],
                    mx.zeros((ctx.shape[0] - v, ctx.shape[-1]),
                             dtype=ctx.dtype)
                ], axis=0)
            result.append(ctx)
        return result
