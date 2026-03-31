"""Validation tests for T5 encoder: MLX vs PyTorch."""
import sys
import os
import types
import numpy as np
import pytest
import importlib
import importlib.util

# Setup paths
MLX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PYTORCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Matrix-Game-3'))

# Load conftest helpers
_conftest_spec = importlib.util.spec_from_file_location(
    "conftest", os.path.join(os.path.dirname(__file__), "conftest.py"))
_conftest = importlib.util.module_from_spec(_conftest_spec)
_conftest_spec.loader.exec_module(_conftest)
assert_close = _conftest.assert_close
torch_to_mlx = _conftest.torch_to_mlx


def _load_module_with_mock_tokenizers(name, file_path):
    """Load a T5 module with tokenizer import mocked out."""
    import torch

    # Patch torch.cuda.current_device to avoid CUDA init on Mac
    _orig_current_device = torch.cuda.current_device
    torch.cuda.current_device = lambda: 'cpu'

    # Set up fake parent packages so relative imports resolve
    for pkg_name in ["wan", "wan.modules"]:
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []
            sys.modules[pkg_name] = pkg

    # Mock the tokenizers submodule
    mock_tokenizers = types.ModuleType("wan.modules.tokenizers")
    mock_tokenizers.HuggingfaceTokenizer = None
    sys.modules["wan.modules.tokenizers"] = mock_tokenizers

    spec = importlib.util.spec_from_file_location(
        f"wan.modules.{name}", file_path,
        submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "wan.modules"
    spec.loader.exec_module(mod)

    # Restore
    torch.cuda.current_device = _orig_current_device

    return mod


def _load_pt_t5():
    """Load PyTorch T5 module, mocking the tokenizer import."""
    pt_t5_path = os.path.join(PYTORCH_ROOT, "wan", "modules", "t5.py")
    return _load_module_with_mock_tokenizers("pt_t5", pt_t5_path)


def _load_mlx_t5():
    """Load MLX T5 module, mocking the tokenizer import."""
    mlx_t5_path = os.path.join(MLX_ROOT, "wan", "modules", "t5.py")
    return _load_module_with_mock_tokenizers("mlx_t5", mlx_t5_path)


def _copy_linear_weights(pt_linear, mlx_linear):
    """Copy weights from a PyTorch nn.Linear to an MLX nn.Linear."""
    import mlx.core as mx
    mlx_linear.weight = mx.array(pt_linear.weight.detach().cpu().float().numpy())


def _copy_embedding_weights(pt_emb, mlx_emb):
    """Copy weights from a PyTorch nn.Embedding to an MLX nn.Embedding."""
    import mlx.core as mx
    mlx_emb.weight = mx.array(pt_emb.weight.detach().cpu().float().numpy())


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: T5LayerNorm
# ──────────────────────────────────────────────────────────────────────────────

def test_t5_layer_norm():
    """Compare T5LayerNorm output between PyTorch and MLX."""
    import torch
    import mlx.core as mx

    pt_t5 = _load_pt_t5()
    mlx_t5 = _load_mlx_t5()

    dim = 64
    np.random.seed(42)
    x_np = np.random.randn(2, 8, dim).astype(np.float32)

    # PyTorch
    pt_norm = pt_t5.T5LayerNorm(dim)
    pt_norm.weight.data.fill_(1.0)
    x_pt = torch.from_numpy(x_np)
    out_pt = pt_norm(x_pt)

    # MLX
    mlx_norm = mlx_t5.T5LayerNorm(dim)
    mlx_norm.weight = mx.ones(dim)
    x_mx = mx.array(x_np)
    out_mx = mlx_norm(x_mx)

    assert_close(out_mx, out_pt, atol=1e-4, msg="T5LayerNorm mismatch")

    # Test with non-trivial weights
    w_np = np.random.randn(dim).astype(np.float32)
    pt_norm.weight.data = torch.from_numpy(w_np)
    mlx_norm.weight = mx.array(w_np)

    out_pt = pt_norm(x_pt)
    out_mx = mlx_norm(x_mx)
    assert_close(out_mx, out_pt, atol=1e-4, msg="T5LayerNorm with weights mismatch")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: T5Attention
# ──────────────────────────────────────────────────────────────────────────────

def test_t5_attention():
    """Compare T5Attention output between PyTorch and MLX."""
    import torch
    import mlx.core as mx

    pt_t5 = _load_pt_t5()
    mlx_t5 = _load_mlx_t5()

    dim, dim_attn, num_heads = 64, 64, 4
    B, L = 2, 8
    np.random.seed(42)
    x_np = np.random.randn(B, L, dim).astype(np.float32)

    # PyTorch
    pt_attn = pt_t5.T5Attention(dim, dim_attn, num_heads, dropout=0.0)
    pt_attn.train(False)

    # MLX
    mlx_attn = mlx_t5.T5Attention(dim, dim_attn, num_heads, dropout=0.0)

    # Copy weights
    _copy_linear_weights(pt_attn.q, mlx_attn.q)
    _copy_linear_weights(pt_attn.k, mlx_attn.k)
    _copy_linear_weights(pt_attn.v, mlx_attn.v)
    _copy_linear_weights(pt_attn.o, mlx_attn.o)

    # Forward
    x_pt = torch.from_numpy(x_np)
    x_mx = mx.array(x_np)

    with torch.no_grad():
        out_pt = pt_attn(x_pt)
    out_mx = mlx_attn(x_mx)

    assert_close(out_mx, out_pt, atol=1e-4, msg="T5Attention self-attn mismatch")

    # Test with mask
    mask_np = np.array([[1, 1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0]], dtype=np.float32)
    mask_pt = torch.from_numpy(mask_np)
    mask_mx = mx.array(mask_np)

    with torch.no_grad():
        out_pt = pt_attn(x_pt, mask=mask_pt)
    out_mx = mlx_attn(x_mx, mask=mask_mx)

    assert_close(out_mx, out_pt, atol=1e-4, msg="T5Attention with mask mismatch")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: T5FeedForward
# ──────────────────────────────────────────────────────────────────────────────

def test_t5_feed_forward():
    """Compare T5FeedForward output between PyTorch and MLX."""
    import torch
    import mlx.core as mx

    pt_t5 = _load_pt_t5()
    mlx_t5 = _load_mlx_t5()

    dim, dim_ffn = 64, 128
    B, L = 2, 8
    np.random.seed(42)
    x_np = np.random.randn(B, L, dim).astype(np.float32)

    # PyTorch
    pt_ffn = pt_t5.T5FeedForward(dim, dim_ffn, dropout=0.0)
    pt_ffn.train(False)

    # MLX
    mlx_ffn = mlx_t5.T5FeedForward(dim, dim_ffn, dropout=0.0)

    # Copy weights: gate is nn.Sequential(Linear, GELU) in PyTorch
    _copy_linear_weights(pt_ffn.gate[0], mlx_ffn.gate_linear)
    _copy_linear_weights(pt_ffn.fc1, mlx_ffn.fc1)
    _copy_linear_weights(pt_ffn.fc2, mlx_ffn.fc2)

    # Forward
    x_pt = torch.from_numpy(x_np)
    x_mx = mx.array(x_np)

    with torch.no_grad():
        out_pt = pt_ffn(x_pt)
    out_mx = mlx_ffn(x_mx)

    assert_close(out_mx, out_pt, atol=1e-4, msg="T5FeedForward mismatch")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: T5RelativeEmbedding
# ──────────────────────────────────────────────────────────────────────────────

def test_t5_relative_embedding():
    """Compare T5RelativeEmbedding position bias between PyTorch and MLX."""
    import torch
    import mlx.core as mx

    pt_t5 = _load_pt_t5()
    mlx_t5 = _load_mlx_t5()

    num_buckets, num_heads = 32, 4
    lq, lk = 8, 8

    # Bidirectional (encoder)
    pt_emb = pt_t5.T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
    mlx_emb = mlx_t5.T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
    _copy_embedding_weights(pt_emb.embedding, mlx_emb.embedding)

    with torch.no_grad():
        out_pt = pt_emb(lq, lk)
    out_mx = mlx_emb(lq, lk)

    assert_close(out_mx, out_pt, atol=1e-5, msg="T5RelativeEmbedding bidirectional mismatch")

    # Unidirectional (decoder)
    pt_emb2 = pt_t5.T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False)
    mlx_emb2 = mlx_t5.T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False)
    _copy_embedding_weights(pt_emb2.embedding, mlx_emb2.embedding)

    with torch.no_grad():
        out_pt2 = pt_emb2(lq, lk)
    out_mx2 = mlx_emb2(lq, lk)

    assert_close(out_mx2, out_pt2, atol=1e-5, msg="T5RelativeEmbedding unidirectional mismatch")

    # Test asymmetric lq != lk
    with torch.no_grad():
        out_pt3 = pt_emb(6, 10)
    out_mx3 = mlx_emb(6, 10)

    assert_close(out_mx3, out_pt3, atol=1e-5, msg="T5RelativeEmbedding asymmetric mismatch")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: T5SelfAttention (encoder block)
# ──────────────────────────────────────────────────────────────────────────────

def test_t5_encoder_block():
    """Compare a single T5SelfAttention encoder block between PyTorch and MLX."""
    import torch
    import mlx.core as mx

    pt_t5 = _load_pt_t5()
    mlx_t5 = _load_mlx_t5()

    dim, dim_attn, dim_ffn, num_heads, num_buckets = 64, 64, 128, 4, 32
    B, L = 2, 8
    np.random.seed(42)
    x_np = np.random.randn(B, L, dim).astype(np.float32)

    # Test with shared_pos=True (pos_bias provided externally)
    pt_block = pt_t5.T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                                     shared_pos=True, dropout=0.0)
    pt_block.train(False)

    mlx_block = mlx_t5.T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                                       shared_pos=True, dropout=0.0)

    # Copy weights
    mlx_block.norm1.weight = mx.array(pt_block.norm1.weight.detach().cpu().float().numpy())
    _copy_linear_weights(pt_block.attn.q, mlx_block.attn.q)
    _copy_linear_weights(pt_block.attn.k, mlx_block.attn.k)
    _copy_linear_weights(pt_block.attn.v, mlx_block.attn.v)
    _copy_linear_weights(pt_block.attn.o, mlx_block.attn.o)
    mlx_block.norm2.weight = mx.array(pt_block.norm2.weight.detach().cpu().float().numpy())
    _copy_linear_weights(pt_block.ffn.gate[0], mlx_block.ffn.gate_linear)
    _copy_linear_weights(pt_block.ffn.fc1, mlx_block.ffn.fc1)
    _copy_linear_weights(pt_block.ffn.fc2, mlx_block.ffn.fc2)

    # Create pos_bias from a shared relative embedding
    pt_pos_emb = pt_t5.T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
    mlx_pos_emb = mlx_t5.T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
    _copy_embedding_weights(pt_pos_emb.embedding, mlx_pos_emb.embedding)

    with torch.no_grad():
        pos_bias_pt = pt_pos_emb(L, L)
    pos_bias_mx = mlx_pos_emb(L, L)

    # Forward
    x_pt = torch.from_numpy(x_np)
    x_mx = mx.array(x_np)

    with torch.no_grad():
        out_pt = pt_block(x_pt, pos_bias=pos_bias_pt)
    out_mx = mlx_block(x_mx, pos_bias=pos_bias_mx)

    assert_close(out_mx, out_pt, atol=1e-4, msg="T5SelfAttention block mismatch")

    # Test with shared_pos=False (block has its own pos_embedding)
    pt_block2 = pt_t5.T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                                      shared_pos=False, dropout=0.0)
    pt_block2.train(False)

    mlx_block2 = mlx_t5.T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                                        shared_pos=False, dropout=0.0)

    # Copy weights
    mlx_block2.norm1.weight = mx.array(pt_block2.norm1.weight.detach().cpu().float().numpy())
    _copy_linear_weights(pt_block2.attn.q, mlx_block2.attn.q)
    _copy_linear_weights(pt_block2.attn.k, mlx_block2.attn.k)
    _copy_linear_weights(pt_block2.attn.v, mlx_block2.attn.v)
    _copy_linear_weights(pt_block2.attn.o, mlx_block2.attn.o)
    mlx_block2.norm2.weight = mx.array(pt_block2.norm2.weight.detach().cpu().float().numpy())
    _copy_linear_weights(pt_block2.ffn.gate[0], mlx_block2.ffn.gate_linear)
    _copy_linear_weights(pt_block2.ffn.fc1, mlx_block2.ffn.fc1)
    _copy_linear_weights(pt_block2.ffn.fc2, mlx_block2.ffn.fc2)
    _copy_embedding_weights(pt_block2.pos_embedding.embedding,
                            mlx_block2.pos_embedding.embedding)

    with torch.no_grad():
        out_pt2 = pt_block2(x_pt)
    out_mx2 = mlx_block2(x_mx)

    assert_close(out_mx2, out_pt2, atol=1e-4,
                 msg="T5SelfAttention block (non-shared pos) mismatch")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
