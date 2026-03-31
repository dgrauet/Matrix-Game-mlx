"""Validation tests for ActionModule: mouse/keyboard conditioning.

Compares MLX implementation against PyTorch reference with shared weights.
Uses small dimensions to keep tests fast.
"""
import sys
import os
import numpy as np
import pytest

# -- path setup ---------------------------------------------------------------
# MLX root must come first so `wan.modules` resolves to the MLX package.
# Do NOT add PYTORCH_ROOT to sys.path — the PyTorch `wan/__init__.py` imports
# diffusers and other heavy deps. PyTorch reference code is loaded via importlib.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PYTORCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Matrix-Game-3'))

import importlib.util
_conftest_spec = importlib.util.spec_from_file_location(
    "conftest", os.path.join(os.path.dirname(__file__), "conftest.py"))
_conftest = importlib.util.module_from_spec(_conftest_spec)
_conftest_spec.loader.exec_module(_conftest)
assert_close = _conftest.assert_close
torch_to_mlx = _conftest.torch_to_mlx

import mlx.core as mx
import torch

from wan.modules.action_module import ActionModule as ActionModuleMLX


# -- PyTorch reference --------------------------------------------------------
import torch.nn as pt_nn
from einops import rearrange

# CPU-compatible PyTorch posemb functions (the reference uses torch.cuda.current_device())
from typing import Union, Tuple, List


def _to_tuple_pt(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd_pt(start, *args, dim=2):
    if len(args) == 0:
        num = _to_tuple_pt(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple_pt(start, dim=dim)
        stop = _to_tuple_pt(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple_pt(start, dim=dim)
        stop = _to_tuple_pt(args[0], dim=dim)
        num = _to_tuple_pt(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]  # CPU
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)
    return grid


def get_1d_rotary_pos_embed_pt(dim, pos, theta=10000.0, use_real=False,
                                theta_rescale_factor=1.0, interpolation_factor=1.0):
    if isinstance(pos, int):
        pos = torch.arange(pos).float()  # CPU
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # CPU
    freqs = torch.outer(pos * interpolation_factor, freqs)
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos, freqs_sin
    else:
        return torch.polar(torch.ones_like(freqs), freqs)


def get_nd_rotary_pos_embed_pt(rope_dim_list, start, *args, theta=10000.0,
                                use_real=False, theta_rescale_factor=1.0,
                                interpolation_factor=1.0):
    grid = get_meshgrid_nd_pt(start, *args, dim=len(rope_dim_list))
    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed_pt(
            rope_dim_list[i], grid[i].reshape(-1), theta,
            use_real=use_real, theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)
    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)
        sin = torch.cat([emb[1] for emb in embs], dim=1)
        return cos, sin
    else:
        return torch.cat(embs, dim=1)


def reshape_for_broadcast_pt(freqs_cis, x, head_first=False):
    ndim = x.ndim
    if isinstance(freqs_cis, tuple):
        if head_first:
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        if head_first:
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half_pt(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb_pt(xq, xk, freqs_cis, head_first=False):
    assert isinstance(freqs_cis, tuple)
    cos, sin = reshape_for_broadcast_pt(freqs_cis, xq, head_first)
    xq_out = (xq.float() * cos[:, :xq.shape[1], :, :] + rotate_half_pt(xq.float()) * sin[:, :xq.shape[1], :, :]).type_as(xq)
    xk_out = (xk.float() * cos[:, :xk.shape[1], :, :] + rotate_half_pt(xk.float()) * sin[:, :xk.shape[1], :, :]).type_as(xk)
    return xq_out, xk_out


class WanRMSNorm_pt(pt_nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = pt_nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class ActionModule_pt(pt_nn.Module):
    """Minimal inline copy of the PyTorch ActionModule for testing."""

    def __init__(
        self,
        mouse_dim_in=2, keyboard_dim_in=6, hidden_size=128,
        img_hidden_size=1536, keyboard_hidden_dim=1024, mouse_hidden_dim=1024,
        vae_time_compression_ratio=4, windows_size=3, heads_num=16,
        patch_size=[1, 2, 2], qk_norm=True, qkv_bias=False,
        rope_dim_list=[8, 28, 28], rope_theta=256,
        mouse_qk_dim_list=[8, 28, 28], enable_mouse=True,
        enable_keyboard=True, blocks=[], local_attn_size=6,
    ):
        super().__init__()
        self.local_attn_size = local_attn_size
        self.enable_mouse = enable_mouse
        self.enable_keyboard = enable_keyboard
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.mouse_qk_dim_list = mouse_qk_dim_list
        self.heads_num = heads_num
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.patch_size = patch_size

        if self.enable_keyboard:
            self.keyboard_embed = pt_nn.Sequential(
                pt_nn.Linear(keyboard_dim_in, hidden_size, bias=True),
                pt_nn.SiLU(),
                pt_nn.Linear(hidden_size, hidden_size, bias=True),
            )

        if self.enable_mouse:
            c = mouse_hidden_dim
            self.mouse_mlp = pt_nn.Sequential(
                pt_nn.Linear(mouse_dim_in * vae_time_compression_ratio * windows_size + img_hidden_size, c, bias=True),
                pt_nn.GELU(approximate="tanh"),
                pt_nn.Linear(c, c),
                pt_nn.LayerNorm(c),
            )
            head_dim = c // heads_num
            self.t_qkv = pt_nn.Linear(c, c * 3, bias=qkv_bias)
            self.img_attn_q_norm = WanRMSNorm_pt(head_dim, eps=1e-6) if qk_norm else pt_nn.Identity()
            self.img_attn_k_norm = WanRMSNorm_pt(head_dim, eps=1e-6) if qk_norm else pt_nn.Identity()
            self.proj_mouse = pt_nn.Linear(c, img_hidden_size, bias=qkv_bias)

        if self.enable_keyboard:
            head_dim_key = keyboard_hidden_dim // heads_num
            self.key_attn_q_norm = WanRMSNorm_pt(head_dim_key, eps=1e-6) if qk_norm else pt_nn.Identity()
            self.key_attn_k_norm = WanRMSNorm_pt(head_dim_key, eps=1e-6) if qk_norm else pt_nn.Identity()
            self.mouse_attn_q = pt_nn.Linear(img_hidden_size, keyboard_hidden_dim, bias=qkv_bias)
            self.keyboard_attn_kv = pt_nn.Linear(
                hidden_size * windows_size * vae_time_compression_ratio,
                keyboard_hidden_dim * 2, bias=qkv_bias,
            )
            self.proj_keyboard = pt_nn.Linear(keyboard_hidden_dim, img_hidden_size, bias=qkv_bias)

    def get_rotary_pos_embed(self, video_length, height, width, head_dim, rope_dim_list=None):
        target_ndim = 3
        latents_size = [video_length, height, width]
        if isinstance(self.patch_size, int):
            rope_sizes = [s // self.patch_size for s in latents_size]
        elif isinstance(self.patch_size, list):
            rope_sizes = [s // self.patch_size[idx] for idx, s in enumerate(latents_size)]
        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed_pt(
            rope_dim_list, rope_sizes, theta=self.rope_theta,
            use_real=True, theta_rescale_factor=1,
        )
        cutoff = video_length * rope_sizes[1] * rope_sizes[2] // self.patch_size[0]
        return freqs_cos[-cutoff:], freqs_sin[-cutoff:]

    def forward(self, x, tt, th, tw, mouse_condition=None, keyboard_condition=None,
                mouse_cond_memory=None, keyboard_cond_memory=None):
        B, N_frames, C = keyboard_condition.shape
        assert tt * th * tw == x.shape[1]
        cond1 = ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0
        cond2 = N_frames % self.vae_time_compression_ratio == 0
        assert cond1 or cond2
        if cond1:
            N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1
        else:
            N_feats = N_frames // self.vae_time_compression_ratio

        pad_t = self.vae_time_compression_ratio * self.windows_size
        memory_length = 0

        if self.enable_mouse and mouse_condition is not None:
            hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=tt, S=th * tw)
            B, N_frames, C = mouse_condition.shape
            if cond1:
                mouse_condition = torch.cat([mouse_condition[:, 0:1, :].repeat(1, pad_t, 1), mouse_condition], dim=1)
            else:
                mouse_condition = torch.cat([mouse_condition[:, 0:1, :].repeat(1, pad_t - 4, 1), mouse_condition], dim=1)
            group_mouse = [mouse_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t, :] for i in range(N_feats)]
            group_mouse = torch.stack(group_mouse, dim=1)
            if mouse_cond_memory is not None:
                memory_length = mouse_cond_memory.shape[1]
                mouse_cond_memory = mouse_cond_memory.unsqueeze(2).repeat(1, 1, pad_t, 1)
                group_mouse = torch.cat([mouse_cond_memory, group_mouse], dim=1)
            group_mouse = group_mouse.unsqueeze(-1).repeat(1, 1, 1, 1, th * tw)
            group_mouse = rearrange(group_mouse, 'b t window d s -> (b s) t (window d)')
            group_mouse = torch.cat([hidden_states, group_mouse], dim=-1)
            group_mouse = self.mouse_mlp(group_mouse)
            mouse_qkv = self.t_qkv(group_mouse)
            q, k, v = rearrange(mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
            q = self.img_attn_q_norm(q).to(v)
            k = self.img_attn_k_norm(k).to(v)
            if memory_length > 0:
                freqs_cos_mem, freqs_sin_mem = self.get_rotary_pos_embed(memory_length, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
                qq_mem, kk_mem = apply_rotary_emb_pt(q[:, :memory_length], k[:, :memory_length], (freqs_cos_mem, freqs_sin_mem), head_first=False)
                q[:, :memory_length, :], k[:, :memory_length, :] = qq_mem, kk_mem
                freqs_cos_pred, freqs_sin_pred = self.get_rotary_pos_embed(tt - memory_length, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
                qq_pred, kk_pred = apply_rotary_emb_pt(q[:, memory_length:], k[:, memory_length:], (freqs_cos_pred, freqs_sin_pred), head_first=False)
                q[:, memory_length:, :], k[:, memory_length:, :] = qq_pred, kk_pred
            else:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed(tt, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
                qq, kk = apply_rotary_emb_pt(q, k, (freqs_cos, freqs_sin), head_first=False)
                q, k = qq, kk
            q_pt = q.transpose(1, 2)
            k_pt = k.transpose(1, 2)
            v_pt = v.transpose(1, 2)
            attn = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt).transpose(1, 2).contiguous()
            B_orig = x.shape[0]
            attn = rearrange(attn, '(b S) T h d -> b (T S) (h d)', b=B_orig)
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B_orig)
            attn = self.proj_mouse(attn)
            hidden_states = hidden_states + attn
        else:
            hidden_states = x
            B_orig = x.shape[0]

        if self.enable_keyboard and keyboard_condition is not None:
            if cond1:
                keyboard_condition = torch.cat([keyboard_condition[:, 0:1, :].repeat(1, pad_t, 1), keyboard_condition], dim=1).to(self.keyboard_embed[0].weight.dtype)
            else:
                keyboard_condition = torch.cat([keyboard_condition[:, 0:1, :].repeat(1, pad_t - 4, 1), keyboard_condition], dim=1).to(self.keyboard_embed[0].weight.dtype)
            keyboard_condition = self.keyboard_embed(keyboard_condition)
            group_keyboard = [keyboard_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t, :] for i in range(N_feats)]
            group_keyboard = torch.stack(group_keyboard, dim=1)
            if keyboard_cond_memory is not None:
                memory_length = keyboard_cond_memory.shape[1]
                keyboard_cond_memory = self.keyboard_embed(keyboard_cond_memory)
                keyboard_cond_memory = keyboard_cond_memory.unsqueeze(2).repeat(1, 1, pad_t, 1)
                group_keyboard = torch.cat([keyboard_cond_memory, group_keyboard], dim=1)
            group_keyboard = group_keyboard.reshape(group_keyboard.shape[0], group_keyboard.shape[1], -1)
            mouse_q = self.mouse_attn_q(hidden_states)
            keyboard_kv = self.keyboard_attn_kv(group_keyboard)
            q = rearrange(mouse_q, "B L (H D) -> B L H D", H=self.heads_num)
            k, v = rearrange(keyboard_kv, "B L (K H D) -> K B L H D", K=2, H=self.heads_num)
            q = self.key_attn_q_norm(q).to(v)
            k = self.key_attn_k_norm(k).to(v)
            S = th * tw
            q = rearrange(q, "B (T S) H D -> (B S) T H D", S=S)
            if memory_length > 0:
                freqs_cos_mem, freqs_sin_mem = self.get_rotary_pos_embed(memory_length, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
                qq_mem, kk_mem = apply_rotary_emb_pt(q[:, :memory_length], k[:, :memory_length], (freqs_cos_mem, freqs_sin_mem), head_first=False)
                q[:, :memory_length, :], k[:, :memory_length, :] = qq_mem, kk_mem
                freqs_cos_pred, freqs_sin_pred = self.get_rotary_pos_embed(tt - memory_length, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
                qq_pred, kk_pred = apply_rotary_emb_pt(q[:, memory_length:], k[:, memory_length:], (freqs_cos_pred, freqs_sin_pred), head_first=False)
                q[:, memory_length:, :], k[:, memory_length:, :] = qq_pred, kk_pred
            else:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed(tt, self.patch_size[1], self.patch_size[2], k.shape[-1], self.rope_dim_list)
                qq, kk = apply_rotary_emb_pt(q, k, (freqs_cos, freqs_sin), head_first=False)
                q, k = qq, kk
            k = k.repeat(S, 1, 1, 1)
            v = v.repeat(S, 1, 1, 1)
            q_pt = q.transpose(1, 2)
            k_pt = k.transpose(1, 2)
            v_pt = v.transpose(1, 2)
            attn = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=False).transpose(1, 2).contiguous()
            attn = rearrange(attn, '(B S) T H D -> B (T S) (H D)', S=S)
            attn = self.proj_keyboard(attn)
            hidden_states = hidden_states + attn

        return hidden_states


# =============================================================================
# Weight copying utilities
# =============================================================================

def copy_weights_pt_to_mlx(pt_model, mlx_model, config):
    """Copy weights from PyTorch ActionModule to MLX ActionModule."""
    if config.get("enable_keyboard", True):
        mlx_model.keyboard_embed_linear1.weight = torch_to_mlx(pt_model.keyboard_embed[0].weight)
        mlx_model.keyboard_embed_linear1.bias = torch_to_mlx(pt_model.keyboard_embed[0].bias)
        mlx_model.keyboard_embed_linear2.weight = torch_to_mlx(pt_model.keyboard_embed[2].weight)
        mlx_model.keyboard_embed_linear2.bias = torch_to_mlx(pt_model.keyboard_embed[2].bias)

    if config.get("enable_mouse", True):
        mlx_model.mouse_mlp_linear1.weight = torch_to_mlx(pt_model.mouse_mlp[0].weight)
        mlx_model.mouse_mlp_linear1.bias = torch_to_mlx(pt_model.mouse_mlp[0].bias)
        mlx_model.mouse_mlp_linear2.weight = torch_to_mlx(pt_model.mouse_mlp[2].weight)
        if pt_model.mouse_mlp[2].bias is not None:
            mlx_model.mouse_mlp_linear2.bias = torch_to_mlx(pt_model.mouse_mlp[2].bias)
        mlx_model.mouse_mlp_layernorm.weight = torch_to_mlx(pt_model.mouse_mlp[3].weight)
        mlx_model.mouse_mlp_layernorm.bias = torch_to_mlx(pt_model.mouse_mlp[3].bias)

        mlx_model.t_qkv.weight = torch_to_mlx(pt_model.t_qkv.weight)
        if pt_model.t_qkv.bias is not None:
            mlx_model.t_qkv.bias = torch_to_mlx(pt_model.t_qkv.bias)

        mlx_model.img_attn_q_norm.weight = torch_to_mlx(pt_model.img_attn_q_norm.weight)
        mlx_model.img_attn_k_norm.weight = torch_to_mlx(pt_model.img_attn_k_norm.weight)

        mlx_model.proj_mouse.weight = torch_to_mlx(pt_model.proj_mouse.weight)
        if pt_model.proj_mouse.bias is not None:
            mlx_model.proj_mouse.bias = torch_to_mlx(pt_model.proj_mouse.bias)

    if config.get("enable_keyboard", True):
        mlx_model.key_attn_q_norm.weight = torch_to_mlx(pt_model.key_attn_q_norm.weight)
        mlx_model.key_attn_k_norm.weight = torch_to_mlx(pt_model.key_attn_k_norm.weight)

        mlx_model.mouse_attn_q.weight = torch_to_mlx(pt_model.mouse_attn_q.weight)
        if pt_model.mouse_attn_q.bias is not None:
            mlx_model.mouse_attn_q.bias = torch_to_mlx(pt_model.mouse_attn_q.bias)

        mlx_model.keyboard_attn_kv.weight = torch_to_mlx(pt_model.keyboard_attn_kv.weight)
        if pt_model.keyboard_attn_kv.bias is not None:
            mlx_model.keyboard_attn_kv.bias = torch_to_mlx(pt_model.keyboard_attn_kv.bias)

        mlx_model.proj_keyboard.weight = torch_to_mlx(pt_model.proj_keyboard.weight)
        if pt_model.proj_keyboard.bias is not None:
            mlx_model.proj_keyboard.bias = torch_to_mlx(pt_model.proj_keyboard.bias)


# =============================================================================
# Shared test configuration
# =============================================================================

# Small config: heads_num=4, head_dim=16, rope_dim_list sums to 16
SMALL_CONFIG = dict(
    mouse_dim_in=2,
    keyboard_dim_in=6,
    hidden_size=32,
    img_hidden_size=64,
    keyboard_hidden_dim=64,
    mouse_hidden_dim=64,
    vae_time_compression_ratio=4,
    windows_size=3,
    heads_num=4,
    patch_size=[1, 2, 2],
    qk_norm=True,
    qkv_bias=False,
    rope_dim_list=[4, 6, 6],
    rope_theta=256,
    mouse_qk_dim_list=[4, 6, 6],
    local_attn_size=6,
)

B = 1
tt = 2
th = 2
tw = 2
N_tokens = tt * th * tw  # 8
# N_frames=5: (5-1)/4 + 1 = 2 = N_feats = tt
N_frames = 5


def make_inputs(seed=42, enable_mouse=True, enable_keyboard=True):
    """Create matched NumPy inputs."""
    np.random.seed(seed)
    img_hidden = SMALL_CONFIG["img_hidden_size"]
    x_np = np.random.randn(B, N_tokens, img_hidden).astype(np.float32) * 0.1
    mouse_np = None
    keyboard_np = None
    if enable_mouse:
        mouse_np = np.random.randn(B, N_frames, SMALL_CONFIG["mouse_dim_in"]).astype(np.float32) * 0.1
    if enable_keyboard:
        keyboard_np = np.random.randn(B, N_frames, SMALL_CONFIG["keyboard_dim_in"]).astype(np.float32) * 0.1
    return x_np, mouse_np, keyboard_np


# =============================================================================
# Tests
# =============================================================================

class TestActionModuleMouse:
    """Test mouse conditioning path only."""

    def test_action_module_mouse(self):
        config = {**SMALL_CONFIG, "enable_mouse": True, "enable_keyboard": False}
        x_np, mouse_np, _ = make_inputs(seed=42, enable_mouse=True, enable_keyboard=False)
        # keyboard_condition is needed for the shape assertion in forward
        np.random.seed(99)
        keyboard_np = np.random.randn(B, N_frames, SMALL_CONFIG["keyboard_dim_in"]).astype(np.float32) * 0.1

        pt_model = ActionModule_pt(**config)
        pt_model.eval()

        mlx_model = ActionModuleMLX(**config)
        copy_weights_pt_to_mlx(pt_model, mlx_model, config)

        with torch.no_grad():
            out_pt = pt_model(
                torch.from_numpy(x_np), tt, th, tw,
                mouse_condition=torch.from_numpy(mouse_np),
                keyboard_condition=torch.from_numpy(keyboard_np),
            )

        out_mlx = mlx_model(
            mx.array(x_np), tt, th, tw,
            mouse_condition=mx.array(mouse_np),
            keyboard_condition=mx.array(keyboard_np),
        )

        assert_close(out_mlx, out_pt, atol=1e-3, rtol=1e-3,
                     msg="ActionModule mouse-only path")


class TestActionModuleKeyboard:
    """Test keyboard conditioning path only."""

    def test_action_module_keyboard(self):
        config = {**SMALL_CONFIG, "enable_mouse": False, "enable_keyboard": True}
        x_np, _, _ = make_inputs(seed=42, enable_mouse=False, enable_keyboard=False)
        np.random.seed(99)
        keyboard_np = np.random.randn(B, N_frames, SMALL_CONFIG["keyboard_dim_in"]).astype(np.float32) * 0.1

        pt_model = ActionModule_pt(**config)
        pt_model.eval()

        mlx_model = ActionModuleMLX(**config)
        copy_weights_pt_to_mlx(pt_model, mlx_model, config)

        with torch.no_grad():
            out_pt = pt_model(
                torch.from_numpy(x_np), tt, th, tw,
                mouse_condition=None,
                keyboard_condition=torch.from_numpy(keyboard_np),
            )

        out_mlx = mlx_model(
            mx.array(x_np), tt, th, tw,
            mouse_condition=None,
            keyboard_condition=mx.array(keyboard_np),
        )

        assert_close(out_mlx, out_pt, atol=1e-3, rtol=1e-3,
                     msg="ActionModule keyboard-only path")


class TestActionModuleFull:
    """Test both mouse and keyboard paths together."""

    def test_action_module_full(self):
        config = {**SMALL_CONFIG, "enable_mouse": True, "enable_keyboard": True}
        x_np, mouse_np, keyboard_np = make_inputs(seed=42, enable_mouse=True, enable_keyboard=True)

        pt_model = ActionModule_pt(**config)
        pt_model.eval()

        mlx_model = ActionModuleMLX(**config)
        copy_weights_pt_to_mlx(pt_model, mlx_model, config)

        with torch.no_grad():
            out_pt = pt_model(
                torch.from_numpy(x_np), tt, th, tw,
                mouse_condition=torch.from_numpy(mouse_np),
                keyboard_condition=torch.from_numpy(keyboard_np),
            )

        out_mlx = mlx_model(
            mx.array(x_np), tt, th, tw,
            mouse_condition=mx.array(mouse_np),
            keyboard_condition=mx.array(keyboard_np),
        )

        assert_close(out_mlx, out_pt, atol=1e-3, rtol=1e-3,
                     msg="ActionModule full (mouse + keyboard) path")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
